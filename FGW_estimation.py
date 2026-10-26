from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import torch


@dataclass
class Profile:
    nodes: np.ndarray
    attributes: np.ndarray
    geometry: np.ndarray


def build_profile(events, node_features, edge_features, cutoff, beta=1.0, scales=3, probes=32, seed=0):
    chosen = events[events.ts < cutoff]
    if len(chosen) == 0:
        raise ValueError("The profile needs at least one event before cutoff")
    nodes = np.unique(np.concatenate((chosen.u, chosen.i))).astype(np.int64)
    n = len(nodes)
    src = np.searchsorted(nodes, chosen.u)
    dst = np.searchsorted(nodes, chosen.i)
    start = float(chosen.ts.min())
    span = max(float(cutoff) - start, 1e-12)
    weights = np.exp(-beta * (float(cutoff) - chosen.ts.to_numpy(dtype=np.float64)) / span)
    edge_index = chosen.idx.to_numpy(dtype=np.int64)
    values = edge_features[edge_index].astype(np.float32)
    width = node_features.shape[1]
    if values.shape[1] != width:
        if values.shape[1] > width:
            values = values[:, :width]
        else:
            values = np.pad(values, ((0, 0), (0, width - values.shape[1])))
    sums = np.zeros((n, width), dtype=np.float64)
    mass = np.zeros(n, dtype=np.float64)
    np.add.at(sums, src, values * weights[:, None])
    np.add.at(sums, dst, values * weights[:, None])
    np.add.at(mass, src, weights)
    np.add.at(mass, dst, weights)
    historical = sums / np.maximum(mass[:, None], 1e-12)
    static = node_features[nodes].astype(np.float64)
    attributes = np.where(np.any(static != 0, axis=1, keepdims=True), static, historical).astype(np.float32)
    adjacency = sp.coo_matrix((np.tile(weights, 2), (np.concatenate((src, dst)), np.concatenate((dst, src)))), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = sp.eye(n, format="csr") - sp.diags(inverse) @ adjacency @ sp.diags(inverse)
    generator = np.random.default_rng(seed)
    sketch = generator.standard_normal((n, min(probes, n))).astype(np.float64) / np.sqrt(min(probes, n))
    times = np.geomspace(0.1, 10.0, scales)
    geometry = np.concatenate([expm_multiply(-float(t) * laplacian, sketch) for t in times], axis=1)
    geometry = (geometry / np.sqrt(scales)).astype(np.float32)
    return Profile(nodes, attributes, geometry)


def _balanced(logits, rows, columns, iterations=12):
    matrix = torch.exp(logits - logits.max()).clamp_min(1e-30)
    for _ in range(iterations):
        matrix = matrix * (rows / matrix.sum(dim=1).clamp_min(1e-30))[:, None]
        matrix = matrix * (columns / matrix.sum(dim=0).clamp_min(1e-30))[None, :]
    matrix = matrix * (rows / matrix.sum(dim=1).clamp_min(1e-30))[:, None]
    return matrix


def _structural_factor(features, rank):
    indices = torch.linspace(0, len(features) - 1, min(rank, len(features)), device=features.device).long()
    landmarks = features[indices]
    columns = torch.cdist(features, landmarks)
    center = columns[indices]
    inverse = torch.linalg.pinv(center, hermitian=True)
    return columns, inverse


def _structural_transport(columns, inverse, factors):
    projected = columns.T @ factors
    return projected.T @ inverse @ projected


class Transport:
    def __init__(self, downstream_nodes, pretrain_nodes, q, r):
        self.downstream_nodes = downstream_nodes
        self.pretrain_nodes = pretrain_nodes
        self.q = q
        self.r = r
        self.g = q.sum(axis=0)

    def reference(self, pretrain_attributes):
        pooled = self.r.T @ pretrain_attributes
        return (self.q / np.maximum(self.g[None, :], 1e-12)) @ pooled * len(self.downstream_nodes)

    def alignment_target(self, pretrain_attributes):
        return self.reference(pretrain_attributes)


def estimate_fgw(downstream, pretrain, alpha=0.5, rank=16, iterations=150, lr=0.05, entropy=0.01, device="cpu", seed=0):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    torch.manual_seed(seed)
    device = torch.device(device)
    xd = torch.as_tensor(downstream.attributes, device=device)
    xp = torch.as_tensor(pretrain.attributes, device=device)
    fd = torch.as_tensor(downstream.geometry, device=device)
    fp = torch.as_tensor(pretrain.geometry, device=device)
    cd, idown = _structural_factor(fd, 32)
    cp, ipre = _structural_factor(fp, 32)
    n, m = len(xd), len(xp)
    rank = min(rank, n, m)
    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((m,), 1.0 / m, device=device)
    g = torch.full((rank,), 1.0 / rank, device=device)
    q_logits = torch.nn.Parameter(torch.randn(n, rank, device=device) * 0.01)
    r_logits = torch.nn.Parameter(torch.randn(m, rank, device=device) * 0.01)
    optimizer = torch.optim.Adam((q_logits, r_logits), lr=lr)
    attribute_scale = torch.cat((xd, xp)).square().sum(dim=1).mean().clamp_min(1e-8)
    structure_scale = (cd.mean() * cp.mean()).clamp_min(1e-8)
    for _ in range(iterations):
        optimizer.zero_grad()
        q = _balanced(q_logits, a, g)
        r = _balanced(r_logits, b, g)
        xq = xd.T @ q
        xr = xp.T @ r
        attribute = -2.0 * ((xq * xr) / g).sum() / attribute_scale
        cq = _structural_transport(cd, idown, q)
        cr = _structural_transport(cp, ipre, r)
        structure = -2.0 * ((cq * cr) / (g[:, None] * g[None, :])).sum() / structure_scale
        regularizer = (q * q.clamp_min(1e-30).log()).sum() + (r * r.clamp_min(1e-30).log()).sum()
        loss = (1.0 - alpha) * attribute + alpha * structure + entropy * regularizer
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        q = _balanced(q_logits, a, g, 50).cpu().numpy()
        r = _balanced(r_logits, b, g, 50).cpu().numpy()
    return Transport(downstream.nodes, pretrain.nodes, q, r)
