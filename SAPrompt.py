import numpy as np
import torch
from torch import nn


class SAPrompt(nn.Module):
    def __init__(self, node_features, pretrain_profile, downstream_profile, transport, hidden=64, time_dimension=16):
        super().__init__()
        base = np.asarray(node_features, dtype=np.float32).copy()
        history = base.copy()
        reference = base.copy()
        active = np.zeros(len(base), dtype=np.float32)
        history[downstream_profile.nodes] = downstream_profile.attributes
        reference[downstream_profile.nodes] = transport.reference(pretrain_profile.attributes)
        active[downstream_profile.nodes] = 1.0
        base[downstream_profile.nodes] = downstream_profile.attributes
        self.register_buffer("base", torch.from_numpy(base))
        self.register_buffer("history", torch.from_numpy(history))
        self.register_buffer("reference", torch.from_numpy(reference.astype(np.float32)))
        self.register_buffer("active", torch.from_numpy(active[:, None]))
        self.register_buffer("frequency", torch.logspace(0, -4, time_dimension // 2))
        width = base.shape[1]
        self.gate = nn.Sequential(nn.Linear(2 * width + time_dimension, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, timestamp, time_origin, time_scale):
        phase = ((float(timestamp) - time_origin) / max(time_scale, 1e-12)) * self.frequency
        encoding = torch.cat((phase.cos(), phase.sin())).expand(len(self.base), -1)
        inputs = torch.cat((self.base, self.history, encoding), dim=1)
        rho = torch.sigmoid(self.gate(inputs)) * self.active
        prompted = self.base + rho * (self.reference - self.base)
        return prompted, rho

    def alignment_loss(self, prompted, node_ids):
        ids = torch.as_tensor(np.unique(node_ids), dtype=torch.long, device=prompted.device)
        valid = self.active[ids, 0] > 0
        if not valid.any():
            return prompted.sum() * 0
        differences = prompted[ids[valid]] - self.reference[ids[valid]]
        return differences.square().sum(dim=1).mean()
