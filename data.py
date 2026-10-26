from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    events: pd.DataFrame
    node_features: np.ndarray
    edge_features: np.ndarray
    pretrain: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(root, name):
    folder = Path(root) / name
    events = pd.read_csv(folder / f"ml_{name}.csv").sort_values(["ts", "idx"], kind="stable").reset_index(drop=True)
    required = {"u", "i", "ts", "idx", "label"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(events.columns))}")
    nodes = np.load(folder / f"ml_{name}_node.npy").astype(np.float32)
    edges = np.load(folder / f"ml_{name}.npy").astype(np.float32)
    if events[["u", "i"]].to_numpy().min() < 0 or events[["u", "i"]].to_numpy().max() >= len(nodes):
        raise ValueError("Node IDs exceed the node feature array")
    if events.idx.min() < 0 or events.idx.max() >= len(edges):
        raise ValueError("Edge IDs exceed the edge feature array")
    if len(events) < 100:
        raise ValueError("At least 100 chronological events are required")
    n = len(events)
    p0, p1, p2 = int(0.80 * n), int(0.81 * n), int(0.82 * n)
    return Dataset(events, nodes, edges, events.iloc[:p0].copy(), events.iloc[p0:p1].copy(), events.iloc[p1:p2].copy(), events.iloc[p2:].copy())


def sampled_support(events, shots, seed):
    count = min(shots, len(events))
    positions = np.sort(np.random.default_rng(seed).choice(len(events), count, replace=False))
    return events.iloc[positions].copy()


def event_arrays(events):
    return (
        events.u.to_numpy(dtype=np.int64),
        events.i.to_numpy(dtype=np.int64),
        events.ts.to_numpy(dtype=np.float64),
        events.idx.to_numpy(dtype=np.int64),
        events.label.to_numpy(dtype=np.float32),
    )


def negative_destinations(events, candidates, seed):
    generator = np.random.default_rng(seed)
    positives = set(zip(events.u.to_numpy(), events.i.to_numpy()))
    output = np.empty(len(events), dtype=np.int64)
    for position, source in enumerate(events.u.to_numpy()):
        for _ in range(100):
            candidate = int(generator.choice(candidates))
            if (int(source), candidate) not in positives:
                break
        output[position] = candidate
    return output
