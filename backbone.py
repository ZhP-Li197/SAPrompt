import numpy as np
import torch

from TGN.tgn import TGN
from utils.data_processing import Data, compute_time_statistics
from utils.utils import get_neighbor_finder

from data import event_arrays


def as_tgn_data(events):
    source, destination, time, edge, label = event_arrays(events)
    return Data(source, destination, time, edge, label)


def finder(events, max_node):
    return get_neighbor_finder(as_tgn_data(events), False, max_node_idx=max_node)


def create_backbone(dataset, device, neighbors, history):
    shifts = compute_time_statistics(dataset.pretrain.u.to_numpy(), dataset.pretrain.i.to_numpy(), dataset.pretrain.ts.to_numpy())
    model = TGN(
        neighbor_finder=finder(history, len(dataset.node_features) - 1),
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        message_dimension=100,
        memory_dimension=dataset.node_features.shape[1],
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        memory_updater_type="gru",
        n_neighbors=neighbors,
        mean_time_shift_src=shifts[0],
        std_time_shift_src=max(shifts[1], 1e-6),
        mean_time_shift_dst=shifts[2],
        std_time_shift_dst=max(shifts[3], 1e-6),
    ).to(device)
    model.original_node_features = torch.as_tensor(dataset.node_features, device=device)
    return model


def negatives(events, candidates, generator):
    sampled = generator.choice(candidates, len(events), replace=True)
    positive = events.i.to_numpy(dtype=np.int64)
    equal = sampled == positive
    while equal.any():
        sampled[equal] = generator.choice(candidates, int(equal.sum()), replace=True)
        equal = sampled == positive
    return sampled.astype(np.int64)


def probabilities(backbone, events, negative, neighbors):
    source, destination, time, edge, _ = event_arrays(events)
    return backbone.compute_edge_probabilities(source, destination, negative, time, edge, neighbors)
