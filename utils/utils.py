import numpy as np
import torch

class MergeLayer(torch.nn.Module):

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for (source, destination, edge_idx, timestamp) in zip(data.sources, data.destinations, data.edge_idxs, data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
    return NeighborFinder(adj_list, uniform=uniform)

class NeighborFinder:

    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        for neighbors in adj_list:
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
        self.uniform = uniform
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i])

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        assert len(source_nodes) == len(timestamps)
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int32)
        for (i, (source_node, timestamp)) in enumerate(zip(source_nodes, timestamps)):
            (source_neighbors, source_edge_idxs, source_edge_times) = self.find_before(source_node, timestamp)
            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]
                    assert len(source_neighbors) <= n_neighbors
                    assert len(source_edge_times) <= n_neighbors
                    assert len(source_edge_idxs) <= n_neighbors
                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
        return (neighbors, edge_idxs, edge_times)
