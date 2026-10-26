import json

import numpy as np
import torch

from backbone import create_backbone, negatives, probabilities


def pretrain(args, dataset, output):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    model = create_backbone(dataset, device, args.neighbors, dataset.pretrain)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    candidates = np.unique(dataset.pretrain.i.to_numpy(dtype=np.int64))
    for epoch in range(args.pretrain_epochs):
        model.train()
        model.memory.__init_memory__()
        losses = []
        for start in range(0, len(dataset.pretrain), args.batch_size):
            batch = dataset.pretrain.iloc[start:start + args.batch_size]
            negative = negatives(batch, candidates, generator)
            optimizer.zero_grad()
            positive_probability, negative_probability = probabilities(model, batch, negative, args.neighbors)
            loss = torch.nn.functional.binary_cross_entropy(positive_probability.flatten(), torch.ones(len(batch), device=device))
            loss = loss + torch.nn.functional.binary_cross_entropy(negative_probability.flatten(), torch.zeros(len(batch), device=device))
            loss.backward()
            optimizer.step()
            model.memory.detach_memory()
            losses.append(float(loss.detach()))
        print(json.dumps({"stage": "pretrain", "epoch": epoch + 1, "loss": float(np.mean(losses))}))
    path = output / f"tgn_{args.data}.pt"
    torch.save(model.state_dict(), path)
    print(path)
