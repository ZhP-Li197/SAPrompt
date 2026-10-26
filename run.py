import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

from FGW_estimation import build_profile, estimate_fgw
from SAPrompt import SAPrompt
from backbone import create_backbone, finder, negatives, probabilities
from data import event_arrays, load_dataset, sampled_support
from pretraining import pretrain


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("pretrain", "adapt"))
    p.add_argument("--data", default="wikipedia")
    p.add_argument("--data-root", default=str(ROOT / "Data"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=str(ROOT / "code" / "results"))
    p.add_argument("--task", choices=("link", "node"), default="link")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--scales", type=int, default=3)
    p.add_argument("--probes", type=int, default=32)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--fgw-iterations", type=int, default=150)
    p.add_argument("--alignment", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def node_logits(backbone, head, events, negative, neighbors):
    source, destination, time, edge, _ = event_arrays(events)
    embeddings = backbone.compute_temporal_embeddings(source, destination, negative, time, edge, neighbors)[0]
    return head(embeddings).flatten()


def build_prompt(args, dataset, pretrain_profile, history, cutoff, gate=None):
    downstream = build_profile(history, dataset.node_features, dataset.edge_features, cutoff, args.beta, args.scales, args.probes, args.seed)
    coupling = estimate_fgw(downstream, pretrain_profile, args.alpha, args.rank, args.fgw_iterations, device=args.device, seed=args.seed)
    prompt = SAPrompt(dataset.node_features, pretrain_profile, downstream, coupling).to(args.device)
    if gate is not None:
        prompt.gate = gate
    return prompt


def inject(backbone, prompt, timestamp, origin, scale):
    prompted, _ = prompt(timestamp, origin, scale)
    backbone.node_raw_features = prompted
    backbone.embedding_module.node_features = prompted
    return prompted


def warmup(backbone, events, args):
    candidates = np.unique(events.i.to_numpy(dtype=np.int64))
    generator = np.random.default_rng(args.seed)
    with torch.no_grad():
        for start in range(0, len(events), args.batch_size):
            batch = events.iloc[start:start + args.batch_size]
            negative = negatives(batch, candidates, generator)
            probabilities(backbone, batch, negative, args.neighbors)


def run_events(backbone, prompt, events, args, head, support=None, optimizer=None, seed=0, return_records=False, candidate_nodes=None):
    device = torch.device(args.device)
    generator = np.random.default_rng(seed)
    candidates = np.unique(events.i.to_numpy(dtype=np.int64)) if candidate_nodes is None else candidate_nodes
    if len(candidates) < 2:
        candidates = np.arange(1, len(prompt.base), dtype=np.int64)
    selected = set() if support is None else set(support.idx.to_numpy(dtype=np.int64))
    first_support_time = None if support is None else float(support.ts.min())
    origin = float(events.ts.min())
    scale = max(float(events.ts.max()) - origin, 1e-12)
    scores, labels, losses, records = [], [], [], []
    position = 0
    while position < len(events):
        selected_here = optimizer is not None and int(events.iloc[position].idx) in selected
        if selected_here:
            end = position + 1
        else:
            end = min(position + args.batch_size, len(events))
            if optimizer is not None:
                for j in range(position + 1, end):
                    if int(events.iloc[j].idx) in selected:
                        end = j
                        break
        batch = events.iloc[position:end]
        negative = negatives(batch, candidates, generator)
        early_history = first_support_time is not None and float(batch.ts.max()) < first_support_time
        if selected_here:
            optimizer.zero_grad()
            prompted = inject(backbone, prompt, float(batch.ts.iloc[0]), origin, scale)
            if args.task == "link":
                positive, negative_probability = probabilities(backbone, batch, negative, args.neighbors)
                loss = torch.nn.functional.binary_cross_entropy(positive.flatten(), torch.ones(len(batch), device=device))
                loss = loss + torch.nn.functional.binary_cross_entropy(negative_probability.flatten(), torch.zeros(len(batch), device=device))
            else:
                logits = node_logits(backbone, head, batch, negative, args.neighbors)
                targets = torch.as_tensor(batch.label.to_numpy(dtype=np.float32), device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            ids = np.concatenate((batch.u.to_numpy(), batch.i.to_numpy()))
            loss = loss + args.alignment * prompt.alignment_loss(prompted, ids)
            loss.backward()
            optimizer.step()
            backbone.memory.detach_memory()
            losses.append(float(loss.detach()))
        else:
            with torch.no_grad():
                if early_history:
                    backbone.node_raw_features = backbone.original_node_features
                    backbone.embedding_module.node_features = backbone.node_raw_features
                else:
                    inject(backbone, prompt, float(batch.ts.iloc[0]), origin, scale)
                if args.task == "link":
                    positive, negative_probability = probabilities(backbone, batch, negative, args.neighbors)
                    positive_values = positive.flatten().cpu().tolist()
                    negative_values = negative_probability.flatten().cpu().tolist()
                    scores.extend(positive_values)
                    scores.extend(negative_values)
                    labels.extend([1] * len(batch))
                    labels.extend([0] * len(batch))
                    records.extend((int(u), int(i), float(p), float(q)) for u, i, p, q in zip(batch.u, batch.i, positive_values, negative_values))
                else:
                    logits = node_logits(backbone, head, batch, negative, args.neighbors)
                    scores.extend(logits.sigmoid().cpu().tolist())
                    labels.extend(batch.label.to_numpy(dtype=np.int64).tolist())
        position = end
    metrics = {"loss": float(np.mean(losses)) if losses else None, "events": len(events)}
    if len(set(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
    if return_records:
        metrics["records"] = records
    return metrics


def subset_metrics(records, seen, inductive):
    chosen = [(p, q) for u, i, p, q in records if ((u not in seen or i not in seen) == inductive)]
    if not chosen:
        return {"events": 0}
    scores = [p for p, _ in chosen] + [q for _, q in chosen]
    labels = [1] * len(chosen) + [0] * len(chosen)
    return {"events": len(chosen), "auc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores))}


def adapt(args, dataset, output):
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / f"tgn_{args.data}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    backbone = create_backbone(dataset, device, args.neighbors, dataset.pretrain)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    backbone.load_state_dict(state)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    support = sampled_support(dataset.train, args.shots, args.seed)
    pretrain_cutoff = float(dataset.train.ts.min())
    pretrain_profile = build_profile(dataset.pretrain, dataset.node_features, dataset.edge_features, pretrain_cutoff, args.beta, args.scales, args.probes, args.seed)
    train_cutoff = float(support.ts.min())
    train_history = dataset.events[dataset.events.ts < train_cutoff]
    train_prompt = build_prompt(args, dataset, pretrain_profile, train_history, train_cutoff)
    head = torch.nn.Linear(dataset.node_features.shape[1], 1).to(device) if args.task == "node" else None
    parameters = list(train_prompt.gate.parameters()) + ([] if head is None else list(head.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-5)
    combined_train = dataset.events.iloc[:len(dataset.pretrain) + len(dataset.train)]
    train_finder = finder(combined_train, len(dataset.node_features) - 1)
    validation_prompt = build_prompt(args, dataset, pretrain_profile, combined_train, float(dataset.validation.ts.min()), train_prompt.gate)
    candidate_nodes = np.unique(dataset.train.i.to_numpy(dtype=np.int64))
    best_score, best_gate, best_head = -np.inf, None, None
    for epoch in range(args.epochs):
        backbone.memory.__init_memory__()
        backbone.set_neighbor_finder(train_finder)
        train_prompt.train()
        if head is not None:
            head.train()
        backbone.node_raw_features = torch.as_tensor(dataset.node_features, device=device)
        backbone.embedding_module.node_features = backbone.node_raw_features
        warmup(backbone, dataset.pretrain, args)
        training = run_events(backbone, train_prompt, dataset.train, args, head, support, optimizer, args.seed + epoch, candidate_nodes=candidate_nodes)
        memory_snapshot = backbone.memory.backup_memory()
        backbone.set_neighbor_finder(finder(dataset.events.iloc[:len(combined_train) + len(dataset.validation)], len(dataset.node_features) - 1))
        train_prompt.eval()
        if head is not None:
            head.eval()
        validation = run_events(backbone, validation_prompt, dataset.validation, args, head, seed=args.seed + 100, candidate_nodes=candidate_nodes)
        backbone.memory.restore_memory(memory_snapshot)
        score = validation.get("auc", -training["loss"])
        if score > best_score:
            best_score = score
            best_gate = {key: value.detach().cpu().clone() for key, value in train_prompt.gate.state_dict().items()}
            best_head = None if head is None else {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        print(json.dumps({"stage": "adapt", "epoch": epoch + 1, "train": training, "validation": validation}))
    train_prompt.gate.load_state_dict(best_gate)
    if head is not None:
        head.load_state_dict(best_head)
    backbone.memory.__init_memory__()
    test_history = dataset.events.iloc[:len(dataset.pretrain) + len(dataset.train) + len(dataset.validation)]
    test_prompt = build_prompt(args, dataset, pretrain_profile, test_history, float(dataset.test.ts.min()), train_prompt.gate)
    backbone.set_neighbor_finder(finder(dataset.events, len(dataset.node_features) - 1))
    backbone.node_raw_features = torch.as_tensor(dataset.node_features, device=device)
    backbone.embedding_module.node_features = backbone.node_raw_features
    with torch.no_grad():
        warmup(backbone, dataset.pretrain, args)
        run_events(backbone, train_prompt, dataset.train, args, head, support=support, seed=args.seed, candidate_nodes=candidate_nodes)
        run_events(backbone, validation_prompt, dataset.validation, args, head, seed=args.seed + 100, candidate_nodes=candidate_nodes)
        all_test = run_events(backbone, test_prompt, dataset.test, args, head, seed=args.seed + 200, return_records=True, candidate_nodes=candidate_nodes)
    seen = set(dataset.pretrain.u) | set(dataset.pretrain.i) | set(dataset.train.u) | set(dataset.train.i)
    records = all_test.pop("records")
    result = {"all_test": all_test, "best_validation_score": best_score}
    if args.task == "link":
        result["transductive"] = subset_metrics(records, seen, False)
        result["inductive"] = subset_metrics(records, seen, True)
    torch.save({"gate": best_gate, "head": best_head, "args": vars(args)}, output / f"saprompt_{args.task}_{args.data}.pt")
    with open(output / f"saprompt_{args.task}_{args.data}.json", "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


def main():
    args = parser()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_root, args.data)
    if args.stage == "pretrain":
        pretrain(args, dataset, output)
    else:
        adapt(args, dataset, output)


if __name__ == "__main__":
    main()
