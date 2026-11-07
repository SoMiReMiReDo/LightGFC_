"""Train on the graph-free artifact, then select by validation accuracy."""

import torch
from torch.nn import functional as F

from .graph import identity_adjacency, normalized_adjacency, sparse_tensor
from .models import build_model


def prepare_graph(x, adjacency, model_name, device):
    x = x.to(device)
    if model_name == "mlp":
        return x, None
    if model_name in {"gcn", "sgc", "jknet"}:
        adjacency = normalized_adjacency(adjacency)
    operator = sparse_tensor(adjacency, device)
    if model_name == "sgc":
        return torch.sparse.mm(operator, torch.sparse.mm(operator, x)), None
    return x, operator


@torch.no_grad()
def accuracy(model, graph, indices, labels):
    model.eval()
    pred = model(*graph)[indices].argmax(dim=1)
    return (pred == labels).float().mean().item()


def evaluate(benchmark, features, labels, config, device):
    model = build_model(config.model, features.size(1), config.hidden, benchmark.num_classes, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    synthetic_adj = None if config.model in {"mlp", "sgc"} else identity_adjacency(len(features), device)
    val_x, val_adj, val_idx = benchmark.graph("val")
    val_graph = prepare_graph(val_x, val_adj, config.model, device)
    val_idx = val_idx.to(device)
    val_labels = benchmark.labels("val").to(device)

    best_accuracy = -1
    for epoch in range(1, config.epochs + 1):
        if epoch == config.epochs // 2:
            for group in optimizer.param_groups:
                group["lr"] = config.lr * 0.1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features, synthetic_adj), labels).backward()
        optimizer.step()

        if epoch == 1 or epoch == config.epochs or epoch % config.eval_every == 0:
            val_accuracy = accuracy(model, val_graph, val_idx, val_labels)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_x, test_adj, test_idx = benchmark.graph("test")
    if benchmark.inductive:
        del val_graph
        test_graph = prepare_graph(test_x, test_adj, config.model, device)
    else:
        test_graph = val_graph
    test_labels = benchmark.labels("test").to(device)
    test_accuracy = accuracy(model, test_graph, test_idx.to(device), test_labels)
    return best_epoch, best_accuracy, test_accuracy
