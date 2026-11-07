"""The three stages of LightGFC, using supervision from training nodes only."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph import normalized_adjacency, sparse_tensor
from .models import MLP


def proto_features(x, adjacency, hops, device):
    """Average X, A_hat X, ..., A_hat^(K-1) X (Eq. 2–3)."""
    propagated = x.to(device)
    if hops == 1:
        return propagated
    operator = sparse_tensor(normalized_adjacency(adjacency), device)
    total = propagated.clone()
    for _ in range(1, hops):
        propagated = torch.sparse.mm(operator, propagated)
        total.add_(propagated)
    return total.div_(hops)


def class_budgets(counts, weights, budget):
    """Allocate an exact integer budget with one node per training class."""
    if not len(counts) <= budget <= counts.sum():
        raise ValueError(
            f"Budget {budget} must be between the number of training classes "
            f"({len(counts)}) and training nodes ({counts.sum()})"
        )
    allocation = np.ones(len(counts), dtype=np.int64)
    remaining = budget - len(counts)
    while remaining:
        capacity = counts - allocation
        eligible_weights = np.where(capacity > 0, weights, 0)
        if not eligible_weights.any():
            eligible_weights = capacity
        quota = remaining * eligible_weights / eligible_weights.sum()
        whole = np.minimum(quota.astype(np.int64), capacity)
        allocation += whole
        remaining -= whole.sum()

        eligible = np.flatnonzero(allocation < counts)
        order = np.argsort(-(quota[eligible] % 1), kind="stable")
        extras = eligible[order[:remaining]]
        allocation[extras] += 1
        remaining -= len(extras)
    return allocation


def pretrain_mlp(x, y, num_classes, config):
    """Fit the two-layer, dropout-free MLP on labeled training features."""
    model = MLP(x.size(1), config.hidden, num_classes).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_mlp, weight_decay=5e-4)
    for _ in range(config.mlp_epochs):
        order = torch.randperm(len(y), device=y.device)
        for batch in order.split(config.mlp_batch_size):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[batch]), y[batch]).backward()
            optimizer.step()
    return model.eval().requires_grad_(False)


def initialize_nodes(raw, proto, labels, budget, num_classes, variant, seed):
    """Partition each class into seeded groups and apply the class-to-node weights."""
    counts = torch.bincount(labels, minlength=num_classes)
    if variant == "no-similarity":
        scores = torch.ones(len(labels), device=labels.device)
    else:
        scores = (1 - F.cosine_similarity(raw, proto, dim=1)).clamp_min_(0)
    class_weights = scores.new_zeros(num_classes).index_add_(0, labels, scores)
    if not class_weights.any():
        scores = torch.ones_like(scores)
        class_weights = counts.to(scores.dtype)

    budgets = class_budgets(counts.cpu().numpy(), class_weights.cpu().numpy(), budget)
    rng = np.random.default_rng(seed)
    features = []
    for label, count in enumerate(budgets):
        members = (labels == label).nonzero(as_tuple=True)[0]
        members = members[torch.as_tensor(rng.permutation(len(members)), device=members.device)]
        weights = scores[members]
        total = class_weights[label]
        if total == 0:
            weights = torch.ones_like(weights)
            total = weights.sum()
        weighted = proto[members] * (weights / total).unsqueeze(1)
        features.extend(group.sum(dim=0) for group in torch.tensor_split(weighted, int(count)))

    repeats = torch.from_numpy(budgets).to(labels.device)
    synthetic_labels = torch.arange(num_classes, device=labels.device).repeat_interleave(repeats)
    return torch.stack(features), synthetic_labels


def project_nodes(initial, labels, raw, targets, num_classes, teacher, config):
    """Fit an identity-initialized node projection using the enabled losses (Eq. 7–10)."""
    alpha = config.alpha if teacher is not None else 0
    beta = config.beta if config.variant not in {"no-prototype", "no-both"} else 0
    if alpha == beta == 0:
        return initial

    projector = nn.Parameter(torch.eye(len(initial), device=initial.device))
    optimizer = torch.optim.Adam([projector], lr=config.lr_projector)
    if beta:
        original_counts = torch.bincount(targets, minlength=num_classes).unsqueeze(1)
        synthetic_counts = torch.bincount(labels, minlength=num_classes).unsqueeze(1)
        prototypes = raw.new_zeros(num_classes, raw.size(1))
        prototypes.index_add_(0, targets, raw).div_(original_counts)

    for _ in range(config.projector_epochs):
        optimizer.zero_grad(set_to_none=True)
        features = projector @ initial
        loss = alpha * F.cross_entropy(teacher(features), labels) if alpha else 0
        if beta:
            means = features.new_zeros(num_classes, features.size(1))
            means.index_add_(0, labels, features).div_(synthetic_counts)
            loss = loss + beta * (means - prototypes).square().sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return projector @ initial


def condense(benchmark, config, budget, device, seed):
    x, adjacency, indices = benchmark.graph("train")
    features = proto_features(x, adjacency, config.hops, device)
    if benchmark.inductive:
        raw = x.to(device)
    else:
        raw = x[indices].to(device)
        features = features[indices.to(device)]
    labels = benchmark.labels("train").to(device)
    num_classes = benchmark.num_classes

    teacher = None
    if config.variant not in {"no-label", "no-both"} and config.alpha:
        teacher = pretrain_mlp(features, labels, num_classes, config)
    initial, synthetic_labels = initialize_nodes(
        raw, features, labels, budget, num_classes, config.variant, seed,
    )
    synthetic_features = project_nodes(initial, synthetic_labels, raw, labels, num_classes, teacher, config)
    return synthetic_features, synthetic_labels
