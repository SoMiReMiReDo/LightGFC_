"""Sparse adjacency operations shared by condensation and evaluation."""

import numpy as np
import scipy.sparse as sp
import torch


def normalized_adjacency(adjacency):
    """Return D^(-1/2) (A + I) D^(-1/2) as CSR."""
    with_loops = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(with_loops.sum(axis=1)).ravel()
    scale = sp.diags(degree ** -0.5, format="csr")
    return (scale @ with_loops @ scale).tocsr()


def sparse_tensor(adjacency, device):
    adjacency.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(adjacency.indptr).to(device),
        torch.from_numpy(adjacency.indices).to(device),
        torch.from_numpy(adjacency.data).to(device),
        size=adjacency.shape,
    )


def identity_adjacency(num_nodes, device):
    # Downstream message passing sees only self-connections on synthetic data.
    index = torch.arange(num_nodes, dtype=torch.int32, device=device)
    rows = torch.arange(num_nodes + 1, dtype=torch.int32, device=device)
    return torch.sparse_csr_tensor(rows, index, torch.ones(num_nodes, device=device),
                                   size=(num_nodes, num_nodes))
