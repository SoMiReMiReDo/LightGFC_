"""Load benchmark splits and construct sparse graph views."""

from dataclasses import dataclass
from functools import cached_property
import json

from filelock import FileLock
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Flickr, Planetoid, Reddit

from .config import DATASETS


PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer"}
LEGACY = {"arxiv": "ogbn-arxiv", "flickr": "flickr", "reddit": "reddit"}
LEGACY_FILES = ("adj_full.npz", "feats.npy", "role.json", "class_map.json")
SPLITS = ("train", "val", "test")


@dataclass
class Benchmark:
    x: torch.Tensor
    y: torch.Tensor
    adjacency: sp.csr_matrix
    splits: dict
    source: str
    inductive: bool

    @cached_property
    def train_classes(self):
        return self.y[self.splits["train"]].unique(sorted=True)

    @property
    def num_classes(self):
        return len(self.train_classes)

    def labels(self, split):
        labels = self.y[self.splits[split]]
        indices = torch.searchsorted(self.train_classes, labels).clamp_max(self.num_classes - 1)
        # A class absent from training cannot match any downstream prediction.
        return torch.where(self.train_classes[indices] == labels, indices, -1)

    def graph(self, split):
        """Return the features, adjacency and supervised indices for a split."""
        indices = self.splits[split]
        if not self.inductive:
            return self.x, self.adjacency, indices
        nodes = indices.numpy()
        return self.x[indices], self.adjacency[nodes][:, nodes], torch.arange(len(nodes))


def _read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _legacy_labels(mapping, num_nodes):
    try:
        labels = np.asarray([mapping[str(node)] for node in range(num_nodes)])
    except KeyError as exc:
        raise ValueError(f"Missing label for node {exc.args[0]} in class_map.json") from exc
    if labels.ndim == 2:
        if not np.isin(labels, [0, 1]).all() or not (labels.sum(axis=1) == 1).all():
            raise ValueError("Legacy class_map.json must contain single-label or one-hot targets")
        labels = labels.argmax(axis=1)
    return torch.as_tensor(labels, dtype=torch.long)


def _load_legacy(folder):
    missing = [name for name in LEGACY_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete legacy dataset at {folder}. Missing: {', '.join(missing)}")
    adjacency = sp.load_npz(folder / "adj_full.npz").tocsr().astype(np.float32)
    role = _read_json(folder / "role.json")
    splits = {
        split: torch.tensor(role[key], dtype=torch.long)
        for split, key in zip(SPLITS, ("tr", "va", "te"))
    }
    features = torch.from_numpy(np.load(folder / "feats.npy")).float()
    labels = _legacy_labels(_read_json(folder / "class_map.json"), len(features))
    return features, labels, adjacency, splits


def _load_official(name, root):
    if name in PLANETOID:
        data = Planetoid(str(root / "planetoid"), PLANETOID[name], split="public")[0]
        source = "pyg-planetoid-public"
    elif DATASETS[name].inductive:
        dataset_class = Flickr if name == "flickr" else Reddit
        data = dataset_class(str(root / name))[0]
        source = f"pyg-{name}"
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        cache = root / f"ogbn_{name}"
        alternate_cache = root / f"ogbn_{name}_pyg"
        if alternate_cache.exists():
            cache = alternate_cache
        if cache.exists() and not (cache / "RELEASE_v1.txt").is_file():
            raise ValueError(f"Incomplete or outdated OGB cache at {cache}. Use a fresh --data-dir.")

        # OGB 1.3.6 saves PyG Data objects in its processed cache.
        with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            dataset = PygNodePropPredDataset(name=f"ogbn-{name}", root=str(root))
            data = dataset[0]
            index = dataset.get_idx_split()
        splits = {
            split: index[key].reshape(-1).long()
            for split, key in zip(SPLITS, ("train", "valid", "test"))
        }
        source = f"ogbn-{name}"

    if source.startswith("pyg-"):
        splits = {split: data[f"{split}_mask"].nonzero(as_tuple=True)[0] for split in SPLITS}
    src, dst = data.edge_index.numpy()
    adjacency = sp.csr_matrix(
        (np.ones(len(src), dtype=np.float32), (dst, src)),
        shape=(data.num_nodes, data.num_nodes),
    )
    return data.x.float(), data.y.reshape(-1).long(), adjacency, splits, source


def _validate_splits(splits, num_nodes):
    combined = torch.cat(list(splits.values()))
    if (any(index.ndim != 1 or len(index) == 0 for index in splits.values())
            or combined.min() < 0 or combined.max() >= num_nodes
            or len(combined.unique()) != len(combined)):
        raise ValueError("Train, validation and test indices must be non-empty, disjoint and in range")


def load_benchmark(name, root, source="auto"):
    preset = DATASETS[name]
    folder = root / LEGACY[name] if name in LEGACY else None
    use_legacy = source == "legacy" or (
        source == "auto" and folder is not None
        and any((folder / file).exists() for file in LEGACY_FILES)
    )
    if use_legacy:
        if folder is None:
            raise ValueError(f"Legacy data is supported for {', '.join(LEGACY)}")
        x, y, adjacency, splits = _load_legacy(folder)
        resolved_source = "legacy-graphsaint"
    else:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / f".{name}.lock")):
            x, y, adjacency, splits, resolved_source = _load_official(name, root)

    _validate_splits(splits, len(x))
    if adjacency.shape != (len(x), len(x)) or len(y) != len(x):
        raise ValueError("Features, labels and adjacency must describe the same nodes")

    if y[splits["train"]].min() < 0:
        raise ValueError("Training nodes must have nonnegative class labels")

    if preset.standardize_features:
        scaler = StandardScaler().fit(x[splits["train"]].numpy())
        x = torch.from_numpy(scaler.transform(x.numpy()))

    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.data.fill(1)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    return Benchmark(x, y, adjacency, splits, resolved_source, preset.inductive)
