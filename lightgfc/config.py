"""Resolved experiment settings and per-dataset defaults."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class DatasetPreset:
    ratios: tuple[float, float, float]
    epochs: int
    loss_weight: float
    inductive: bool = False
    standardize_features: bool = False


DATASETS = {
    "cora": DatasetPreset((0.013, 0.026, 0.052), epochs=1500, loss_weight=0.5),
    "citeseer": DatasetPreset((0.009, 0.018, 0.036), epochs=1500, loss_weight=0.5),
    "arxiv": DatasetPreset(
        (0.0005, 0.0025, 0.005), epochs=1500, loss_weight=0.5, standardize_features=True,
    ),
    "flickr": DatasetPreset((0.001, 0.005, 0.01), epochs=2000, loss_weight=0.005, inductive=True),
    "reddit": DatasetPreset(
        (0.0005, 0.001, 0.002), epochs=2000, loss_weight=0.005, inductive=True, standardize_features=True,
    ),
    "products": DatasetPreset((0.00025, 0.0005, 0.001), epochs=2000, loss_weight=0.005),
}
MODELS = ("gcn", "mlp", "sage", "sgc", "gin", "jknet")
VARIANTS = ("full", "no-label", "no-prototype", "no-both", "no-similarity")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    ratio: float
    ratio_base: str
    data_dir: Path
    data_source: str
    output_dir: Path
    variant: str
    model: str
    gpu: int
    seed: int
    runs: int
    seeds: tuple[int, ...]
    hops: int
    hidden: int
    mlp_epochs: int
    projector_epochs: int
    epochs: int
    eval_every: int
    mlp_batch_size: int
    lr_mlp: float
    lr_projector: float
    lr: float
    dropout: float
    weight_decay: float
    alpha: float
    beta: float


def resolve_config(options):
    """Fill in dataset defaults and validate parsed command-line values."""
    values = options.copy()
    preset = DATASETS[values["dataset"]]
    defaults = {
        "ratio": preset.ratios[1],
        "epochs": preset.epochs,
        "alpha": preset.loss_weight,
        "beta": preset.loss_weight,
    }
    for name, default in defaults.items():
        if values[name] is None:
            values[name] = default
    if values["ratio_base"] == "auto":
        values["ratio_base"] = "train" if preset.inductive else "all"

    if not 0 < values["ratio"] <= 1:
        raise ValueError("--ratio must be in (0, 1]")
    if not 0 <= values["dropout"] < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if values["gpu"] < -1:
        raise ValueError("--gpu must be -1 or a CUDA device index")
    for name in ("runs", "hidden", "mlp_epochs", "projector_epochs", "epochs", "eval_every", "mlp_batch_size"):
        if values[name] < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "lr_mlp", "lr_projector", "alpha", "beta", "weight_decay"):
        value = values[name]
        if not isfinite(value) or value < 0 or (name.startswith("lr") and value == 0):
            raise ValueError(f"invalid value for --{name.replace('_', '-')}")

    seed, runs, seeds = values["seed"], values["runs"], values["seeds"]
    if seeds is None:
        if not 0 <= seed < seed + runs <= 2**32:
            raise ValueError("seeds must be distinct integers in [0, 2**32)")
        seeds = range(seed, seed + runs)
    else:
        if seed != 0 or runs != 1:
            raise ValueError("--seeds cannot be combined with --seed or --runs")
        if len(set(seeds)) != len(seeds) or not all(0 <= item < 2**32 for item in seeds):
            raise ValueError("seeds must be distinct integers in [0, 2**32)")
    values["seeds"] = tuple(seeds)

    variant, alpha, beta = values["variant"], values["alpha"], values["beta"]
    if variant not in {"no-label", "no-both"} and alpha == 0 and (variant == "no-prototype" or beta == 0):
        raise ValueError("at least one enabled condensation loss must have positive weight")
    if variant == "no-label" and beta == 0:
        raise ValueError("--no-label variant requires a positive --beta")
    return ExperimentConfig(**values)
