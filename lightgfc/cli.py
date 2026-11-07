"""Command-line entry point for LightGFC experiments."""

import argparse
from pathlib import Path

from .config import DATASETS, MODELS, VARIANTS, resolve_config


def parse_args(argv=None, default_variant="full"):
    parser = argparse.ArgumentParser(description="Run a LightGFC node-classification experiment.")
    parser.add_argument("--dataset", "--dataset_name", choices=DATASETS, type=str.lower, default="cora")
    parser.add_argument("--ratio", type=float, help="Condensed fraction, e.g. 0.026 for 2.6%%")
    parser.add_argument(
        "--ratio-base", choices=("auto", "all", "train"), default="auto",
        help="Budget denominator. Auto uses all nodes for transductive data and training nodes otherwise.",
    )
    parser.add_argument("--data-dir", "--raw_data_dir", type=Path, default=Path("data"))
    parser.add_argument("--data-source", choices=("auto", "official", "legacy"), default="auto")
    parser.add_argument("--output-dir", "--result_path", type=Path, default=Path("results"))
    parser.add_argument("--variant", choices=VARIANTS, default=default_variant)
    parser.add_argument("--model", "--test_gnn", choices=MODELS, type=str.lower, default="gcn")
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA device index, or -1 for CPU")
    parser.add_argument("--seed", type=int, default=0, help="First seed for consecutive runs")
    parser.add_argument("--runs", "--repeat", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list")
    parser.add_argument("--hops", type=int, choices=range(1, 6), default=3,
                        help="Number of terms in the hop average, including the input features")
    parser.add_argument("--hidden", "--n_dim", type=int, default=256)
    parser.add_argument("--mlp-epochs", "--mlp_epoch", type=int, default=200)
    parser.add_argument("--projector-epochs", "--condensation_epoch", type=int, default=300)
    parser.add_argument("--epochs", "--epoch", type=int)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--lr-mlp", "--lr_mlp", type=float, default=0.005)
    parser.add_argument("--lr-projector", "--lr_feat", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.005, help="Downstream learning rate")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=5e-4)
    parser.add_argument("--alpha", "--alpha1", type=float, help="Label-aware loss weight")
    parser.add_argument("--beta", "--deta", type=float, help="Prototype-aware loss weight")
    options = vars(parser.parse_args(argv))
    try:
        return resolve_config(options)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None, default_variant="full"):
    config = parse_args(argv, default_variant=default_variant)

    from .datasets import load_benchmark
    from .experiment import get_device, run_experiment

    try:
        device = get_device(config.gpu)
        benchmark = load_benchmark(config.dataset, config.data_dir, config.data_source)
        run_experiment(config, device, benchmark)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"LightGFC: {exc}") from exc
