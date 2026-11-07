"""Orchestrate condensation and downstream evaluation across seeds."""

import csv
import time

import torch
from torch_geometric import seed_everything

from .artifacts import RUN_COLUMNS, create_run_dir, save_condensed, write_summary
from .condensation import condense
from .evaluation import evaluate


def get_device(gpu):
    if gpu == -1:
        return torch.device("cpu")
    if gpu >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {gpu} is unavailable. Use --gpu -1 for CPU.")
    return torch.device(f"cuda:{gpu}")


def run_experiment(config, device, benchmark):
    train_nodes = len(benchmark.splits["train"])
    base_nodes = train_nodes if config.ratio_base == "train" else len(benchmark.x)
    budget = round(config.ratio * base_nodes)
    if not benchmark.num_classes <= budget <= train_nodes:
        raise ValueError(
            f"Ratio {config.ratio:g} on {base_nodes} {config.ratio_base} nodes gives {budget} "
            f"synthetic nodes. The supported budget is {benchmark.num_classes} to {train_nodes}."
        )

    run_dir = create_run_dir(config, device, benchmark, budget, base_nodes)
    print(f"Results directory: {run_dir.resolve()}", flush=True)

    test_accuracies = []
    condensation_times = []
    with (run_dir / "runs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(RUN_COLUMNS)
        for seed in config.seeds:
            seed_everything(seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            features, labels = condense(benchmark, config, budget, device, seed)
            peak_memory = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = torch.cuda.max_memory_allocated(device)
            seconds = time.perf_counter() - start

            # Match downstream initialization across ablations for each seed.
            seed_everything(seed)
            epoch, val_acc, test_acc = evaluate(benchmark, features, labels, config, device)
            save_condensed(run_dir, seed, features, labels)
            del features, labels

            test_accuracies.append(100 * test_acc)
            condensation_times.append(seconds)
            writer.writerow((seed, epoch, 100 * val_acc, 100 * test_acc, seconds, peak_memory))
            file.flush()
            print(f"Seed {seed} | {budget} nodes | epoch {epoch} | "
                  f"val {100 * val_acc:.2f}% | test {100 * test_acc:.2f}% | "
                  f"condensation {seconds:.3f}s", flush=True)

    mean, std = write_summary(run_dir, test_accuracies, condensation_times)
    print(f"Test accuracy: {mean:.2f}%" if std is None else f"Test accuracy: {mean:.2f} ± {std:.2f}%")
    return run_dir
