"""Write resolved settings, metrics, and graph-free tensors to disk."""

import json
import platform
import statistics
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
import torch_geometric


RUN_COLUMNS = (
    "seed", "best_epoch", "val_accuracy_pct", "test_accuracy_pct",
    "condensation_seconds", "condensation_peak_cuda_bytes",
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_run_dir(config, device, benchmark, budget, base_nodes):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{config.dataset}-{config.ratio:g}-{config.variant}-{config.model}-"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=config.output_dir))
    settings = asdict(config)
    settings.update(
        data_dir=str(config.data_dir.resolve()),
        output_dir=str(config.output_dir.resolve()),
        resolved_data_source=benchmark.source,
        split_sizes={split: len(indices) for split, indices in benchmark.splits.items()},
        input_nodes=len(benchmark.x),
        input_adjacency_entries=benchmark.adjacency.nnz,
        budget=budget,
        ratio_base_nodes=base_nodes,
        realized_ratio=budget / base_nodes,
        python=platform.python_version(),
        platform=platform.platform(),
        torch=str(torch.__version__),
        torch_geometric=torch_geometric.__version__,
        numpy=version("numpy"),
        scipy=version("scipy"),
        scikit_learn=version("scikit-learn"),
        ogb=version("ogb"),
        class_ids=benchmark.train_classes.tolist(),
        cuda=torch.version.cuda,
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    _write_json(run_dir / "config.json", settings)
    return run_dir


def save_condensed(run_dir, seed, features, labels):
    artifact = run_dir / f"condensed-seed-{seed}.pt"
    temporary = artifact.with_suffix(".tmp")
    torch.save({"x": features.cpu(), "y": labels.cpu()}, temporary)
    temporary.replace(artifact)


def write_summary(run_dir, test_accuracies, condensation_times):
    mean = statistics.mean(test_accuracies)
    std = statistics.stdev(test_accuracies) if len(test_accuracies) > 1 else None
    _write_json(run_dir / "summary.json", {
        "runs": len(test_accuracies),
        "test_accuracy_mean_pct": mean,
        "test_accuracy_sample_std_pct": std,
        "condensation_seconds_mean": statistics.mean(condensation_times),
    })
    return mean, std
