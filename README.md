# LightGFC

Experiment code for *Lightweight Graph-Free Condensation With MLP-Driven Optimization* by Fuxin Yu, Xin Zheng, Yi Li, Yanqing Guo, Xikun Zhang, Jeffrey Chan, and Feng Xia.

LightGFC aggregates multi-hop features, pretrains an MLP on labeled training nodes, and produces a graph-free set of condensed nodes. Downstream models train on those nodes. The original validation split selects the checkpoint, and the original test split scores it.

## Install

From the repository root, use Python 3.11 and install the project. For Linux CUDA runs, install the matching PyTorch wheel before installing LightGFC. See the [PyTorch 2.7.1 wheel instructions](https://pytorch.org/get-started/previous-versions/#v271).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For Conda, run `conda env create -f environment.yml`, activate `lightgfc`, and run `python -m pip install --no-deps -e .`. The pinned runtime dependencies live in `requirements.txt`.

## Run

This short CPU run downloads Cora into `data/` and saves results under `results/`.

```bash
python -m lightgfc --dataset cora --ratio 0.026 --gpu -1 \
  --mlp-epochs 5 --projector-epochs 5 --epochs 10
```

An installed environment also provides the `lightgfc` command. Use `python -m lightgfc --help` for all options. The original `main.py`, `mainWithAll.py`, and `mainWithout*.py` scripts forward to the same CLI.

See the [experiment guide](docs/experiments.md) for data sources, paper settings, ablations, and output formats.

## Project layout

| Path | Responsibility |
| --- | --- |
| `lightgfc/cli.py`, `lightgfc/config.py` | Command-line interface, validation, and dataset defaults |
| `lightgfc/datasets.py`, `lightgfc/graph.py` | Data loading, splits, and sparse adjacency |
| `lightgfc/condensation.py` | Multi-hop aggregation, MLP pretraining, and node condensation |
| `lightgfc/models.py`, `lightgfc/evaluation.py` | Downstream models and validation-based checkpoint selection |
| `lightgfc/experiment.py`, `lightgfc/artifacts.py` | Multi-seed execution and saved results |
| `tests/`, `docs/` | Offline regression tests and experiment documentation |

Run the offline tests with `python -m unittest discover -s tests -v`. The paper citation is in [CITATION.cff](CITATION.cff). The software license is pending maintainer confirmation.
