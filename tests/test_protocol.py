import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric import seed_everything

from lightgfc.cli import parse_args
from lightgfc.condensation import class_budgets, condense, proto_features
from lightgfc.datasets import Benchmark
from lightgfc.evaluation import evaluate


CPU = torch.device("cpu")


def toy_benchmark(inductive=False):
    x = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.], [1., 1.], [2., 2.], [1., 2.]])
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0])
    adjacency = sp.diags([1, 1], [-1, 1], shape=(len(x), len(x)), dtype=np.float32, format="csr")
    splits = {"train": torch.arange(4), "val": torch.tensor([4]), "test": torch.tensor([5, 6])}
    return Benchmark(x, y, adjacency, splits, "test-fixture", inductive)


class ExperimentProtocolTest(unittest.TestCase):
    def test_exact_budget_keeps_rare_classes_and_respects_capacity(self):
        counts = np.array([1, 2, 10])
        weights = np.array([1000., 1., 0.])
        np.testing.assert_array_equal(class_budgets(counts, weights, 9), [1, 2, 6])
        with self.assertRaisesRegex(ValueError, "Budget"):
            class_budgets(counts, weights, 2)

    def test_inductive_aggregation_ignores_held_out_features(self):
        benchmark = toy_benchmark(inductive=True)
        x, adjacency, _ = benchmark.graph("train")
        original = proto_features(x, adjacency, 3, CPU)
        benchmark.x[4:] = 10000
        x, adjacency, _ = benchmark.graph("train")
        torch.testing.assert_close(original, proto_features(x, adjacency, 3, CPU))

    def test_transductive_condensation_ignores_held_out_labels(self):
        benchmark = toy_benchmark()
        args = parse_args(["--mlp-epochs", "2", "--projector-epochs", "2", "--hidden", "4"])
        seed_everything(19)
        expected = condense(benchmark, args, 3, CPU, 19)
        benchmark.y[4:] = 10000
        seed_everything(19)
        actual = condense(benchmark, args, 3, CPU, 19)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(expected_tensor, actual_tensor)

    def test_test_set_is_scored_once_with_validation_selected_weights(self):
        benchmark = toy_benchmark()
        args = parse_args(["--model", "mlp", "--hidden", "4", "--epochs", "3", "--eval-every", "1"])
        checkpoint = {}

        def score(model, graph, indices, labels):
            if len(labels) == 2:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[key])
                return 0.25
            if not checkpoint:
                checkpoint.update({key: value.clone() for key, value in model.state_dict().items()})
                return 0.75
            return 0.5

        with patch("lightgfc.evaluation.accuracy", side_effect=score) as mock_score:
            result = evaluate(benchmark, benchmark.x[:4], benchmark.y[:4], args, CPU)
        self.assertEqual(result, (1, 0.75, 0.25))
        self.assertEqual([len(call.args[3]) for call in mock_score.call_args_list], [1, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
