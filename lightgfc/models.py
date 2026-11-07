"""Downstream classifiers. Every model returns unnormalized logits."""

from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINConv, JumpingKnowledge, SAGEConv


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adjacency=None):
        return self.fc2(F.relu(self.fc1(x)))


def _gcn_layer(in_dim, out_dim):
    return GCNConv(in_dim, out_dim, normalize=False, add_self_loops=False)


class GraphClassifier(nn.Module):
    def __init__(self, name, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if name == "gcn":
            self.first = _gcn_layer(in_dim, hidden_dim)
            self.second = _gcn_layer(hidden_dim, out_dim)
        elif name == "sage":
            self.first = SAGEConv(in_dim, hidden_dim)
            self.second = SAGEConv(hidden_dim, out_dim)
        else:
            self.first = GINConv(
                nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
                train_eps=True,
            )
            self.second = GINConv(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)),
                train_eps=True,
            )
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if name == "gin" else nn.Identity()

    def forward(self, x, adjacency):
        hidden = self.dropout(F.relu(self.batch_norm(self.first(x, adjacency))))
        return self.second(hidden, adjacency)


class JKNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.first = _gcn_layer(in_dim, hidden_dim)
        self.second = _gcn_layer(hidden_dim, hidden_dim)
        self.jump = JumpingKnowledge("cat")
        self.classifier = nn.Linear(2 * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        first = self.dropout(F.relu(self.first(x, adjacency)))
        second = self.dropout(F.relu(self.second(first, adjacency)))
        return self.classifier(self.jump([first, second]))


class SGC(nn.Module):
    """The two propagation steps are precomputed before training and evaluation."""

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency=None):
        return self.classifier(self.dropout(x))


def build_model(name, in_dim, hidden_dim, out_dim, dropout):
    if name == "mlp":
        return MLP(in_dim, hidden_dim, out_dim)
    if name == "sgc":
        return SGC(in_dim, out_dim, dropout)
    if name == "jknet":
        return JKNet(in_dim, hidden_dim, out_dim, dropout)
    return GraphClassifier(name, in_dim, hidden_dim, out_dim, dropout)
