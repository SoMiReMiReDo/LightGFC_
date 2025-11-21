from scr.module import *


class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid)) 
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid)) 
            self.layers.append(GCNConv(nhid, nout))  
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

class SGC(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, cached=False, dropout=0):
        super().__init__()
        self.layers = SGConv(nin, nout, nlayers, cached=cached) 
        self.H_val =None
        self.H_test=None
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        self.layers.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) 
    
    def MLP(self, H):
        x = self.layers.lin(H)
        return F.log_softmax(x, dim=1)    

class MLP(nn.Module):   # 训练用
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP0(nn.Module):  # 处理用
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP0, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = data.x  # 从Data里取特征
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(SAGEConv(nin, nout))
        else:
            # 输入层
            self.layers.append(SAGEConv(nin, nhid))
            # 隐藏层
            for _ in range(nlayers - 2):
                self.layers.append(SAGEConv(nhid, nhid))
            # 输出层
            self.layers.append(SAGEConv(nhid, nout))

        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 遍历除最后一层外的所有层
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        # 应用最后一层
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if nlayers == 1:
            mlp = nn.Sequential(nn.Linear(nin, nout))
            self.convs.append(GINConv(mlp, train_eps=True))
        else:
            # 输入层
            mlp_in = nn.Sequential(nn.Linear(nin, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
            self.convs.append(GINConv(mlp_in, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(nhid))

            # 隐藏层
            for _ in range(nlayers - 2):
                mlp = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU(), nn.Linear(nhid, nhid))
                self.convs.append(GINConv(mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(nhid))

            # 输出层
            mlp_out = nn.Sequential(nn.Linear(nhid, nout))
            self.convs.append(GINConv(mlp_out, train_eps=True))

        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 遍历除最后一层外的所有卷积层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 应用最后一层
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class JKNet(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5, jk_mode='cat'):
        super().__init__()
        # 底层使用GCNConv作为卷积层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nin, nhid))
        for _ in range(nlayers - 1):
            self.convs.append(GCNConv(nhid, nhid))

        # Jumping Knowledge层用于聚合各层输出
        self.jump = JumpingKnowledge(mode=jk_mode, channels=nhid, num_layers=nlayers)
        self.dropout = dropout

        # 分类器的输入维度取决于聚合模式
        if jk_mode == 'cat':
            self.classifier = nn.Linear(nlayers * nhid, nout)
        else:  # 'max' 或 'lstm' 模式
            self.classifier = nn.Linear(nhid, nout)

        self.initialize()

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()
        self.jump.reset_parameters()

    # def forward(self, data):
    #     x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    #     layer_outputs = []
    #
    #     # 计算每一层的输出
    #     for conv in self.convs:
    #         x = conv(x, edge_index, edge_attr)
    #         x = F.relu(x)
    #         x = F.dropout(x, self.dropout, training=self.training)
    #         layer_outputs.append(x)
    #
    #     # 使用Jumping Knowledge聚合所有层的输出
    #     h = self.jump(layer_outputs)
    #
    #     # 应用最终的分类器
    #     out = self.classifier(h)
    #
    #     return F.log_softmax(out, dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index)  # 删除 edge_attr
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            layer_outputs.append(x)
        h = self.jump(layer_outputs)
        out = self.classifier(h)
        return F.log_softmax(out, dim=1)

