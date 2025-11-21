import time

from torch.cuda import graph

from scr.para import *
from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *
from torch import nn, optim
import shutil

from trainLogger import TrainingLogger

def generate_labels(labels_train):
    # labels_train 是 Tensor 或 numpy数组

    labels_np = labels_train.cpu().numpy() if isinstance(labels_train, torch.Tensor) else labels_train
    counter = Counter(labels_np)  # 统计每个类别的样本数量，dict {类别: 数量}

    # 这里第一个返回值可以是合成标签列表（例如重复标签或者压缩标签）
    # 简单示范：这里直接返回原标签作为合成标签（示例）
    labels_syn = labels_np.tolist()

    num_class_dict = dict(counter)  # 转为普通dict返回

    return labels_syn, num_class_dict

def generate_auxiliary_info(feat_train, labels_train, feat_syn, labels_syn, nclass, device):
    index = []
    index_syn = []

    _, num_class_dict = generate_labels(labels_train)
    coeff = []
    coeff_sum = 0.0
    for c in range(nclass):
        idx_train = torch.where(labels_train == c)[0]
        idx_syn = torch.where(labels_syn == c)[0]
        index.append(idx_train)
        index_syn.append(idx_syn)

        if c in num_class_dict:
            coe = num_class_dict[c] / max(num_class_dict.values())
            coeff_sum += coe
            coeff.append(coe)
        else:
            coeff.append(0.0)
    coeff_sum = torch.tensor(coeff_sum).to(device)

    d = feat_train.shape[1]
    knn_class = []
    for c in range(nclass):
        if c in num_class_dict and len(index[c]) > 0:
            knn = faiss.IndexFlatL2(d)
            knn.add(feat_train[index[c]].cpu().numpy().astype('float32'))
            knn_class.append(knn)
        else:
            knn_class.append(0)

    return index, index_syn, coeff, coeff_sum, knn_class

args = para()
args.result_path =  f'./results/'
args = create_folder(args)
args = device_setting(args)
seed_everything(args.seed)

# conda run -n cgc1 --no-capture-output python /home/yfx/CGC/mainWithMLP.py

# 将 argparse.Namespace 转换为 dict
config_dict = vars(args)

# export WANDB_BASE_URL=https://api.bandw.top

# todo MyWandb
# wandb.init(project="x.o_"+args.dataset_name+"数据集_<3",name=time.strftime("%Y,%m,%d-%H:%M:%S"),config=config_dict)
# # wandb.init(project="CGC_aug的提升",name=time.strftime("普通的H"))
# log_test = {}
# wandb.log(log_test)


## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)
ori_graph = datasets

##hyper para
args = hyperpara(args) if args.generate_adj == 1 else hyperpara_noadj(args)

# print(args.dataset_name)

## cond data
graph_file = args.folder+f'{args.dataset_name}_{args.ratio}.pt' if args.generate_adj == 1 else args.folder+f'{args.dataset_name}_noadj_{args.ratio}.pt'
# if os.path.exists(graph_file):
#     graph = torch.load(graph_file, map_location= args.device)
# else:


H = conv_graph_multi(args, data) # 这里的H是一个 H0, H1, H2 组成的三元组

data = Data(
    x=H,
    y=data.y,
    edge_index=data.edge_index,  # 如果你还需要图结构
    train_mask=data.train_mask
    # val_mask=data.val_mask,
    # test_mask=data.test_mask
)

# todo ======================================================================================================================
begin = time.time()

args, x_syn, y_syn = generate_labels_kcenter(args, data)
# print(data.x.shape)
# print(data.y.shape)
# print(x_syn.size())
# print(y_syn.size())
feat_syn = x_syn.to(args.device)
labels_syn = y_syn.to(args.device)

device = args.device

# Step 2: 切分原始训练数据
feat_train = data.x.to(device)
labels_train = data.y.to(device)

# Step 3: 构造辅助信息
nclass = len(torch.unique(labels_train))
index, index_syn, coeff, coeff_sum, knn_class = generate_auxiliary_info(feat_train, labels_train, feat_syn, labels_syn, nclass, device)

# Step 5: 将 feat_syn 设为可训练参数
feat_syn_param = nn.Parameter(feat_syn.clone())


if args.generate_adj == 1: # (A',X')
    # (A',X')
    h = feat_syn.data

    h = F.normalize(h, dim=1)
    a = torch.mm(h, h.t())
    adj = (a > args.adj_T).float()
    adj = adj - torch.diag(torch.diag(adj, 0))

    graph = Data(x=feat_syn_param.data, y=labels_syn, edge_index=adj.nonzero().t(),
                 edge_attr=adj[adj.nonzero()[:, 0], adj.nonzero()[:, 1]],
                 train_mask=torch.ones(len(feat_syn.data), dtype=torch.bool))

    # a = get_adj(h, args.adj_T)
    # x = get_feature(a, h, args.alpha)
    # graph = Data(x=x, y=labels_syn, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
else:   # (I,X')
    # (I,X')
    graph = Data(x=feat_syn_param.data, y=labels_syn, edge_index=torch.eye(len(feat_syn.data)).nonzero().t(),
                 edge_attr=torch.ones(len(feat_syn.data)), train_mask=torch.ones(len(feat_syn.data), dtype=torch.bool))

    # graph = Data(x=h, y=labels_syn, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))


args.cond_time = time.time()-begin  # 单次压缩时间
# todo ======================================================================================================================


# gnn model training
# graph已经被压缩完了，重新跑了一边的gnn训练过程
graph=graph.to(args.device)
acc= []
for repeat in range(args.repeat):
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    args.test_gnn = model.__class__.__name__
    acc.append(model_training(model, args, ori_graph, graph, data_val, data_test, wandb))
# result_record(args, acc)
print(acc)

# todo 记录数据
csv_logger = TrainingLogger('./res/826AblationTest/noBoth.csv', mode='csv')
csv_logger.log(args, acc=acc)

# 删除数据集Cora
if args.dataset_name in ['cora', 'citeseer']:
    if args.dataset_name == 'arxiv':
        path = args.raw_data_dir + 'ogbn_arxiv'
    else:
        path = args.raw_data_dir + args.dataset_name
    try:
        # os.remove(path)
        shutil.rmtree(path)
        print(f"已删除: {path}")
    except Exception as e:
        print(f"删除 {path} 失败: {e}")

print("================================================")
print(args.ratio)
print(args.dataset_name)
print("================================================")
