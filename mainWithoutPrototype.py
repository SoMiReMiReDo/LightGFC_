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

def train_feat_syn(feat_syn,labels_syn,validation_model,feat_train_batch,index,index_syn,knn_class,coeff,coeff_sum,args,device):
    validation_model.train()
    for param in validation_model.parameters():
        param.requires_grad = False

    optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
    loss_fn = nn.MSELoss()

    for i in range(args.condensation_epoch + 1):
        optimizer_feat.zero_grad()

        output_syn_batch = validation_model(feat_syn)
        loss = F.nll_loss(output_syn_batch, labels_syn)

        feat_loss = torch.tensor(0.0, device=device)
        dis_loss = torch.tensor(0.0, device=device)
        nclass = labels_syn.max().item() + 1
        for c in range(nclass):
            if coeff[c] > 0 and len(index[c]) > 0 and len(index_syn[c]) > 0:
                feat_loss += coeff[c] * loss_fn(
                    feat_train_batch[index[c]].mean(dim=0),
                    feat_syn[index_syn[c]].mean(dim=0)
                )
                _, I = knn_class[c].search(
                    feat_syn[index_syn[c]].detach().cpu().numpy(),
                    args.anchor
                )
                I = torch.tensor(I.ravel(), dtype=torch.long, device=device)

                # print(index_syn[c].shape)
                # print(feat_syn[index_syn[c]].shape)
                # print(feat_train_batch[index[c]][I].mean(dim=1).shape)

        #         dis_loss += coeff[c] * loss_fn(
        #             feat_syn[index_syn[c]],
        #             feat_train_batch[index[c]][I].mean(dim=1)
        #         )
        feat_loss = feat_loss / coeff_sum
        # dis_loss = dis_loss / coeff_sum

        # if i % 10 == 0:
        #     print("loss: ",loss)
        #     print("feat_loss: ",feat_loss)

        # total_loss = loss + args.feat_alpha * feat_loss + args.dis_alpha * dis_loss
        # total_loss = loss +  feat_loss +  dis_loss
        # todo alpha1是mlp， deta是class
        total_loss = args.alpha1 *loss
        total_loss.backward()
        optimizer_feat.step()

        if i % 10 == 0:
            validation_model.eval()
            with torch.no_grad():
                output_syn = validation_model(feat_syn)
                acc = accuracy(output_syn, labels_syn)
                print(f"[Epoch {i}] Loss: {total_loss.item():.4f} | Acc: {acc:.4f}")
# /                wandb.log({'feat_Loss': total_loss, 'feat_Acc': acc})
            validation_model.train()


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

# todo ======================================================================================================================
#   预训练MLP 进行压缩
H = conv_graph_multi(args, data) # 这里的H是一个 H0, H1, H2 组成的三元组

MLP_model = MLP(in_dim=data.x.shape[1], hidden_dim=128, out_dim=args.num_class ).to(args.device)

optimizer = torch.optim.Adam(MLP_model.parameters(), lr=args.lr_mlp, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# output_syn_batch = validation_model.forward(feat_syn)

MLP_model.train()
print("========pre-training MLP_model========")
for epoch in range(args.mlp_epoch):
    optimizer.zero_grad()
    out = MLP_model(H)  # 预测 logits
    loss = criterion(out, data.y)  # L = L[MLP(X), Y]
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        # wandb.log({'mlp_Loss': loss, 'mlp_Acc': acc})
        # wandb.log({'loss': loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

MLP_model.eval()
# ---------------------------------------------------------


# yy = data.y
# data = Data(x=H,y=yy)

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

# Step 6: 优化合成特征
train_feat_syn(feat_syn_param,labels_syn,MLP_model,feat_train,index,index_syn,knn_class,coeff,coeff_sum,args,device)

# Step 7: 保存训练好的合成特征和标签
# torch.save(feat_syn_param.data, f'{root}/saved_ours_large/feat_{args.dataset}_kcenter_{args.seed}.pt')
# torch.save(labels_syn, f'{root}/saved_ours_large/labels_{args.dataset}_kcenter_{args.seed}.pt')

completed_feat_syn = feat_syn.data.detach().clone()  # 复制一份防止修改
# torch.save(feat_syn.data, feat_path)
        # torch.save(labels_syn, label_path)

"""
if args.generate_adj == 1: # (A',X')
    a = get_adj(h, args.adj_T)
    x = get_feature(a, h, args.alpha)
    graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
else:   # (I,X')
    graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))
"""


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


print('Condensation time:',  f'{args.cond_time:.3f}', 's')
# print('#edges:', int(torch.sum(a).item())) if args.generate_adj == 1 else print('No adj')
# print('#training labels:', data.train_mask.sum().item())
# print('#augmented labels:', len(H_aug))
# args.changed_label = len(H_aug)-data.train_mask.sum().item()
# torch.save(graph, graph_file)


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
csv_logger = TrainingLogger('./res/826AblationTest/noFeat.csv', mode='csv')
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
