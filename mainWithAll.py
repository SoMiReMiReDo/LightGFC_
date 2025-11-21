import time

from scr.para import *
from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *
from torch import nn, optim

import wandb
import shutil

from trainLogger import TrainingLogger


args = para()
args.result_path =  f'./results/'
args = create_folder(args)
args = device_setting(args)
seed_everything(args.seed)


def count_nodes_per_class(graph):
    labels = graph.y.cpu().numpy()
    class_count = Counter(labels)

    print("====== Synthetic Graph Class Distribution ======")
    for cls, num in sorted(class_count.items()):
        print(f"Class {cls}: {num} nodes")
    print("Total nodes:", len(labels))

    return class_count


def generate_labels(labels_train):

    labels_np = labels_train.cpu().numpy() if isinstance(labels_train, torch.Tensor) else labels_train
    if args.dataset_name in ['arxiv']:
        labels_np = labels_np.flatten()

    counter = Counter(labels_np)

    labels_syn = labels_np.tolist()

    num_class_dict = dict(counter)

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

        feat_loss = feat_loss / coeff_sum

        total_loss = args.alpha1 *loss + args.deta *feat_loss
        total_loss.backward()
        optimizer_feat.step()

        if i % 10 == 0:
            validation_model.eval()
            with torch.no_grad():
                output_syn = validation_model(feat_syn)
                acc = accuracy(output_syn, labels_syn)
                print(f"[Epoch {i}] Loss: {total_loss.item():.4f} | Acc: {acc:.4f}")
#                 wandb.log({'feat_Loss': total_loss, 'feat_Acc': acc})
            validation_model.train()


# MyWandb
# wandb.init(project="",name=time.strftime("%Y,%m,%d-%H:%M:%S"))
# log_test = {}
# wandb.log(log_test)

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

##hyper para
args = hyperpara(args) if args.generate_adj == 1 else hyperpara_noadj(args)


H0 = conv_graph_multi(args, data)

# MLP
MLP_model = MLP(in_dim=data.x.shape[1], hidden_dim=128, out_dim=args.num_class ).to(args.device)
optimizer = torch.optim.Adam(MLP_model.parameters(), lr=args.lr_mlp, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

MLP_model.train()

print("========pre-training MLP_model========")
for epoch in range(args.mlp_epoch):
    optimizer.zero_grad()
    out = MLP_model(H0)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        # wandb.log({'mlp_Loss': loss, 'mlp_Acc': acc})
        # wandb.log({'loss': loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

MLP_model.eval()


begin = time.time()



H = conv_graph_multi0(args, data)


W = linear_model(args, H, data, data_test)
new_conf = F.cosine_similarity(H0[data.train_mask], data.x[data.train_mask], dim=1).to(args.device)

H_aug, y_aug, conf = data_assessment(args, data, W, H, new_conf)

# args, label_cond = generate_labels_syn(args, data)
args, label_cond = generate_labels_conf(args, data, new_conf)

M_norm = mask_generation_conf(H_aug, y_aug, args, 'spectral', conf)
h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))


graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))
device = args.device

feat_syn = graph.x.to(args.device)
labels_syn = graph.y.to(args.device)

feat_train = data.x.to(device)
labels_train = data.y.to(device)

nclass = len(torch.unique(labels_train))
index, index_syn, coeff, coeff_sum, knn_class = generate_auxiliary_info(feat_train, labels_train, feat_syn, labels_syn, nclass, device)


feat_syn_param = nn.Parameter(feat_syn.clone())


train_feat_syn(feat_syn_param,labels_syn,MLP_model,feat_train,index,index_syn,knn_class,coeff,coeff_sum,args,device)
args.cond_time = time.time()-begin
# todo ======================================================================================================================

graph = Data(x=feat_syn_param.data, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))

print('Condensation time:',  f'{args.cond_time:.3f}', 's')
print('#training labels:', data.train_mask.sum().item())
print('#augmented labels:', len(H_aug))
args.changed_label = len(H_aug)-data.train_mask.sum().item()

graph=graph.to(args.device)
acc= []
for repeat in range(args.repeat):
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    args.test_gnn = model.__class__.__name__
    acc.append(model_training(model, args, data, graph, data_val, data_test, wandb))
# result_record(args, acc)
print(acc)

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


#
print("================================================")
print(args.ratio)
print(args.dataset_name)
print(args.alpha1)
print(args.deta)
print("================================================")


# graph_file = './CondGraph/'+args.dataset_name+'.pt'
# graph_file = './CondGraph/light_'+args.dataset_name+'_'+ str(args.ratio) + '.pt'
# torch.save(graph, graph_file)


# csv_logger = TrainingLogger('./res/911RedditFlickr/'+args.dataset_name+'.csv', mode='csv')
# csv_logger.log(args, acc=acc)

# print_peak_memory(args.device)




# 调用统计函数
class_distribution = count_nodes_per_class(graph)
