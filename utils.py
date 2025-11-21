import wandb
import math

from scr.para import *
from scr.module import *
from scr.models import *


def generate_labels_syn(args, data):
    reduction_rate = ratio_transfer(args)

    from collections import Counter
    counter = Counter(np.array(data.y[data.train_mask].cpu()))
    num_class_dict = {}
    n = len(data.y[data.train_mask])

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]

    num_class = np.zeros(len(num_class_dict), dtype=int)
    for i in range(len(num_class_dict)):
        num_class[i] = num_class_dict[i]

    labels_syn = []
    for i in range(args.num_class):
        labels_syn += [i] * num_class[i]
    labels_syn = torch.tensor(labels_syn).long()

    args.budget = sum(num_class)
    args.budget_cla = num_class
    return args, labels_syn



def generate_labels_conf(args, data, new_conf_train):

    # 1. 获取原始训练集的标签
    y_train = data.y[data.train_mask].cpu()

    # 2. 计算每个类别的总置信度
    class_conf_sums = []
    for i in range(args.num_class):
        # 找到属于类别 i 的所有节点的索引
        mask_class_i = (y_train == i)
        # 如果该类别在训练集中存在节点，则计算其置信度之和
        if mask_class_i.sum() > 0:
            conf_sum = new_conf_train[mask_class_i.cpu()].sum()
            class_conf_sums.append(conf_sum.item())
        else:
            class_conf_sums.append(0.0)

    # 3. 根据置信度总和计算每个类别应占的比例
    total_conf_sum = sum(class_conf_sums)
    if total_conf_sum == 0:  # 防止除以零
        # 如果所有置信度都为0，则退回到按节点数量比例分配
        print("Warning: Total confidence is zero. Falling back to original node count proportion.")
        return generate_labels_syn(args, data)

    proportions = [s / total_conf_sum for s in class_conf_sums]

    # 4. 计算总的压缩预算
    reduction_rate = ratio_transfer(args)
    total_budget = int(len(y_train) * reduction_rate)

    # 5. 根据比例分配每个类别的预算名额
    # 使用一种公平的取整方法，确保总和精确等于 total_budget
    num_class_list_float = [p * total_budget for p in proportions]
    num_class_list = [int(n) for n in num_class_list_float]

    # 处理四舍五入导致的数量差异
    remainder = total_budget - sum(num_class_list)
    if remainder > 0:
        # 将余数分配给小数部分最大的那些类别
        frac_parts = [n - int(n) for n in num_class_list_float]
        top_indices = np.argsort(frac_parts)[-remainder:]
        for i in top_indices:
            num_class_list[i] += 1

    # 6. 生成新的标签列表
    labels_syn = []
    for i in range(args.num_class):
        labels_syn.extend([i] * num_class_list[i])

    labels_syn = torch.tensor(labels_syn).long()

    # 7. 更新 args 中的预算信息
    args.budget = sum(num_class_list)
    args.budget_cla = num_class_list

    print(f"Original class distribution in train set: {dict(Counter(y_train.numpy()))}")
    print(f"New class distribution based on confidence: {dict(Counter(labels_syn.numpy()))}")

    return args, labels_syn


def device_setting(args):
    if args.gpu != -1:
        args.device='cuda'
    else:
        args.device='cpu'  
    torch.cuda.set_device(args.gpu)
    return args


def create_folder(args):
    args.folder =  args.cond_folder + f'{args.dataset_name}/'
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    return args


def ratio_transfer(args):
    if args.dataset_name == 'cora':
        if args.ratio == 0.013:
            return 0.25
        if args.ratio == 0.026:
            return 0.5
        if args.ratio == 0.052:
            return 0.75
        if args.ratio == 0.104:
            return 1
        return None


    elif args.dataset_name == 'citeseer':
        if args.ratio == 0.009:
            return 0.25
        if args.ratio == 0.018:
            return 0.5
        if args.ratio == 0.036:
            return 1
        if args.ratio == 0.072:
            return 1
        return None

    elif args.dataset_name == 'arxiv':
        if args.ratio == 0.0005:
            return 0.001
        if args.ratio == 0.0025:
            return 0.005
        if args.ratio == 0.005:
            return 0.01
        return None

    else:
        return args.ratio
    

def conv_graph_multi(args, data):
    if args.kernel == "gcn":
        adj_norm = normalize_adj_sparse(data).to(data.x.device)
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        return (H0+ H1+ H2)/3

def conv_graph_multi0(args, data):
    if args.kernel == "gcn":
        adj_norm = normalize_adj_sparse(data).to(data.x.device)
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        return H0, H1, H2

def conv_graph_multi_ablation(args, data, k):
    adj_norm = normalize_adj_sparse(data).to(data.x.device)
    if k == 1:
        return data.x
    if k == 2:
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        return (H0+ H1)/2
    if k == 3:
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        return (H0+ H1+ H2)/3
    if k == 4:
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        H3 = torch.spmm(adj_norm, H2)
        return (H0+ H1+ H2+ H3)/4
    if k == 5:
        H0 = data.x
        H1 = torch.spmm(adj_norm, H0)
        H2 = torch.spmm(adj_norm, H1)
        H3 = torch.spmm(adj_norm, H2)
        H4 = torch.spmm(adj_norm, H3)
        return (H0+ H1+ H2+ H3+ H4)/5


def normalize_adj_sparse(data):
    try:
        mx = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()), shape=(data.x.shape[0], data.x.shape[0]))
        if type(mx) is not sp.lil.lil_matrix:
            mx = mx.tolil()
        if mx[0, 0] == 0 :
            mx = mx + sp.eye(mx.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)

        sparse_mx = mx.tocoo().astype(np.float32)
        sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
        sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
        sparseconcat=torch.cat((sparserow, sparsecol),1)
        sparsedata=torch.FloatTensor(sparse_mx.data)
        adj = torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

    except Exception as e:
        adj = normalize_sparse_tensor(data.edge_index)
    return adj


def linear_model(args, H, data, data_test, bootstrap=False):
    feat = sum(H)/len(H)
    if bootstrap:
        ref_nodes = sample_k_nodes_per_label(data.y, data.train_mask, 100, args.num_class)
    else:
        ref_nodes = data.train_mask.nonzero().view(-1)

    Y_L = torch.nn.functional.one_hot(data.y[ref_nodes], args.num_class).float()
    W = torch.linalg.lstsq(feat[ref_nodes.cpu()].cpu(), Y_L.cpu(), driver="gelss")[0]

    return  W


def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    sampled_indices = [
        label_indices[torch.randperm(len(label_indices))[:k]]
        for label_indices in ref_node_idx
    ]
    return visible_nodes[torch.cat(sampled_indices)]


def add_self_loops_(edge_index, edge_weight=None, fill_value=1, num_nodes=None):

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_sparse_tensor(adj, fill_value=1):
    row, col, _ = adj.coo()
    edge_index = torch.stack([row, col])
    num_nodes= adj.sizes()[0]
    edge_weight = torch.ones(adj.nnz()).to(adj.device())

    edge_index, edge_weight = add_self_loops_(
	edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.sizes()
    return torch.sparse.FloatTensor(edge_index, values, shape)


# 聚类压缩
def mask_generation_conf(H, y, args, method, conf):
    budgets = args.budget_cla
    idx = torch.arange(len(H))
    indices = torch.LongTensor().new_empty((0, 2))
    values = torch.FloatTensor()
    row = 0
    for cls in range(args.num_class):
        cls_mask = (y == cls)
        H_cls = H[cls_mask].cpu().detach().numpy()
        # cluster_labels = clustering(H_cls, budgets[cls], method)
        cluster_labels = clustering_fast(H_cls, budgets[cls], method)
        for center_idx in range(budgets[cls]):
            center_mask = torch.from_numpy(cluster_labels == center_idx)
            idx_center = idx[cls_mask][center_mask]
            cscore = conf[cls_mask][center_mask]
            if len(idx_center) > 0:
                # val = 1.0 / len(idx_center)              
                val = F.softmax(cscore/args.tau, dim=-1)          #calibration
                cls_indices = torch.full((len(idx_center), 1), row, dtype=torch.long)
                combined_indices = torch.cat([cls_indices, idx_center.unsqueeze(1)], dim=1)
                indices = torch.cat([indices, combined_indices], dim=0)
                values = torch.cat([values, val])

            row+=1
    size = torch.Size([args.budget, len(H)])
    mapping_norm = torch.sparse_coo_tensor(indices.t(), values, size)
    return mapping_norm


# def data_assessment(args, data, W, feat):
#     # Feature = H
#     if args.aug_ratio >0:
#
#         H_augs = feat[1][data.train_mask]   # H1
#         y = data.y[data.train_mask]
#
#         logits_aug = H_augs @ W.to(args.device)
#         logits_aug = logits_aug.softmax(dim=-1)
#         y_pred = logits_aug.argmax(1)
#         report = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)
#         acc = [float(metrics['f1-score']) for label, metrics in report.items() if label.isdigit()]
#         weights = 1 - np.array(acc)
#         probabilities = weights[y.cpu()]+1e-20
#         pro = probabilities/sum(probabilities)
#         # todo 按照增强比例对H进行随即采样
#         idx = np.random.choice(len(pro), size=int(len(H_augs)*args.aug_ratio), p=pro)
#
#         #process aug
#         H_aug = H_augs[idx]
#         y_aug = y[idx]
#         logits_aug = logits_aug[idx]
#         y_pred_aug = logits_aug.argmax(1)
#         conf_pred_aug = logits_aug[np.arange(len(logits_aug)), y_aug].cpu()
#         conf_pred_aug[y_pred_aug!=y_aug]=torch.min(conf_pred_aug)
#
#         H = feat[2][data.train_mask]
#         logits = H @ W.to(args.device)
#         logits = logits.softmax(dim=-1)
#         y_pred = logits.argmax(1)
#         conf_pred = logits[np.arange(len(logits)), y].cpu()
#         conf_pred[y_pred!=y]=torch.min(conf_pred)
#
#         H_all = torch.concat([H, H_aug])
#         y_all = torch.concat([y, y_aug])
#         conf_all = torch.concat([conf_pred, conf_pred_aug])
#
#     else:
#         y = data.y[data.train_mask]
#         H = feat[2][data.train_mask]
#         logits = H @ W.to(args.device)
#         logits = logits.softmax(dim=-1)
#         y_pred = logits.argmax(1)
#         conf_pred = logits[np.arange(len(logits)), y].cpu()
#         conf_pred[y_pred!=y]=torch.min(conf_pred)
#
#         H_all = H
#         y_all = y
#         conf_all = conf_pred
#
#     return H_all, y_all, conf_all


def data_assessment(args, data, W, feat, conf_train):
    """
    使用预先计算好的、且形状正确的 conf_train（基于余弦相似度）来代替原有的置信度计算。
    函数签名已修改，接收的 conf_train 参数的形状为 (num_train_nodes,)。
    """
    # 提取训练集标签
    y = data.y[data.train_mask]

    # 如果需要数据增强
    if args.aug_ratio > 0:
        H_augs = feat[1][data.train_mask]  # 使用 H1 进行数据增强评估

        # 增强节点的选择逻辑保持不变（基于线性模型的预测表现）
        logits_aug = H_augs @ W.to(args.device)
        logits_aug = logits_aug.softmax(dim=-1)
        y_pred = logits_aug.argmax(1)
        report = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True, zero_division=0)
        acc = [float(metrics['f1-score']) for label, metrics in report.items() if label.isdigit()]
        weights = 1 - np.array(acc)
        probabilities = weights[y.cpu()] + 1e-20
        pro = probabilities / sum(probabilities)

        # 根据计算出的概率选择要增强的节点索引
        idx = np.random.choice(len(pro), size=int(len(H_augs) * args.aug_ratio), p=pro)

        # 准备增强数据
        H_aug = H_augs[idx]
        y_aug = y[idx]

        # 直接从已经计算好的 conf_train 中为增强节点分配置信度
        conf_aug = conf_train[idx]

        # 准备原始（未增强）的训练数据
        H_original = feat[2][data.train_mask]  # 使用 H2 作为基准特征

        # 拼接原始数据和增强数据
        H_all = torch.concat([H_original, H_aug])
        y_all = torch.concat([y, y_aug])
        conf_all = torch.concat([conf_train, conf_aug])

    # 如果不进行数据增强
    else:
        H_all = feat[2][data.train_mask]
        y_all = y
        conf_all = conf_train

    return H_all, y_all, conf_all




def square_feat_map(z, c=2**-.5):
  polf = PolynomialFeatures(include_bias=True)
  x = polf.fit_transform(z)
  coefs = np.ones(len(polf.powers_))
  coefs[0] = c
  coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
  coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c) 
  return x * coefs


def clustering_fast(H, num_center, method):

    if num_center == len(H):
        return np.arange(num_center)

    if method == 'kmeans':
        kmeans = faiss.Kmeans(int(H.shape[1]), int(num_center), gpu=False)
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(H.astype('float32'))
        _, I = kmeans.index.search(H.astype('float32'), 1)
        cluster_labels = I.flatten()
    elif method == 'spectral':
        H = StandardScaler(with_std=False).fit_transform(H)
        svd = TruncatedSVD(num_center)
        svd.fit(H.T)
        U = svd.components_.T

        Z = square_feat_map(U)
        r = Z.sum(0)
        D = Z @ r 
        Z_hat = Z / D[:,None]**.5
        
        svd = TruncatedSVD(num_center+1)
        svd.fit(Z_hat.T)
        Q = svd.components_.T[:,1:]
        kmeans = faiss.Kmeans(int(Q.shape[1]), int(num_center), gpu=False)
        kmeans.cp.min_points_per_centroid = 1
        kmeans.train(Q.astype('float32'))
        _, I = kmeans.index.search(Q.astype('float32'), 1)
        cluster_labels = I.flatten()
    elif method == 'nocluster':
        samples = np.arange(int(H.shape[0]))
        np.random.shuffle(samples)
        cluster_labels = np.zeros(int(H.shape[0]), dtype=int)
        
        cluster_size = int (int(H.shape[0]) // int(num_center))
        
        for i in range(int(num_center)):
            cluster_labels[samples[i * cluster_size : (i + 1) * cluster_size]] = i

    return cluster_labels


def get_adj(h, adj_T):
    h = F.normalize(h, dim=1)
    a = torch.mm(h, h.t())
    adj = (a>adj_T).float()
    adj = adj - torch.diag(torch.diag(adj, 0))
    return adj


def get_feature(a, h, alpha):
    lap = laplacian_tensor(a)  
    a_norm = normalize_adj_tensor(a)
    a_conv = a_norm
    for _ in range(2):
        a_conv = torch.mm(a_conv, a_norm)
    aTa = torch.mm(a_conv, a_conv.t())
    aTh = alpha*torch.mm(a_conv.t(), h)
    k = (aTa+lap)
    x = torch.linalg.solve(k, aTh)
    return x 


def laplacian_tensor(adj):
    r_inv = adj.sum(1).flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-10
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    L_norm = r_mat_inv @ L @ r_mat_inv
    return L_norm


def normalize_adj_tensor(adj):
    device = adj.device

    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def feature_processing(args, data_val, data_test, k):

    model = SGC(data_val.num_features, None, int(data_val.y.max()+1), k, cached=True).to(args.device)
    model(data_val)
    feat_val = model.layers._cached_x
    model = SGC(data_val.num_features, None, int(data_val.y.max()+1), k, cached=True).to(args.device)
    model(data_test)
    feat_test = model.layers._cached_x
    return feat_val, feat_test



# todo 这里是Gnn的训练过程
def model_training(model, args, data, graph, data_val=None, data_test=None, wandb=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = test_acc = 0
    max_acc = 0
    best_acc = 0

    for epoch in range(1, args.epoch+1):
        if epoch == args.epoch // 2:
            lr = args.lr*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model(graph)
        if args.dataset_name in ['arxiv']:
            graph.y = graph.y.flatten()
        loss = F.nll_loss(output[graph.train_mask], graph.y[graph.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)


        if args.dataset_name in ['flickr', 'reddit']:
            train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch:03d}, Test: {tmp_test_acc:.4f}')
            if tmp_test_acc > best_acc:
                best_acc = tmp_test_acc

        else:
            acc = test(model, data.to(args.device))
            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Test: {acc:.4f}')
                # print(acc)

                if max_acc < acc:
                    max_acc = acc
                # print("max_acc:", max_acc)

                """wandb.log({'gnn_Loss': loss, 'gnn_Acc': acc[0]})"""

                if max_acc > best_acc:
                    best_acc = max_acc

        # if max_acc > best_val_acc:
        #     best_val_acc = max_acc
        #     test_acc = tmp_test_acc

        # if epoch%10 == 0 :
            # wandb.log({'train_acc': train_acc, 'val_acc': val_acc})
            # wandb.log({'loss': loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})

        # if epoch%10 == 0 :
            # print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    print()
    return best_acc
#   return test_acc


def test_inductive(args, model, data_val, data_test, k=2):
    if model.__class__.__name__ == 'SGC' or model.__class__.__name__ == 'SGC2':
        if model.H_val == None:
            model.H_val,  model.H_test = feature_processing(args, data_val, data_test, k)
        with torch.no_grad():
            model.eval()
            accs = []
            accs.append(0)

            out = model.MLP(model.H_val)
            pred = out.argmax(1)
            acc = pred.eq(data_val.y).sum().item() / len(data_val.y)
            accs.append(acc)      

            out = model.MLP(model.H_test)
            pred = out.argmax(1)
            acc = pred.eq(data_test.y).sum().item() / len(data_test.y)
            accs.append(acc)                
    else:
        with torch.no_grad():
            model.eval()
            accs = []
            accs.append(0)
            for data in [data_val, data_test]:
                out = model(data)
                pred = out.argmax(1)
                acc = pred.eq(data.y).sum().item() / len(data.y)
                accs.append(acc)
    return accs


def test(model, data):
    with torch.no_grad():
        model.eval()
        out, accs = model(data), []
        for _, mask in data('test_mask'):
            pred = out[mask].argmax(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs[0]


def result_record(args, ALL_ACCs):
    result_path_file = args.result_path + f"{args.dataset_name}.csv" if args.generate_adj == 1 else args.result_path + f"{args.dataset_name}_noadj.csv"
    ALL_ACC = [np.mean(ALL_ACCs, axis=0)*100, np.std(ALL_ACCs, axis=0, ddof=1)*100] if len(ALL_ACCs) > 1 else [ALL_ACCs[0]*100, 0]
    with open(result_path_file ,'a+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow( ["budget ratio:" f"{args.ratio}",
        "kernel:" f"{args.kernel}", 
        "conv_depth:" f"{args.conv_depth}",
        "cond time:" f'{args.cond_time:.3f}s',  
        "changed label:" f"{args.changed_label}",
        "test GNN:" f"{args.test_gnn}", 
        "lr:" f"{args.lr}",
        "weight_decay:" f"{args.weight_decay}",
        "dropout:" f"{args.dropout}",
        "clustering:" f"{args.clustering}",
        "adj_T:" f"{args.adj_T}",
        "alpha:" f"{args.alpha}",
        "tau:" f"{args.tau}",
        "aug_ratio:" f"{args.aug_ratio}",   
         f"{ALL_ACC[0]:.1f}",
         f"{ALL_ACC[0]:.1f}+{ALL_ACC[1]:.1f}"])


def compute_rel_pos(H_list, embed_dim=None, num_heads=3, angle_proj=None, device=None):
    h_shape = H_list[0].shape
    # print("h_shape: ",len(h_shape))

    if len(h_shape) == 2:
        H_list = [h.unsqueeze(0) for h in H_list]
        batch_size, num_nodes, feat_dim = H_list[0].shape
    elif len(h_shape) == 3:
        batch_size, num_nodes, feat_dim = h_shape
    else:
        raise ValueError(f"Unsupported H shape: {h_shape}")

    embed_dim = feat_dim if embed_dim is None else embed_dim
    head_dim = embed_dim // num_heads

    if angle_proj is None:
        angle_proj = nn.Linear(feat_dim, head_dim, bias=False).to(device or H_list[0].device)

    H_all = torch.stack(H_list, dim=2)  # (batch_size, num_nodes, num_layers, feat_dim)
    angles = angle_proj(H_all)

    angles = torch.remainder(angles, 2 * math.pi)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    seq_len = batch_size * num_nodes * len(H_list)
    rel_pos_cos = cos.view(seq_len, head_dim)
    rel_pos_sin = sin.view(seq_len, head_dim)

    return rel_pos_cos, rel_pos_sin


# todo ======================================================================================================================
from sklearn.metrics import pairwise_distances
from collections import Counter

def kcenter_greedy(X, budget):
    """
    标准 K-Center Greedy 算法：从 X 中选出 budget 个代表样本索引
    """
    n = X.shape[0]
    selected = [np.random.randint(0, n)]
    distances = pairwise_distances(X, X[selected], metric='euclidean').squeeze()

    for _ in range(budget - 1):
        idx = np.argmax(distances)
        selected.append(idx)
        dist_new = pairwise_distances(X, X[[idx]], metric='euclidean').squeeze()
        distances = np.minimum(distances, dist_new)

    return selected

def generate_labels_kcenter(args, data):
    """
    使用 K-Center 方法进行图训练节点压缩，输出压缩后的特征和标签矩阵。
    返回值: args, x_syn, y_syn
    """
    reduction_rate = ratio_transfer(args)
    device = data.x.device

    X = data.x[data.train_mask]
    y = data.y[data.train_mask]

    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    if args.dataset_name in ['arxiv']:
        y_np = y_np.flatten()

    n = len(y_np)
    counter = Counter(y_np)
    class_indices = {c: np.where(y_np == c)[0] for c in counter}

    sorted_counter = sorted(counter.items(), key=lambda x: x[1])
    num_class_dict = {}
    sum_ = 0

    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]

    num_class = np.zeros(len(num_class_dict), dtype=int)
    selected_all = []

    for i, c in enumerate(sorted(num_class_dict.keys())):
        budget = num_class_dict[c]
        idx = class_indices[c]
        class_X = X_np[idx]
        selected_local = kcenter_greedy(class_X, budget)
        selected_global = np.array(idx)[selected_local]
        selected_all.extend(selected_global)
        num_class[i] = budget

    selected_all = torch.tensor(selected_all).long().to(device)
    x_syn = X[selected_all]
    y_syn = y[selected_all]

    args.budget = sum(num_class)
    args.budget_cla = num_class

    return args, x_syn, y_syn


def accuracy(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


import psutil

def print_peak_memory(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cpu":

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = mem_info.rss / 1024**2   # MB
        vms = mem_info.vms / 1024**2   # MB
        print(f"CPU [RSS] : {rss:.2f} MB")
        print(f"CPU [VMS] : {vms:.2f} MB")
        return

    torch.cuda.synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print(f" [Allocated] : {peak_allocated / 1024**2:.2f} MB ({peak_allocated / 1024**3:.2f} GB)")
    print(f" [Reserved ] : {peak_reserved / 1024**2:.2f} MB ({peak_reserved / 1024**3:.2f} GB)")


