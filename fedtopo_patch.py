# fedtopo_patch.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from utils import *
import gudhi
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import psutil


mem = psutil.virtual_memory()
print(f"[DEBUG] 当前总内存: {mem.total / 1024**3:.2f} GB")
print(f"[DEBUG] 当前可用内存: {mem.available / 1024**3:.2f} GB")
print(f"[DEBUG] 当前已用内存: {mem.used / 1024**3:.2f} GB")


def extract_layer_features(model, x, layer_name='layer3', pool_size=32, device='cuda'):
    """
    获取某模型某一层的输出，并池化到 pool_size×pool_size。
    返回: tensor, shape [B, C, pool_size, pool_size]
    """
    feats = []
    def hook_fn(module, input, output):
        feats.append(output.detach())
    # 找到目标层
    layer = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    _ = model(x.to(device))
    handle.remove()
    feat = feats[0]
    feat = nn.functional.adaptive_avg_pool2d(feat, pool_size)
    return feat

def batch_channel_pi(feat_bchw, K=8, pi=None):
    B, C, H, W = feat_bchw.shape
    pi_vecs = []
    feat_cpu = feat_bchw.cpu().detach()
    for i in range(B):
        vlist = []
        for k in range(min(K, C)):
            arr = feat_cpu[i, k].numpy()
            import gudhi
            cc = gudhi.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
            bars = cc.persistence()
            # 只取条形码 birth-death 二元组
            bars_pd = [pair[1] for pair in bars if len(pair) > 1 and pair[1][1] > pair[1][0]]
            if len(bars_pd) == 0:
                bars_pd = np.zeros((0, 2), dtype=np.float32)  # 空 shape
            else:
                bars_pd = np.array(bars_pd, dtype=np.float32).reshape(-1, 2)
            try:
                v = pi.transform([bars_pd]).ravel()
            except Exception:
                # 兼容 PI transform 的各版本行为
                v = np.zeros(pi.fit([np.array([[0, 0]])]).transform([np.array([[0, 0]])]).shape[1], dtype=np.float32)
            vlist.append(v)
        vlist = np.stack(vlist, 0)
        pi_vecs.append(vlist.mean(axis=0))
    pi_vecs = torch.tensor(np.stack(pi_vecs, 0), device=feat_bchw.device, dtype=torch.float32)
    return pi_vecs


def train_net_fedtopo(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      global_model, history, round, device="cpu", pi=None, K=8, pool_size=32, alpha=0.3):
    """
    FedTopo 训练核心新实现！彻底避免特征“短路”，使用PI signature欧氏loss。
    pi: gudhi.representations.PersistenceImage 实例, K: PCA分量数, alpha: topo loss权重
    """

    def get_optimizer(args_optimizer, net, lr, reg, rho):
        if args_optimizer == 'adam':
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg)
        elif args_optimizer == 'amsgrad':
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=reg, amsgrad=True)
        elif args_optimizer == 'sgd':
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=rho, weight_decay=reg)
        else:
            raise NotImplementedError

    logger = None
    try:
        import logging
        logger = logging.getLogger()
    except Exception:
        class DummyLogger:
            def info(self, *a, **kw): print(*a)
        logger = DummyLogger()

    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = get_optimizer(args_optimizer, net, lr, getattr(net, 'reg', 0.0), getattr(net, 'rho', 0.0))
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) != type([1]):
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        total_loss_collector, ce_loss_collector, topo_loss_collector = [], [], []
        for tmp in train_dataloader:
            for x, target in tmp:
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                # === 1. forward 得到 local/global PI 向量 ===
                local_feat  = extract_layer_features(net, x,  layer_name='layer3', pool_size=pool_size, device=device)
                global_feat = extract_layer_features(global_model, x, layer_name='layer3', pool_size=pool_size, device=device)
                local_pi  = batch_channel_pi(local_feat, K=K, pi=pi)   # [B, M]
                global_pi = batch_channel_pi(global_feat, K=K, pi=pi)  # [B, M]

                # === 2. Loss ===
                out = net(x)
                ce_loss = criterion(out, target)
                topo_loss = torch.norm(local_pi - global_pi, p=2) / x.shape[0]
                total_loss = ce_loss + alpha * topo_loss

                total_loss.backward()
                optimizer.step()

                total_loss_collector.append(total_loss.item())
                ce_loss_collector.append(ce_loss.item())
                topo_loss_collector.append(topo_loss.item())

        epoch_total_loss = np.mean(total_loss_collector)
        epoch_ce_loss = np.mean(ce_loss_collector)
        epoch_topo_loss = np.mean(topo_loss_collector)
        logger.info(f'Epoch: {epoch} Total: {epoch_total_loss:.4f} CE: {epoch_ce_loss:.4f} Topo: {epoch_topo_loss:.4f}')

    # 结尾照常统计 acc/记录 history ...
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')

    # 计算相似性
    similarity = compute_feature_similarity(local_features, global_features)
    logging.info(f">> Similarity {similarity.item():.4f}")  # 记录相似度

    # 在返回前计算拓扑距离
    with torch.no_grad():
        distance = 1 - similarity.item()  # 使用1 - 相似度作为距离

    # 更新历史记录
    history['client_total_loss'][net_id][round] = epoch_total_loss
    history['client_ce_loss'][net_id][round] = epoch_ce_loss
    history['client_topo_loss'][net_id][round] = epoch_topo_loss
    history['client_train_acc'][net_id][round] = train_acc
    history['client_test_acc'][net_id][round] = test_acc
    history['client_topo_distance'][net_id][round] = distance
    history['client_similarity'][net_id][round] = similarity.item()
    history['client_entropy'][net_id][round] = entropy

    return train_acc, test_acc
