import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from topo_data import plot_topology_analysis, plot_client_comparison, plot_training_progress
import matplotlib.pyplot as plt
import warnings
from utils import partition_data, get_dataloader
import os
from sklearn.decomposition import PCA
import logging
from datetime import datetime
import umap
from matplotlib import colors
from geomloss import SamplesLoss 
from gudhi import representations 
import copy


# 主日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    handlers=[
        logging.FileHandler("topo_fed_run.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 固定 10 个类别的颜色
CMAP = plt.get_cmap('tab10', 10)
NORM = colors.Normalize(vmin=0, vmax=9)

class SimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):  # 增大隐藏层维度
        super(SimpleNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批量归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return features, output

def compute_feature_alignment_loss(client_features, global_features, scaling_factor=1.0):
    """
    计算客户端特征和全局特征的统计对齐损失（均值和协方差），并引入一个缩放因子来控制权重
    """
    # 计算均值差异
    client_features = normalize_features(client_features)
    global_features = normalize_features(global_features)
    client_mean = torch.mean(client_features, dim=0)
    global_mean = torch.mean(global_features, dim=0)
    mean_loss = torch.norm(client_mean - global_mean, p=2)
    
    # 计算协方差差异（添加小量防止奇异矩阵）
    client_cov = torch.cov(client_features.T + 1e-4 * torch.randn_like(client_features.T))
    global_cov = torch.cov(global_features.T + 1e-4 * torch.randn_like(global_features.T))
    cov_loss = torch.norm(client_cov - global_cov, p='fro')
    
    return scaling_factor * (mean_loss + 0.1 * cov_loss)


def compute_geometric_alignment_loss(client_features, global_features, scaling_factor=1.0, max_samples=256):
    """
    使用Wasserstein距离衡量分布几何差异（带采样）
    """
    # 随机采样
    idx_client = torch.randperm(client_features.size(0))[:max_samples]
    idx_global = torch.randperm(global_features.size(0))[:max_samples]
    
    client_sample = normalize_features(client_features[idx_client])
    global_sample = normalize_features(global_features[idx_global])

    # 计算Wasserstein距离
    wasserstein_loss = SamplesLoss(
        loss="sinkhorn", 
        p=2, 
        blur=0.05,
        scaling=0.8,       # 增加缩放系数加速收敛
        debias=False       # 关闭去偏置减少计算量
    )(client_sample, global_sample)
    
    return scaling_factor * wasserstein_loss

def compute_topological_loss(client_features, global_features, scaling_factor=1.0):
    """
    使用持续同调比较拓扑特征差异
    """
    # 转换为numpy
    client_features = normalize_features(client_features)
    global_features = normalize_features(global_features)
    client_np = client_features.cpu().detach().numpy()
    global_np = global_features.cpu().detach().numpy()
    
    # 计算持续图
    persistence = representations.PersistenceDiagram()
    client_pers = persistence.fit_transform([client_np])[0]
    global_pers = persistence.fit_transform([global_np])[0]
    
    # 使用Wasserstein距离比较持续图
    tda_loss = representations.WassersteinDistance(
        order=2, enable_autodiff=True
    ).fit([client_pers]).transform([global_pers])[0]
    
    return scaling_factor * torch.tensor(tda_loss, device=device)

def compute_graph_based_loss(client_features, global_features, k=10, scaling_factor=1.0):
    """
    通过k近邻图结构比较几何差异
    """
    # 构建客户端图
    client_features = normalize_features(client_features)
    global_features = normalize_features(global_features)
    client_dist = torch.cdist(client_features, client_features)
    client_topk = torch.topk(client_dist, k=k, largest=False, dim=1).values
    client_graph = torch.mean(client_topk)
    
    # 构建全局图
    global_dist = torch.cdist(global_features, global_features)
    global_topk = torch.topk(global_dist, k=k, largest=False, dim=1).values
    global_graph = torch.mean(global_topk)
    
    # 比较图结构差异
    graph_loss = torch.abs(client_graph - global_graph)
    
    return scaling_factor * graph_loss

class GeometricLossCalculator:
    def __init__(self, n_scales=3):
        self.scales = [0.1, 0.5, 1.0]  # 多尺度半径参数
        self.overlap_measures = []
        
    def _density_estimation(self, features, radius):
        """核密度估计"""
        pairwise_dist = torch.cdist(features, features)
        return torch.mean(torch.exp(-pairwise_dist**2 / (2 * radius**2)))
    
    def compute_multi_scale_loss(self, client_features, global_features):
        """
        多尺度几何相似度计算
        """
        total_loss = 0.0
        for scale in self.scales:
            # 计算每个尺度的密度分布差异
            client_features = normalize_features(client_features)
            global_features = normalize_features(global_features)
            client_density = self._density_estimation(client_features, scale)
            global_density = self._density_estimation(global_features, scale)
            
            # 累积多尺度损失
            total_loss += torch.abs(client_density - global_density)
            
        return total_loss / len(self.scales)

geometric_loss = GeometricLossCalculator()

def compute_geometric_cal_loss(client_features, global_features, scaling_factor=1.0):
    return scaling_factor * geometric_loss.compute_multi_scale_loss(client_features, global_features)


def normalize_features(features):
    return features / (torch.norm(features, dim=1, keepdim=True) + 1e-6)

def hybrid_alignment_loss(client_features, global_features, alpha=0.9):
    statistical_loss = compute_feature_alignment_loss(client_features, global_features)  # 原统计损失
    geometric_loss = compute_geometric_alignment_loss(client_features, global_features)  # 新几何损失
    return alpha * geometric_loss + (1 - alpha) * statistical_loss


def compute_feature_similarity(client, global_):
    # 计算余弦相似度
    client_mean = torch.mean(normalize_features(client), dim=0)
    global_mean = torch.mean(normalize_features(global_), dim=0)
    return torch.cosine_similarity(client_mean.unsqueeze(0), global_mean.unsqueeze(0))

def kde_entropy(features, bandwidth=0.1):
    """
    经验熵估计（特征空间KDE）：features为(N, d)张量
    返回经验熵的均值
    """
    N, d = features.size()
    # pairwise L2距离
    dist = torch.cdist(features, features)
    # 高斯核
    kernel = torch.exp(- dist**2 / (2 * bandwidth**2))
    # 每个点的概率密度（排除自身，避免自熵为无穷大）
    # 通常可加个极小常数防止log(0)
    p = (kernel.sum(dim=1) - 1) / (N - 1) + 1e-10
    logp = torch.log(p)
    entropy = -logp.mean()
    return entropy


def train_client(client_id, task_model, explore_model, train_loader, optimizer_task, optimizer_explore, global_features, round_idx, lambda_explore=0.1):
    task_model.train()
    explore_model.train()
    criterion = nn.CrossEntropyLoss()
    total_entropy = 0.0
    total_loss = 0.0  # 新增总损失变量
    total_task_loss = 0.0  # 新增任务损失变量
    total_explore_loss = 0.0  # 新增探索损失变量

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # 训练任务模型
        features_task, output_task = task_model(data)
        loss_task = criterion(output_task, target) 
        optimizer_task.zero_grad()
        loss_task.backward()
        optimizer_task.step()
        total_task_loss += loss_task.item()  # 累加任务损失

        # 训练探索模型
        _, output_explore = explore_model(data)
        prob = torch.softmax(output_explore, dim=1)
        entropy = kde_entropy(normalize_features(features_explore)) # 特征归一化后防止数值爆炸
        loss_explore = -lambda_explore * entropy
        optimizer_explore.zero_grad()
        loss_explore.backward()
        optimizer_explore.step()
        total_entropy += entropy.item()  # 累加熵值
        total_explore_loss += loss_explore.item()  # 累加探索损失
        features_explore, output_explore = explore_model(data)


    avg_entropy = total_entropy / len(train_loader)
    avg_task_loss = total_task_loss / len(train_loader)  # 平均任务损失
    avg_explore_loss = total_explore_loss / len(train_loader)  # 平均探索损失

    return avg_entropy, avg_task_loss, avg_explore_loss  # 返回多个值

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy


def create_run_directory():
    # 获取当前时间戳作为文件夹名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"results/{timestamp}_mnist_SimpleNet_clients4_rounds10"
    os.makedirs(folder_name, exist_ok=True)

    # 配置日志到文件夹（清空原有配置）
    log_path = os.path.join(folder_name, 'training.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 移除所有已有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # 添加文件处理器
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M'
    ))
    logger.addHandler(fh)
    
    # 添加控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger, folder_name

def gamma_scheduler(round_idx, total_rounds):
    """余弦退火调度"""
    return 0.5 * (1 + np.cos(np.pi * round_idx / total_rounds))

def federated_learning(n_clients=4, n_rounds=10, n_epochs=5, batch_size=32, lr=0.001, lambda_topo=0.1):  
    logger, run_dir = create_run_directory()  # 获取 logger
    
    # 初始化 client_features_list 和 client_targets_list
    client_features_list = []
    client_targets_list = []
    
    datadir = "C:/Users/crestiny/OneDrive/RESEARCH/Code/NIID-Bench-GANFL/data/"
    logdir = "logs/"
    os.makedirs(logdir, exist_ok=True)
    
    # 确保传递所有必要的参数
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset='mnist',
        datadir=datadir,
        logdir=logdir,
        partition='noniid-labeldir',
        n_parties=n_clients,
        beta=0.4
    )
    
    global_task_model = SimpleNet().to(device)  # 任务模型
    global_explore_model = SimpleNet().to(device)  # 探索模型
    client_models = [SimpleNet().to(device) for _ in range(n_clients)]
    explore_models = [SimpleNet().to(device) for _ in range(n_clients)]  # 新增探索模型
    history = {
        'client_task_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_explore_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_entropy': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_topo_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_topo_distance': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'topo_distances': [],
        'client_similarity': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'global_accuracy': [0.0] * (n_rounds + 1),  # 修改这里，确保足够空间存储每一轮的准确度
        'client_accuracy': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
    }

    MAX_SAMPLES = 1000  # 定义最大样本数
    rng = np.random.RandomState(42)  # 定义随机数生成器

    # 1. 生成初始全局特征
    global_task_model.eval()
    global_explore_model.eval()
    global_samples = []
    global_targets = []
    with torch.no_grad():
        global_train_loader, _, _ , _= get_dataloader(
            dataset='mnist',
            datadir=datadir,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=None  # 全部数据
        )
        for data, target in global_train_loader:
            data = data.to(device)
            features_task, _ = global_task_model(data)
            features_explore, _ = global_explore_model(data)
            global_samples.append(features_task.cpu())
            global_samples.append(features_explore.cpu())
            global_targets.append(target.numpy())  # 收集标签
    global_features = torch.cat(global_samples, dim=0).to(device)  # 存储为CPU张量
    global_targets = np.concatenate(global_targets, axis=0)  # 合并标签

    # ---- 在 federated_learning 最开始，做一次全局 fit ----
    umap_reducer = umap.UMAP(n_components=3, random_state=42).fit(global_features.cpu().numpy())

    # 从 0 到 n_rounds (包含 0)
    prev_train_loss = None
    for round_idx in range(0, n_rounds + 1):
        global_emb = None  # 初始化 global_emb 变量
        global_labels = None  # 初始化 global_labels 变量
        if round_idx > 0:
            logger.info(f"=== Round {round_idx}/{n_rounds} ===")  # 记录到日志
            
            # 同步全局模型参数到客户端
            for client_id in range(n_clients):
                client_models[client_id].load_state_dict(global_task_model.state_dict())
                explore_models[client_id].load_state_dict(global_explore_model.state_dict())

            
            client_states = []
            client_entropies = []  # 用于存储每个客户端的熵值
            for client_id in range(n_clients):
                _, _, train_ds, _ = get_dataloader(
                    dataset='mnist',
                    datadir=datadir,
                    train_bs=batch_size,
                    test_bs=batch_size,
                    dataidxs=net_dataidx_map[client_id]
                )
                train_dl = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True
                )
                
                optimizer_task = optim.Adam(client_models[client_id].parameters(), lr=lr)
                optimizer_explore = optim.Adam(explore_models[client_id].parameters(), lr=lr)  # 新增优化器
                avg_entropy, avg_task_loss, avg_explore_loss = train_client(
                    client_id=client_id,
                    task_model=client_models[client_id],
                    explore_model=explore_models[client_id],
                    train_loader=train_dl,
                    optimizer_task=optimizer_task,
                    optimizer_explore=optimizer_explore,
                    global_features=global_features,
                    round_idx=round_idx,
                    lambda_explore=0.1
                )
                client_entropies.append(avg_entropy)  # 记录熵值

                logger.info(f"Round {round_idx}, Client {client_id}: Task loss {avg_task_loss:.4f}, "
                          f"Explore loss {avg_explore_loss:.4f}, Entropy {avg_entropy:.4f}")  # 记录到日志

                # 更新历史记录
                history['client_task_loss'][client_id][round_idx] = avg_task_loss
                history['client_explore_loss'][client_id][round_idx] = avg_explore_loss
                history['client_entropy'][client_id][round_idx] = avg_entropy

                client_states.append(client_models[client_id].state_dict())
                client_states.append(explore_models[client_id].state_dict())

            # 任务模型FedAvg聚合
            task_states = [client_models[i].state_dict() for i in range(n_clients)]
            global_task_state = average_weights(task_states)
            global_task_model.load_state_dict(global_task_state)

            # 探索模型熵加权聚合
            explore_weights = torch.softmax(torch.tensor(client_entropies) * lambda_topo, dim=0)
            explore_states = [explore_models[i].state_dict() for i in range(n_clients)]
            global_explore_state = weighted_average(explore_states, explore_weights)
            global_explore_model.load_state_dict(global_explore_state)

            # 融合
            gamma = gamma_scheduler(round_idx, n_rounds)
            combined_state = combine_models(global_task_state, global_explore_state, gamma)
            global_task_model.load_state_dict(combined_state)

            # 更新全局特征
            global_task_model.eval()
            global_explore_model.eval()
            new_global_samples = []
            with torch.no_grad():
                for data, _ in global_train_loader:
                    data = data.to(device)
                    features_task, _ = global_task_model(data)
                    features_explore, _ = global_explore_model(data)
                    new_global_samples.append(features_task.cpu())
                    new_global_samples.append(features_explore.cpu())
            global_features = torch.cat(new_global_samples, dim=0).to(device)  # 更新全局特征

            # 测试集评估task model
            global_task_accuracy = evaluate_model(global_task_model, global_train_loader, device)
            logger.info(f"Global Task Model Accuracy at Round {round_idx}: {global_task_accuracy * 100:.2f}%")  # 记录到日志  
            history['global_task_accuracy'][round_idx] = global_task_accuracy

            # 测试机评估explore model
            global_explore_accuracy = evaluate_model(global_explore_model, global_train_loader, device)
            logger.info(f"Global Explore Model Accuracy at Round {round_idx}: {global_explore_accuracy * 100:.2f}%")  # 记录到日志
            history['global_explore_accuracy'][round_idx] = global_explore_accuracy

            # 客户端评估
            for client_id in range(n_clients):
                client_accuracy = evaluate_model(client_models[client_id], global_train_loader, device)
                logger.info(f"Client {client_id} Model Accuracy at Round {round_idx}: {client_accuracy * 100:.2f}%")  # 记录到日志
                history['client_accuracy'][client_id][round_idx] = client_accuracy

            # 全局特征可视化
            global_feats = global_features.cpu().numpy()
            global_labels = global_targets.copy()  # 使用副本避免修改原始数据
            print(f"Global features shape: {global_feats.shape}, Global labels shape: {global_labels.shape}")

            if len(global_feats) > MAX_SAMPLES:
                idx = rng.choice(len(global_feats), MAX_SAMPLES, replace=False)
                global_feats = global_feats[idx]
                global_labels = global_labels[:len(global_feats)]  # 确保 global_labels 的大小与 global_feats 一致

            global_emb = umap_reducer.transform(global_feats)  # 这里赋值
            fig_global = plot_topology_analysis(global_emb, global_labels)
            fig_global.savefig(os.path.join(run_dir, f'global_round_{round_idx}.png'))
            plt.close(fig_global)  # 确保每次图形被关闭
            
            # 将全局特征加入对比图
            if global_emb is not None and global_labels is not None:  # 确保不为 None
                all_features = client_features_list + [global_emb]
                all_targets = client_targets_list + [global_labels]  # 使用真实标签
                fig2 = plot_client_comparison(all_features, all_targets, n_clients + 1)
                fig2.savefig(os.path.join(run_dir, f'comparison_round_{round_idx}.png'))
                plt.close(fig2)  # 确保每次图形被关闭

        # --- 无论 round_idx 是不是 0，都画图 ---
        client_features_list = []
        client_targets_list = []
        for client_id in range(n_clients):
            # 提取这个 client 的全部特征
            _, _, ds, _ = get_dataloader(
                dataset='mnist',
                datadir=datadir,
                train_bs=batch_size,
                test_bs=batch_size,
                dataidxs=net_dataidx_map[client_id]
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            feats, targs = [], []
            with torch.no_grad():
                client_models[client_id].eval()
                for x, y in dl:
                    f_task, _ = client_models[client_id](x.to(device))
                    feats.append(f_task.cpu().numpy())
                    targs.append(y.numpy())
            feats = np.concatenate(feats, axis=0)
            targs = np.concatenate(targs, axis=0)

            # 确保 feats 和 targs 的大小一致
            if feats.shape[0] != targs.shape[0]:
                raise ValueError("Features and targets size mismatch.")

            # 下采样
            if feats.shape[0] > MAX_SAMPLES:
                idx = rng.choice(feats.shape[0], MAX_SAMPLES, replace=False)
                feats = feats[idx]
                targs = targs[idx]  # 确保 targs 也使用相同的索引

            # UMAP transform
            emb = umap_reducer.transform(feats)
            fig = plot_topology_analysis(emb, targs, client_id)
            fig.savefig(os.path.join(run_dir, f'client_{client_id}_round_{round_idx}.png'))
            plt.close(fig)  # 确保每次图形被关闭

            # 单独保存3D特征用于对比图
            client_features_list.append(emb)  # emb已经是3D坐标
            client_targets_list.append(targs)

            # 生成对比图时只使用3D视图
            if global_emb is not None and global_labels is not None:  # 确保不为 None
                all_features = client_features_list + [global_emb]
                all_targets = client_targets_list + [global_labels]  # 使用真实标签
                fig = plot_client_comparison(all_features, all_targets, n_clients + 1)
                fig.savefig(os.path.join(run_dir, f'comparison_round_{round_idx}.png'))
                plt.close(fig)  # 确保每次图形被关闭

    # 最后再画一下 loss 曲线
    fig3 = plot_training_progress(history, n_clients, n_rounds)
    fig3.savefig(os.path.join(run_dir, 'training_progress.png'))
    plt.close(fig3)  # 关闭图形
    
    return global_task_model, global_explore_model, history


def clean_diagram(dgm):
    if len(dgm) == 0:
        return dgm
    mask = np.all(np.isfinite(dgm), axis=1)
    return dgm[mask]

def prepare_point_cloud(X):
    if X.shape[1] > X.shape[0]:
        X = X.T
    return X

def plot_training_progress(history, n_clients, n_rounds):
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))  # 调整为3x2布局
    xs = range(1, n_rounds + 1)  # 从第1轮开始
    
    # 1. 总loss
    for client_id in range(n_clients):
        axs[0, 0].plot(xs, history['client_task_loss'][client_id][1:], label=f'Client {client_id}')
    axs[0, 0].set_title('Task Loss per Client')
    
    # 2. training loss
    for client_id in range(n_clients):
        axs[0, 1].plot(xs, history['client_explore_loss'][client_id][1:], label=f'Client {client_id}')
    axs[0, 1].set_title('Explore Loss per Client')
    
    # 3. similarity（新增）
    for client_id in range(n_clients):
        axs[1, 0].plot(xs, history['client_similarity'][client_id][1:], label=f'Client {client_id}')
    axs[1, 0].set_title('Feature Similarity')
    
    # 4. topo distance
    for client_id in range(n_clients):
        axs[1, 1].plot(xs, history['client_topo_distance'][client_id][1:], label=f'Client {client_id}')
    axs[1, 1].set_title('Topological Distance')
    
    # 5. topo loss
    for client_id in range(n_clients):
        axs[2, 0].plot(xs, history['client_topo_loss'][client_id][1:], label=f'Client {client_id}')
    axs[2, 0].set_title('Topology Loss')

    # 6. entropy
    for client_id in range(n_clients):
        axs[2, 1].plot(xs, history['client_entropy'][client_id][1:], label=f'Client {client_id}')
    axs[2, 1].set_title('Entropy')

    # 删除这行代码，因为我们需要显示熵值的图表
    
    for ax in axs.flat:
        ax.set_xlabel('Round')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_3d_features(ax, features, targets, client_id=None):
    """统一3D特征可视化函数（带边缘线）"""
    scatter = ax.scatter(
        features[:, 0], features[:, 1], features[:, 2],
        c=targets, 
        cmap=CMAP,
        norm=NORM,
        s=25,          # 增大点尺寸
        edgecolors='w', # 白色边缘
        linewidth=0.3,  # 边缘线宽
        alpha=0.9,      # 提高透明度
        depthshade=True # 启用深度阴影
    )
    ax.set_xlabel('UMAP1', fontsize=10, labelpad=8)
    ax.set_ylabel('UMAP2', fontsize=10, labelpad=8)
    ax.set_zlabel('UMAP3', fontsize=10, labelpad=8)
    ax.xaxis.pane.set_alpha(0.1)  # 半透明背景
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    title = f"Client {client_id}" if isinstance(client_id, int) else "Global"
    ax.set_title(title, fontsize=12, pad=10)

def plot_topology_analysis(features, targets, client_id=None):
    fig = plt.figure(figsize=(18, 6))  # 增加画布宽度
    fig.subplots_adjust(right=0.85)    # 调整右侧空间
    
    # 2D投影
    ax1 = fig.add_subplot(121)
    scatter2d = ax1.scatter(
        features[:, 0], features[:, 1], 
        c=targets, cmap=CMAP, norm=NORM,
        s=15, edgecolors='none', alpha=0.8
    )
    title = f"Client {client_id}" if client_id is not None else "Global"
    ax1.set_title(f"{title} (2D)", fontsize=12)
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')
    
    # 3D投影
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_features(ax2, features, targets, client_id)
    
    # 将colorbar移动到右侧独立区域
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(scatter2d, cax=cax, orientation='horizontal', label='Class')
    
    plt.tight_layout()
    return fig

def plot_client_comparison(client_features_list, client_targets_list, n_clients):
    """
    统一风格的对比图生成（优化布局）
    """
    nrows = int(np.ceil(np.sqrt(n_clients)))
    ncols = int(np.ceil(n_clients / nrows))
    
    fig = plt.figure(figsize=(6*ncols + 4, 5*nrows))  # 增加右侧空间
    fig.subplots_adjust(right=0.88)  # 调整整体布局
    
    # 绘制所有子图
    for idx, (features, targets) in enumerate(zip(client_features_list, client_targets_list)):
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')
        plot_3d_features(ax, features, targets, client_id=idx if idx < len(client_features_list)-1 else "Global")
    
    # 统一颜色条（右移并优化样式）
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 调整位置到最右侧
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Class Label', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=3.0)
    return fig

def get_global_subset(global_features, num=256):
    indices = torch.randperm(len(global_features))[:num]
    return global_features[indices].to(device)

def aggregate_models(client_entropies, client_task_models, client_explore_models, beta=1.0):
    # 任务模型使用FedAvg
    global_task_state = average_weights([m.state_dict() for m in client_task_models])
    
    # 探索模型使用熵加权
    weights = torch.softmax(torch.tensor(client_entropies) * beta, dim=0)
    global_explore_state = weighted_average([m.state_dict() for m in client_explore_models], weights)
    
    return global_task_state, global_explore_state

def combine_models(task_state, explore_state, gamma):
    """模型参数线性组合"""
    combined = {}
    for key in task_state:
        combined[key] = gamma * task_state[key] + (1 - gamma) * explore_state[key]
    return combined

def plot_dual_features(task_features, explore_features, labels):
    fig = plt.figure(figsize=(12,6))
    
    # 任务模型特征分布
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(task_features[:,0], task_features[:,1], task_features[:,2], c=labels, cmap='tab10')
    ax1.set_title('Task Model Features')
    
    # 探索模型特征分布
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(explore_features[:,0], explore_features[:,1], explore_features[:,2], 
                         c=labels, cmap='tab10')
    ax2.set_title('Explore Model Features')
    
    plt.colorbar(scatter, ax=ax2)
    return fig

def average_weights(w):
    """FedAvg聚合"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.true_divide(w_avg[key], len(w))
    return w_avg

def weighted_average(w, weights):
    """熵加权聚合"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = weights[0] * w[0][key]
        for i in range(1, len(w)):
            w_avg[key] += weights[i] * w[i][key]
    return w_avg

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行联邦学习
    global_task_model, global_explore_model, history = federated_learning(
        n_clients=4,
        n_rounds=6,
        n_epochs=3,      # 减少本地训练轮次
        batch_size=128,   # 增大批次减少采样次数
        lr=0.0005,       # 降低学习率
        lambda_topo=0.5  # 调整拓扑损失权重
    )