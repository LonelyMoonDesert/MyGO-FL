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
from resnetcifar import ResNet18_cifar10  # 导入 ResNet 模型


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

def train_client(client_id, model, train_loader, optimizer, global_features, round_idx, n_rounds, prev_train_loss=None, history=None, alpha=0.9, lambda_topo=0.1):
    model.train()
    total_loss = 0
    total_training_loss = 0
    total_topo_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        features, output = model(data)
        
        # 计算分类损失
        loss = criterion(output, target)
        
        # 计算特征对齐损失
        global_subset = get_global_subset(global_features)  # 获取全局特征子集
        topo_loss = hybrid_alignment_loss(features, global_subset, alpha=alpha)  # 计算拓扑损失
        
        # 计算总损失
        total = loss + lambda_topo * topo_loss
        # total = loss
        
        total.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += total.item()
        total_training_loss += loss.item()
        total_topo_loss += topo_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_training_loss = total_training_loss / len(train_loader)
    avg_topo_loss = total_topo_loss / len(train_loader)
    
    # 计算并记录精度
    accuracy = evaluate_model(model, train_loader, device)  # 计算精度
    logging.info(f"Client {client_id}: Accuracy {accuracy:.4f}")  # 记录精度

    # 计算相似性
    similarity = compute_feature_similarity(features, global_subset)
    logging.info(f"Client {client_id}: Similarity {similarity.item():.4f}")  # 记录相似度

    # 在返回前计算拓扑距离
    with torch.no_grad():
        distance = 1 - similarity.item()  # 使用1 - 相似度作为距离
    
    # 添加日志记录
    logging.info(
        f"Round {round_idx} Client {client_id} | "
        f"Total Loss: {avg_loss:.4f} | "
        f"Train Loss: {avg_training_loss:.4f} | "
        f"Topo Loss: {avg_topo_loss:.4f} | "
        f"Similarity: {similarity.item():.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Distance: {distance:.4f}"
    )
    
    # 在客户端训练循环中更新历史记录
    if history is not None:
        history['client_similarity'][client_id][round_idx] = similarity.item()
        history['client_topo_distance'][client_id][round_idx] = distance
        history['client_accuracy'][client_id][round_idx] = accuracy  # 记录精度
    
    return avg_loss, avg_training_loss, avg_topo_loss, features, distance

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
    
    return folder_name

def federated_learning(n_clients=4, n_rounds=10, n_epochs=5, batch_size=32, lr=0.001, lambda_topo=0.1, alpha=0.9, dataset='mnist', model=SimpleNet):  
    # 创建输出文件夹
    run_dir = create_run_directory()
    
    # 初始化 client_features_list 和 client_targets_list
    client_features_list = []
    client_targets_list = []
    
    datadir = "C:/Users/crestiny/OneDrive/RESEARCH/Code/NIID-Bench-GANFL/data/"
    logdir = "logs/"
    os.makedirs(logdir, exist_ok=True)
    
    # 使用传入的 dataset 参数
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset=dataset,
        datadir=datadir,
        logdir=logdir,
        partition='noniid-labeldir',
        n_parties=n_clients,
        beta=0.4
    )
    
    global_model = model().to(device)  # 创建全局模型
    client_models = [model().to(device) for _ in range(n_clients)]  # 创建客户端模型
    
    history = {
        'client_total_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_training_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_topo_loss': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_topo_distance': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'topo_distances': [],
        'client_similarity': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
        'client_accuracy': [[0.0] * (n_rounds + 1) for _ in range(n_clients)],
    }

    MAX_SAMPLES = 1000  # 定义最大样本数
    rng = np.random.RandomState(42)  # 定义随机数生成器

    # 1. 生成初始全局特征
    global_model.eval()
    global_samples = []
    global_targets = []
    with torch.no_grad():
        global_train_loader, _, _ , _= get_dataloader(
            dataset=dataset,  # 使用传入的 dataset 参数
            datadir=datadir,
            train_bs=batch_size,
            test_bs=batch_size,
            dataidxs=None  # 全部数据
        )
        for data, target in global_train_loader:
            data = data.to(device)
            features, _ = global_model(data)
            global_samples.append(features.cpu())
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
            # 记录当前 Round 信息
            logging.info(f"\n=== Round {round_idx}/{n_rounds} ===")
            logging.info(f"Current alpha: {alpha:.4f}")  # 记录 alpha
            lambda_topo = min(0.1 * (round_idx / n_rounds), 1.0)  # 根据训练轮次动态调整
            logging.info(f"Current lambda_topo: {lambda_topo:.4f}")  # 记录 lambda_topo
            
            # 训练代码
            print(f"\n=== Round {round_idx}/{n_rounds} ===")
            # 同步全局模型参数到客户端
            for client_id in range(n_clients):
                client_models[client_id].load_state_dict(global_model.state_dict())
            
            client_states = []
            for client_id in range(n_clients):
                _, _, train_ds, _ = get_dataloader(
                    dataset=dataset,  # 使用传入的 dataset 参数
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
                
                optimizer = optim.Adam(client_models[client_id].parameters(), lr=lr)
                avg_total_loss, avg_training_loss, avg_topo_loss, features, distance = train_client(
                    client_id=client_id,
                    model=client_models[client_id],
                    train_loader=train_dl,
                    optimizer=optimizer,
                    global_features=global_features,
                    round_idx=round_idx,
                    n_rounds=n_rounds,
                    prev_train_loss=prev_train_loss,
                    history=history,
                    lambda_topo=lambda_topo,
                    alpha=alpha
                )

                print(f"Round {round_idx}, Client {client_id}: Total {avg_total_loss:.4f}, Train {avg_training_loss:.4f}, TopoLoss {avg_topo_loss:.4f}")

                # 更新历史记录
                history['client_total_loss'][client_id][round_idx] = avg_total_loss
                history['client_training_loss'][client_id][round_idx] = avg_training_loss
                history['client_topo_loss'][client_id][round_idx] = avg_topo_loss

                client_states.append(client_models[client_id].state_dict())

            # 聚合全局模型
            global_state = global_model.state_dict()
            for key in global_state:
                global_state[key] = torch.stack(
                    [state[key].float() for state in client_states], dim=0
                ).mean(dim=0)
            global_model.load_state_dict(global_state)

            # 更新全局特征
            global_model.eval()
            new_global_samples = []
            with torch.no_grad():
                for data, _ in global_train_loader:
                    data = data.to(device)
                    features, _ = global_model(data)
                    new_global_samples.append(features.cpu())
            global_features = torch.cat(new_global_samples, dim=0).to(device)  # 更新全局特征

            # 全局特征可视化
            global_feats = global_features.cpu().numpy()
            global_labels = global_targets.copy()  # 使用副本避免修改原始数据

            if len(global_feats) > MAX_SAMPLES:
                idx = rng.choice(len(global_feats), MAX_SAMPLES, replace=False)
                global_feats = global_feats[idx]
                global_labels = global_labels[idx]  # 同步下采样标签

            global_emb = umap_reducer.transform(global_feats)  # 这里赋值
            fig_global = plot_topology_analysis(global_emb, global_labels)
            fig_global.savefig(os.path.join(run_dir, f'global_round_{round_idx}.png'))
            plt.close(fig_global)  # 关闭图形
            
            # 将全局特征加入对比图
            if global_emb is not None and global_labels is not None:  # 确保不为 None
                all_features = client_features_list + [global_emb]
                all_targets = client_targets_list + [global_labels]  # 使用真实标签
                fig2 = plot_client_comparison(all_features, all_targets, n_clients + 1)
                fig2.savefig(os.path.join(run_dir, f'comparison_round_{round_idx}.png'))
                plt.close(fig2)  # 关闭图形

            # 测试集评估全局模型
            global_accuracy = evaluate_model(global_model, global_train_loader, device)
            logging.info(f"Global Model Accuracy at Round {round_idx}: {global_accuracy * 100:.2f}%")  # 记录全局模型精度

        # --- 无论 round_idx 是不是 0，都画图 ---
        client_features_list = []
        client_targets_list = []
        for client_id in range(n_clients):
            # 提取这个 client 的全部特征
            _, _, ds, _ = get_dataloader(
                dataset=dataset,  # 使用传入的 dataset 参数
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
                    f, _ = client_models[client_id](x.to(device))
                    feats.append(f.cpu().numpy())
                    targs.append(y.numpy())
            feats = np.concatenate(feats, axis=0)
            targs = np.concatenate(targs, axis=0)

            # 下采样
            if feats.shape[0] > MAX_SAMPLES:
                idx = rng.choice(feats.shape[0], MAX_SAMPLES, replace=False)
                feats = feats[idx]
                targs = targs[idx]

            # UMAP transform
            emb = umap_reducer.transform(feats)
            fig = plot_topology_analysis(emb, targs, client_id)
            fig.savefig(os.path.join(run_dir, f'client_{client_id}_round_{round_idx}.png'))
            plt.close(fig)

            # 单独保存3D特征用于对比图
            client_features_list.append(emb)  # emb已经是3D坐标
            client_targets_list.append(targs)

            # 生成对比图时只使用3D视图
            if global_emb is not None and global_labels is not None:  # 确保不为 None
                all_features = client_features_list + [global_emb]
                all_targets = client_targets_list + [global_labels]  # 使用真实标签
                fig = plot_client_comparison(all_features, all_targets, n_clients + 1)
                fig.savefig(os.path.join(run_dir, f'comparison_round_{round_idx}.png'))

    # 最后再画一下 loss 曲线
    fig3 = plot_training_progress(history, n_clients, n_rounds)
    fig3.savefig(os.path.join(run_dir, 'training_progress.png'))
    plt.close(fig3)  # 关闭图形
    
    return global_model, history


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
        axs[0, 0].plot(xs, history['client_total_loss'][client_id][1:], label=f'Client {client_id}')
    axs[0, 0].set_title('Total Loss per Client')
    
    # 2. training loss
    for client_id in range(n_clients):
        axs[0, 1].plot(xs, history['client_training_loss'][client_id][1:], label=f'Client {client_id}')
    axs[0, 1].set_title('Training Loss per Client')
    
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
    
    axs[2, 1].axis('off')  # 关闭最后一个子图
    
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

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行联邦学习
    global_model, history = federated_learning(
        n_clients=4,
        n_rounds=5,
        n_epochs=3,      # 减少本地训练轮次
        batch_size=128,   # 增大批次减少采样次数
        lr=0.0005,       # 降低学习率
        lambda_topo=0.5,  # 调整拓扑损失权重
        alpha=0.9,       # 设置 alpha 参数
        dataset='cifar10',  # 设置 dataset 参数
        model=ResNet18_cifar10  # 使用 ResNet 模型
    )