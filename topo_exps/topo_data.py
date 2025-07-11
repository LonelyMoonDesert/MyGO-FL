import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ripser import ripser
from persim import plot_diagrams
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块

class TopoDataset(Dataset):
    def __init__(self, n_samples=1000, n_clients=4, noise=0.1, transform=None):
        """
        生成具有拓扑特性的合成数据集
        
        参数:
            n_samples: 每个客户端的数据量
            n_clients: 客户端数量
            noise: 高斯噪声的标准差
            transform: 数据转换函数
        """
        self.n_samples = n_samples
        self.n_clients = n_clients
        self.noise = noise
        self.transform = transform
        
        # 为每个客户端生成具有不同拓扑结构的数据
        self.data = []
        self.targets = []
        
        # 客户端1: 环形数据
        theta = np.linspace(0, 2*np.pi, n_samples)
        r = 1 + np.random.normal(0, noise, n_samples)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        self.data.append(np.column_stack((x1, x2)))
        self.targets.append(np.zeros(n_samples))
        
        # 客户端2: 双环形数据
        theta = np.linspace(0, 2*np.pi, n_samples)
        r1 = 1 + np.random.normal(0, noise, n_samples)
        r2 = 2 + np.random.normal(0, noise, n_samples)
        x1 = np.concatenate([r1 * np.cos(theta), r2 * np.cos(theta)])
        x2 = np.concatenate([r1 * np.sin(theta), r2 * np.sin(theta)])
        self.data.append(np.column_stack((x1, x2)))
        self.targets.append(np.ones(n_samples * 2))
        
        # 客户端3: 螺旋数据
        theta = np.linspace(0, 4*np.pi, n_samples)
        r = theta/4 + np.random.normal(0, noise, n_samples)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        self.data.append(np.column_stack((x1, x2)))
        self.targets.append(np.ones(n_samples) * 2)
        
        # 客户端4: 双螺旋数据
        theta = np.linspace(0, 4*np.pi, n_samples)
        r1 = theta/4 + np.random.normal(0, noise, n_samples)
        r2 = theta/4 + np.random.normal(0, noise, n_samples)
        x1 = np.concatenate([r1 * np.cos(theta), -r1 * np.cos(theta)])
        x2 = np.concatenate([r1 * np.sin(theta), -r1 * np.sin(theta)])
        self.data.append(np.column_stack((x1, x2)))
        self.targets.append(np.ones(n_samples * 2) * 3)
        
        # 合并所有客户端的数据
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            data = self.transform(data)
            
        return torch.FloatTensor(data), torch.LongTensor([target])[0]

def compute_persistence_diagram(data):
    """
    计算数据的持续同调图
    """
    diagrams = ripser(data)['dgms']
    return diagrams

def plot_topology_analysis(data, targets, client_idx=None):
    """
    可视化数据的拓扑结构
    参数:
        data: 特征数据（已采样和降维）
        targets: 标签
        client_idx: 客户端索引
    """
    # 只做UMAP和可视化
    reducer = umap.UMAP(n_components=3, random_state=42)
    data_3d = reducer.fit_transform(data)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 6))
    
    # 2D投影可视化
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(data_3d[:, 0], data_3d[:, 1], c=targets, cmap='tab10')
    ax1.set_title('UMAP 2D Projection')
    ax1.legend(*scatter.legend_elements(), title="Classes")
    
    # 3D可视化
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], 
                         c=targets, cmap='tab10')
    ax2.set_title('UMAP 3D Visualization')
    
    # 计算并绘制持续同调图
    ax3 = fig.add_subplot(133)
    diagrams = compute_persistence_diagram(data)
    plot_diagrams(diagrams, ax=ax3)
    ax3.set_title('Persistence Diagram')
    
    if client_idx is not None:
        plt.suptitle(f'Topology Analysis for Client {client_idx}')
    
    plt.tight_layout()
    return fig

def plot_client_comparison(client_features_list, client_targets_list, n_clients, nrows=2, ncols=2):
    """
    绘制多个客户端的特征对比
    """
    fig = plt.figure(figsize=(12, 12))

    for idx, (features, targets) in enumerate(zip(client_features_list, client_targets_list)):
        # 绘制 2D 图
        ax_2d = fig.add_subplot(nrows, ncols, idx + 1)
        scatter_2d = ax_2d.scatter(
            features[:, 0], features[:, 1], c=targets, cmap='tab10', s=10, edgecolors='none', alpha=0.7
        )
        ax_2d.set_title(f"Client {idx + 1} (2D)")
        ax_2d.set_xlabel('UMAP1')
        ax_2d.set_ylabel('UMAP2')
        fig.colorbar(scatter_2d, ax=ax_2d, label='Class')

        # 绘制 3D 图
        ax_3d = fig.add_subplot(nrows, ncols, idx + 1 + ncols, projection='3d')  # 创建一个新的3D轴
        scatter_3d = ax_3d.scatter(
            features[:, 0], features[:, 1], targets, c=targets, cmap='tab10', s=10, edgecolors='none', alpha=0.7
        )
        ax_3d.set_title(f"Client {idx + 1} (3D)")
        ax_3d.set_xlabel('UMAP1')
        ax_3d.set_ylabel('UMAP2')
        ax_3d.set_zlabel('Class')
        fig.colorbar(scatter_3d, ax=ax_3d, label='Class')

    plt.tight_layout()
    return fig

def plot_training_progress(history):
    """
    绘制训练过程中的拓扑对齐趋势
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制损失曲线
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['topo_loss'], label='Topology Loss')
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 绘制拓扑距离变化
    ax2.plot(history['topo_distances'], label='Average Topology Distance')
    ax2.set_title('Topology Alignment Progress')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Topology Distance')
    ax2.legend()
    
    plt.tight_layout()
    return fig 