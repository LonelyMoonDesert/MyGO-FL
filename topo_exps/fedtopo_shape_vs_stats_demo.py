import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.db.backends.dummy.base import ignore
from gudhi.representations import PersistenceImage
import gudhi
import warnings

warnings.filterwarnings('ignore')

# 1. 配置参数
N_CLIENTS = 5
DIRICHLET_ALPHA = 0.1
N_CLASSES = 10
N_EPOCH = 2
BATCH_SIZE = 64
LAYER = 'conv1'    # 中间特征层
N_TEST_PER_CLASS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 2. 数据集non-IID划分（Dirichlet）
def split_dataset_dirichlet(dataset, alpha, n_clients, n_classes):
    """返回：每客户端样本索引列表"""
    labels = np.array(dataset.targets)
    idx_list = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha]*n_clients)
        # 归一保证每个客户端有样本
        proportions = np.array([p * (len(idx_c)-len(idx_list[i])) for i, p in enumerate(proportions)])
        proportions = (proportions / proportions.sum()) * len(idx_c)
        proportions = proportions.astype(int)
        proportions[-1] = len(idx_c) - proportions[:-1].sum()
        start = 0
        for i in range(n_clients):
            idx_list[i] += idx_c[start:start+proportions[i]].tolist()
            start += proportions[i]
    return idx_list

# 3. 数据集加载与切分
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
client_indices = split_dataset_dirichlet(dataset, DIRICHLET_ALPHA, N_CLIENTS, N_CLASSES)

# 4. 公共测试集（每类10张）
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
test_class_idxs = [[] for _ in range(N_CLASSES)]
for idx, (img, lbl) in enumerate(testset):
    if len(test_class_idxs[lbl]) < N_TEST_PER_CLASS:
        test_class_idxs[lbl].append(idx)
test_idxs = sum(test_class_idxs, [])
public_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, test_idxs),
    batch_size=100, shuffle=False
)

# 5. 模型定义与钩子
def get_resnet18():
    model = torchvision.models.resnet18(weights=None, num_classes=N_CLASSES)
    return model

def get_activations(model, images, layer=LAYER):
    feats = []
    handle = dict(model.named_modules())[layer].register_forward_hook(
        lambda m, _, out: feats.append(out.detach().cpu())
    )
    _ = model(images.to(DEVICE))
    handle.remove()
    return feats[0]  # shape [B, C, H, W]

# 6. 客户端本地微调
def local_finetune(global_model, dataset, indices, epochs=N_EPOCH):
    model = get_resnet18()
    model.load_state_dict(global_model.state_dict())
    model.to(DEVICE)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=BATCH_SIZE, shuffle=True
    )
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

# 7. 初始化全局模型
global_model = get_resnet18().to(DEVICE)
global_model.apply(lambda m: hasattr(m, 'reset_parameters') and m.reset_parameters())

# 8. 各客户端本地训练
client_models = []
for i in range(N_CLIENTS):
    print(f"训练客户端 {i+1} / {N_CLIENTS}")
    model_i = local_finetune(global_model, dataset, client_indices[i])
    client_models.append(model_i)

# 9. 公共图像激活提取
for x, y in public_loader:
    test_imgs = x
    break   # 一次取出全部（100张）

acts_clients = []
for i, model in enumerate(client_models):
    with torch.no_grad():
        acts = get_activations(model, test_imgs, layer=LAYER)   # [B,C,H,W]
    acts_clients.append(acts)   # N_CLIENTS × [B,C,H,W]

# 10. 统计漂移分析
mu_list, sigma_list = [], []
for acts in acts_clients:
    A = acts.numpy()   # [B,C,H,W]
    A = A.reshape(-1, A.shape[1])  # [B*H*W, C]
    mu = A.mean(axis=0)    # [C,]
    sigma = A.std(axis=0)  # [C,]
    mu_list.append(mu)
    sigma_list.append(sigma)

mu_arr = np.stack(mu_list)       # [N_CLIENTS, C]
sigma_arr = np.stack(sigma_list) # [N_CLIENTS, C]

# 11. 形状相似度：PI
pi = PersistenceImage(
    bandwidth=0.2,
    weight=lambda bd: float(bd[1] - bd[0]),  # 推荐权重写法
    resolution=[16,16]
)
pi.fit([np.array([[0.0, 1.0]], dtype=np.float32)])  # 防止im_range未初始化

def channel_pi(arr):
    cc  = gudhi.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
    bars= cc.persistence()
    diag= [p[1] for p in bars if p[0]==0 and p[1][1]>p[1][0]]
    if len(diag) == 0:
        return np.zeros(pi.resolution[0]*pi.resolution[1], dtype=np.float32)
    return pi.transform([np.array(diag, dtype=np.float32)]).ravel()

def pooled_pi_feats(acts, n_channels=0):
    if n_channels == 0: n_channels = acts.shape[1]
    pooled = nn.AdaptiveAvgPool2d((8,8))(torch.from_numpy(acts)) # [B,C,8,8]
    pooled = pooled.numpy()
    pis = []
    for c in range(n_channels):
        pi_c = []
        for b in range(pooled.shape[0]):
            v = channel_pi(pooled[b, c, :, :])
            # nan保护
            if np.isnan(v).any():
                # print(f"警告：PI特征包含nan！第{c}通道第{b}图像。返回零向量。")
                v = np.zeros_like(v)
            pi_c.append(v)
        pi_c = np.stack(pi_c, axis=0)  # [100,256]
        pis.append(pi_c.mean(axis=0))  # 取均值
    return np.stack(pis)   # [C, 256]

# 提取 PI 向量
pi_vecs = []
for idx, acts in enumerate(acts_clients):
    print(f'正在计算 PI 特征：客户端{idx+1}')
    pi_feats = pooled_pi_feats(acts.numpy())
    print('客户端PI均值/方差', np.mean(pi_feats), np.var(pi_feats))
    pi_vecs.append(pi_feats)

# PI 形状距离矩阵
def calc_pi_dist(pi_vecs):
    D = np.zeros((N_CLIENTS, N_CLIENTS))
    for i in range(N_CLIENTS):
        for j in range(N_CLIENTS):
            d = np.linalg.norm(pi_vecs[i]-pi_vecs[j])
            D[i, j] = d
    return D

D = calc_pi_dist(pi_vecs)

print("PI距离矩阵：\n", D)
print("PI距离向量：", D.flatten())

# 12. 量化指标
mean_var = np.var(mu_arr)
pi_var = np.var(D)

print("均值散布（方差）:", mean_var)
print("PI距离散布（方差）:", pi_var)
print("统计漂移/拓扑漂移比值:", mean_var/pi_var if pi_var > 0 else "N/A")

# 13. 可视化
plt.figure(figsize=(6,3))
sns.heatmap(mu_arr, cmap='viridis', cbar=True)
plt.xlabel('Channel'); plt.ylabel('Client')
plt.title('Activation mean')
plt.tight_layout()
plt.savefig('mean_heatmap.png', dpi=300)
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(D, cmap='Blues')
plt.colorbar()
plt.title('PI Euclidean distance across clients')
plt.tight_layout()
plt.savefig('pi_distance.png', dpi=300)
plt.show()

# 14. 直观对照
def plot_compare_channel(channel=0):
    fig, axes = plt.subplots(2, N_CLIENTS, figsize=(2.2*N_CLIENTS,4))
    for i in range(N_CLIENTS):
        act = acts_clients[i][0, channel]  # [H, W]
        act_tensor = torch.tensor(act).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        pool = nn.functional.adaptive_avg_pool2d(act_tensor, (8,8))[0,0].numpy()
        axes[0,i].imshow(pool, cmap='RdBu'); axes[0,i].set_title(f"Client{i+1} ch{channel}"); axes[0,i].axis('off')
        pi_img = pi_vecs[i][channel].reshape(16,16)
        axes[1,i].imshow(pi_img, cmap='gray'); axes[1,i].set_title("PI"); axes[1,i].axis('off')
    plt.suptitle(f"Channel {channel} visual & PI (sample 0)")
    plt.tight_layout()
    plt.savefig(f'channel{channel}_comp.png', dpi=300)
    plt.show()

# 手动测试
dummy_arr = np.random.rand(8,8)
print("单张随机测试PI向量:", channel_pi(dummy_arr))
plot_compare_channel(channel=0)

# 15. 保存量化数据
with open('fedtopo_metrics.txt', 'w') as f:
    f.write(f"均值方差: {mean_var:.4g}\n")
    f.write(f"PI 距离方差: {pi_var:.4g}\n")
    f.write(f"漂移方差比: {mean_var/pi_var if pi_var > 0 else 'N/A'}\n")

print("所有结果已保存（mean_heatmap.png, pi_distance.png, channel*_comp.png, fedtopo_metrics.txt）")
