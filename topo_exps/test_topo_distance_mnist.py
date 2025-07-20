# topo_distance_mnist_simplecnn.py
"""
MNIST similarity via topological signatures (SimpleCNNMNIST)
-----------------------------------------------------------
比较 SimpleCNNMNIST 各层 (conv1 / conv2) 特征的持久同调可分辨力。
默认距离：Bottleneck；支持 Wasserstein、Persistence Image（改 DISTANCE_MODE 即可）。
"""

# ---------- 0. 基础依赖 ----------
import os, random, itertools, sys, subprocess
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import gudhi as gd
from gudhi import bottleneck_distance
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------- 1. 你的网络 ----------
# 如果已经在别处定义，可改为:
#   from my_models import SimpleCNNMNIST
class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim=16*4*4, hidden_dims=(120, 84), output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)          # 28→24
        self.pool  = nn.MaxPool2d(2, 2)          # /2
        self.conv2 = nn.Conv2d(6, 16, 5)         # 12→8
        self.fc1   = nn.Linear(input_dim, hidden_dims[0])
        self.fc2   = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3   = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # 6×12×12
        x = self.pool(F.relu(self.conv2(x)))     # 16×4×4
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------- 2. 配置 ----------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_ROOT       = "../../data/"
N_PER_CLASS     = 100      # 每类取多少张测试图
BATCH_SIZE      = 128
PH_DIMS         = [0]      # MNIST 小图通常只算 0 维
CHANNEL_MODE    = "raw-1"  # 灰度图
DOWN_SIZE       = 12       # conv1 输出大小
DISTANCE_MODE   = "bottleneck"  # "bottleneck"|"wasserstein"|"pimage"
N_PAIRS         = 1_000    # 正负 pair 各多少

# ---------- 3. 可选依赖 ----------
try:
    from gudhi.wasserstein import wasserstein_distance as wdist
except ImportError:
    wdist = None

try:
    from persim import PersistenceImager
except ImportError:
    PersistenceImager = None

if DISTANCE_MODE == "pimage" and PersistenceImager is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-tda[persim]", "--quiet"])
    from persim import PersistenceImager
if DISTANCE_MODE == "wasserstein" and wdist is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pot", "--quiet"])
    from gudhi.wasserstein import wasserstein_distance as wdist

# ---------- 4. 数据 ----------
transform = transforms.ToTensor()
mnist = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

cls2idx = {c: [] for c in range(10)}
for idx, (_, lab) in enumerate(mnist):
    cls2idx[lab].append(idx)

subset_idx = list(itertools.chain.from_iterable(
    random.sample(v, N_PER_CLASS) for v in cls2idx.values()))
subset = Subset(mnist, subset_idx)
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
labels_global = []

# ---------- 5. 模型 & 截断层 ----------
net = SimpleCNNMNIST().to(DEVICE).eval()
layer_map = OrderedDict({
    "conv1": nn.Sequential(net.conv1),                    # 6×24×24
    "pool1": nn.Sequential(net.conv1, nn.ReLU(), net.pool),  # 6×12×12
    "conv2": nn.Sequential(
        net.conv1, nn.ReLU(), net.pool,
        net.conv2),                                       # 16×8×8
    "pool2": nn.Sequential(
        net.conv1, nn.ReLU(), net.pool,
        net.conv2, nn.ReLU(), net.pool),                  # 16×4×4
    # 全连接层：先走两次池化，再 flatten，最后取 fc1 / fc2
    "fc1": nn.Sequential(
        net.conv1, nn.ReLU(), net.pool,
        net.conv2, nn.ReLU(), net.pool,
        nn.Flatten(), net.fc1),                           # 120 维
    "fc2": nn.Sequential(
        net.conv1, nn.ReLU(), net.pool,
        net.conv2, nn.ReLU(), net.pool,
        nn.Flatten(), net.fc1, nn.ReLU(),
        net.fc2),                                         # 84 维
})


# ---------- 6. TDA 工具 ----------
def cubical_barcode(arr: np.ndarray, dim: int):
    cub = gd.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
    cub.persistence(homology_coeff_field=2)
    return cub.persistence_intervals_in_dimension(dim)

pi = PersistenceImager(pixel_size=0.005) if DISTANCE_MODE == "pimage" else None

# ---------- 7. 特征提取 & 条形图缓存 ----------
print("Extracting features & computing barcodes …")
barcode_cache = {l:{d:[] for d in PH_DIMS} for l in layer_map}
vector_cache  = {l:[] for l in layer_map}

for x, y in tqdm(loader):
    labels_global.extend(y.tolist())
    x = x.to(DEVICE)
    with torch.no_grad():
        for lname, lmodule in layer_map.items():
            feat = lmodule(x)  # [B,C,H,W]
            if lname == "conv2":             # conv2 后还有池化
                feat = net.pool(F.relu(feat))

            if feat.ndim == 2:  # [B, N]
                B, N = feat.shape
                side = int(np.ceil(np.sqrt(N)))  # 最近的整数边长
                pad = side * side - N  # 0-padding 到平方数
                feat = F.pad(feat, (0, pad))  # 右侧补 0
                feat = feat.view(B, 1, side, side)
            # 灰度 or raw-k 通道
            if CHANNEL_MODE.startswith("raw"):
                k = int(CHANNEL_MODE.split("-")[1])
                feat = feat[:, :k]
            else:  # mean
                feat = feat.mean(1, keepdim=True)
            if feat.shape[-1] != DOWN_SIZE:
                feat = F.adaptive_avg_pool2d(feat, (DOWN_SIZE, DOWN_SIZE))
            B,C,H,W = feat.shape
            f_np = feat.cpu().numpy()
            for bi in range(B):
                if DISTANCE_MODE == "pimage":
                    bars = []
                    for c in range(C):
                        bars.extend(cubical_barcode(f_np[bi,c], 0))
                    vec = pi.transform(np.array(bars)).ravel()
                    vector_cache[lname].append(vec)
                else:
                    for d in PH_DIMS:
                        bar_per_ch = [cubical_barcode(f_np[bi,c], d) for c in range(C)]
                        barcode_cache[lname][d].append(bar_per_ch)

# ---------- 8. 距离函数 ----------
def dist_sig(sigA,sigB,dim):
    if DISTANCE_MODE == "bottleneck":
        return np.mean([bottleneck_distance(a,b) for a,b in zip(sigA,sigB)])
    elif DISTANCE_MODE == "wasserstein":
        return np.mean([wdist(a,b,p=2) for a,b in zip(sigA,sigB)])
    else:
        raise RuntimeError

from scipy.spatial.distance import cdist
def dist_vec(v1,v2):
    return cdist(v1[None], v2[None], metric="euclidean")[0,0]

# ---------- 9. 采样 pair & 计算 AUC ----------
print("Sampling pairs & computing distances …")
idx_pool = list(range(len(labels_global)))
results, raw_pairs = {}, {}

for lname in layer_map:
    results[lname] = {}
    for d in (PH_DIMS if DISTANCE_MODE!="pimage" else [0]):
        same, diff = [], []
        while len(same) < N_PAIRS or len(diff) < N_PAIRS:
            i, j = random.sample(idx_pool, 2)
            if DISTANCE_MODE == "pimage":
                dist = dist_vec(vector_cache[lname][i], vector_cache[lname][j])
            else:
                dist = dist_sig(barcode_cache[lname][d][i],
                                barcode_cache[lname][d][j], d)
            sim = -dist
            if labels_global[i]==labels_global[j]:
                if len(same)<N_PAIRS: same.append(sim)
            else:
                if len(diff)<N_PAIRS: diff.append(sim)
        y_true  = np.r_[np.ones(N_PAIRS), np.zeros(N_PAIRS)]
        y_score = np.array(same + diff)
        auc = roc_auc_score(y_true, y_score)
        results[lname][d] = auc
        raw_pairs[(lname,d)] = (y_true,y_score,same,diff)
        print(f"{lname:<6} dim{d}:  AUC = {auc:.3f}")

# ----------10. 汇总 & 可视化 ----------
best_layer,best_dim,best_auc = max(
    ((l,d,a) for l,v in results.items() for d,a in v.items()),
    key=lambda t:t[2])
print(f"\nBest → {best_layer} dim{best_dim}  AUC={best_auc:.3f}")

# 热图
plt.figure(figsize=(8,3))
mat = np.full((len(PH_DIMS), len(layer_map)), np.nan)
for i,d in enumerate(PH_DIMS):
    for j,l in enumerate(layer_map):
        mat[i,j] = results[l].get(d,np.nan)
sns.heatmap(mat, annot=True, fmt=".3f",
            xticklabels=list(layer_map.keys()),
            yticklabels=[f"dim{d}" for d in PH_DIMS], cmap="Blues")
plt.title("AUC heat-map"); plt.tight_layout(); plt.show()

# 最优层分布 & ROC
y_true,y_score,same,diff = raw_pairs[(best_layer,best_dim)]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(same, bins=30, alpha=.6, label="same", density=True)
plt.hist(diff, bins=30, alpha=.6, label="diff", density=True)
plt.title(f"{best_layer} dim{best_dim}  AUC={best_auc:.3f}"); plt.legend()

plt.subplot(1,2,2)
fpr,tpr,_ = roc_curve(y_true,y_score)
plt.plot(fpr,tpr,label=f"AUC={best_auc:.3f}")
plt.plot([0,1],[0,1],'--',c='grey'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(); plt.tight_layout(); plt.show()
