#!/usr/bin/env python
# topo_probe.py
# ---------------------------------------------
# 通用拓扑可分辨力探针：任意 PyTorch 模型 & 多数据集
# ---------------------------------------------
import argparse, importlib, inspect, os, random, sys, subprocess
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gudhi as gd
from gudhi import bottleneck_distance
from sklearn.metrics import roc_auc_score, roc_curve

# ---------- 1. CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generic layer-wise topological discriminativeness probe")
    # 模型
    p.add_argument("--model_module",  type=str, help="e.g. torchvision.models or my_pkg.mynet")
    p.add_argument("--model_class",   type=str, help="e.g. resnet18 or SimpleCNNMNIST")
    p.add_argument("--weights",       type=str, default=None,
                   help="torchvision weight tag or 'path/to/ckpt.pth'")
    p.add_argument("--ckpt",          type=str, default=None,
                   help="state_dict checkpoint to load")
    # 数据
    p.add_argument("--dataset",       type=str, default="cifar10",
                   choices=["mnist","cifar10","cifar100","folder"])
    p.add_argument("--data_root",     type=str, default="../../data/")
    p.add_argument("--folder",        type=str, help="if --dataset folder")
    p.add_argument("--n_per_class",   type=int, default=100)
    # 层
    p.add_argument("--layers",        type=str, default="auto",
                   help="'auto' 或逗号分隔层名")
    # 拓扑 & 特征
    p.add_argument("--dims",          type=str, default="0",
                   help="持久同调维度列表，如 0 或 0,1")
    p.add_argument("--distance",      type=str, default="bottleneck",
                   choices=["bottleneck","wasserstein","pimage"])
    p.add_argument("--channel_mode",  type=str, default="mean",
                   help="mean | raw-K | pca-K")
    p.add_argument("--down",          type=int, default=32,
                   help="adaptive_pool output H=W size")
    # 采样 & 运行
    p.add_argument("--pairs",         type=int, default=1000,
                   help="同类/异类 pair 各多少")
    p.add_argument("--batch",         type=int, default=128)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int, default=2025)
    return p.parse_args()

args = parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

# ---------- 2. 可选依赖 ----------
try:
    from gudhi.wasserstein import wasserstein_distance as wdist
except ImportError:
    wdist = None
if args.distance == "wasserstein" and wdist is None:
    subprocess.check_call([sys.executable,"-m","pip","install","pot","--quiet"])
    from gudhi.wasserstein import wasserstein_distance as wdist

try:
    from persim import PersistenceImager
except ImportError:
    PersistenceImager = None
if args.distance == "pimage" and PersistenceImager is None:
    subprocess.check_call([sys.executable, "-m", "pip",
                           "install", "scikit-tda[persim]", "--quiet"])
    from persim import PersistenceImager

# ---------- 3. 数据加载 ----------
def build_dataset():
    tf = transforms.ToTensor()
    root = args.data_root
    if args.dataset=="mnist":
        ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    elif args.dataset=="cifar10":
        ds = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    elif args.dataset=="cifar100":
        ds = datasets.CIFAR100(root, train=False, download=True, transform=tf)
    elif args.dataset=="folder":
        assert args.folder, "--folder path required"
        ds = datasets.ImageFolder(args.folder,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.Grayscale(num_output_channels=3),
                                      tf]))
    else:
        raise ValueError
    return ds

dataset = build_dataset()
num_classes = len(dataset.classes) if hasattr(dataset, "classes") else len(set([y for _,y in dataset]))
cls2idx = defaultdict(list)
for idx, (_,lab) in enumerate(dataset):
    cls2idx[lab].append(idx)

subset_idx = []
for lab, idxs in cls2idx.items():
    take = min(args.n_per_class, len(idxs))
    subset_idx.extend(random.sample(idxs, take))
subset = Subset(dataset, subset_idx)
loader = DataLoader(subset, batch_size=args.batch, shuffle=False)
labels_global = []

# ---------- 4. 模型加载 ----------
def load_model():
    if args.model_module is None:
        raise ValueError("Need --model_module & --model_class")
    mod = importlib.import_module(args.model_module)
    cls = getattr(mod, args.model_class)
    # 兼容 torchvision 的权重
    if isinstance(args.weights, str) and args.weights and Path(args.weights).suffix=="":
        model = cls(weights=args.weights) if "weights" in inspect.signature(cls).parameters else cls()
    else:
        model = cls()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd)
    return model

net = load_model().to(args.device).eval()

# ---------- 5. 选层策略 ----------
if args.layers == "auto":
    candidate = [name for name, m in net.named_modules()
                 if len(list(m.children()))==0]  # 叶子模块
    # 过滤输出 ndim ≠ 4（非特征图）或 参数量太少
    layers = [n for n in candidate if
              any(k in n.lower() for k in ["conv","layer","stage","block"])]
else:
    layers = [s.strip() for s in args.layers.split(",")]

print(f"Probing layers: {layers}")

# ---------- 6. 持久同调工具 ----------
dims = [int(d) for d in args.dims.split(",")]
pi = PersistenceImager(pixel_size=0.005) if args.distance=="pimage" else None

def cubical_barcode(arr: np.ndarray, dim: int):
    cc = gd.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
    cc.persistence(homology_coeff_field=2)
    return cc.persistence_intervals_in_dimension(dim)

# ---------- 7. 钩子捕获 ----------
feat_dict = {l: [] for l in layers}
hooks = []
def gen_hook(name):
    def _hook(_, __, output):
        feat_dict[name].append(output.detach().cpu())
    return _hook

for n, m in net.named_modules():
    if n in layers:
        hooks.append(m.register_forward_hook(gen_hook(n)))

# ---------- 8. 预处理 & 条形图缓存 ----------
barcode_cache = {l:{d:[] for d in dims} for l in layers}
vector_cache  = {l:[] for l in layers}

def channel_reduce(tensor):
    if args.channel_mode=="mean":
        return tensor.mean(1, keepdim=True)
    elif args.channel_mode.startswith("raw"):
        k = int(args.channel_mode.split("-")[1])
        return tensor[:, :k]
    elif args.channel_mode.startswith("pca"):
        k = int(args.channel_mode.split("-")[1])
        B,C,H,W = tensor.shape
        flat = tensor.permute(0,2,3,1).reshape(-1,C).numpy()
        cov = np.cov(flat, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        pcs = vecs[:, -k:]
        red = torch.from_numpy((flat @ pcs).reshape(B,H,W,k).transpose(0,3,1,2))
        return red.to(tensor.device)
    else:
        raise ValueError

print("Extracting features & building barcodes …")
with torch.no_grad():
    for xb, yb in tqdm(loader):
        labels_global.extend(yb.tolist())
        xb = xb.to(args.device)
        _ = net(xb)   # forward：hook 自动填 feat_dict
        # 处理当前 batch 钩出的特征
        for lname in layers:
            feats = feat_dict[lname]; feat_dict[lname] = []  # pop
            for ft in feats:  # ft : [B,C,H,W] or [B,C,H,W] already list
                ft = channel_reduce(ft)
                if ft.shape[-1] != args.down:
                    ft = F.adaptive_avg_pool2d(ft, (args.down, args.down))
                B,C,H,W = ft.shape
                np_ft = ft.numpy()
                for bi in range(B):
                    if args.distance=="pimage":
                        bars = []
                        for c in range(C):
                            bars.extend(cubical_barcode(np_ft[bi,c], 0))
                        vec = pi.transform(np.array(bars)).ravel()
                        vector_cache[lname].append(vec)
                    else:
                        for d in dims:
                            bar_per_ch = [cubical_barcode(np_ft[bi,c], d) for c in range(C)]
                            barcode_cache[lname][d].append(bar_per_ch)

# ---------- 9. 距离函数 ----------
def dist_sig(a,b,dim):
    if args.distance=="bottleneck":
        return np.mean([bottleneck_distance(x,y) for x,y in zip(a,b)])
    elif args.distance=="wasserstein":
        return np.mean([wdist(x,y,p=2) for x,y in zip(a,b)])
    else:
        raise RuntimeError

from scipy.spatial.distance import cdist
def dist_vec(v1,v2):
    return cdist(v1[None],v2[None],"euclidean")[0,0]

# ----------10. Pair 采样 + AUC ----------
idx_pool = list(range(len(labels_global)))
results, raw_pairs = {}, {}
print("Sampling pairs & computing AUC …")
for lname in layers:
    results[lname] = {}
    for d in (dims if args.distance!="pimage" else [0]):
        same, diff = [], []
        while len(same) < args.pairs or len(diff) < args.pairs:
            i,j = random.sample(idx_pool,2)
            if args.distance=="pimage":
                dist = dist_vec(vector_cache[lname][i], vector_cache[lname][j])
            else:
                dist = dist_sig(barcode_cache[lname][d][i],
                                barcode_cache[lname][d][j], d)
            sim = -dist
            if labels_global[i]==labels_global[j]:
                if len(same)<args.pairs: same.append(sim)
            else:
                if len(diff)<args.pairs: diff.append(sim)
        y_true  = np.r_[np.ones(args.pairs), np.zeros(args.pairs)]
        y_score = np.array(same + diff)
        auc = roc_auc_score(y_true, y_score)
        results[lname][d] = auc
        raw_pairs[(lname,d)] = (y_true,y_score,same,diff)
        print(f"{lname:<20} dim{d}:  AUC={auc:.3f}")

# ----------11. 可视化 ----------
best_layer,best_dim,best_auc = max(((l,d,a) for l,v in results.items()
                                    for d,a in v.items()), key=lambda t:t[2])
print(f"\nBest layer → {best_layer}  dim{best_dim}  AUC={best_auc:.3f}")

plt.figure(figsize=(max(6,len(layers)*1.2), 3))
mat = np.full((len(dims), len(layers)), np.nan)
for i,d in enumerate(dims):
    for j,l in enumerate(layers):
        mat[i,j] = results[l].get(d,np.nan)
sns.heatmap(mat, annot=True, fmt=".3f",
            xticklabels=layers, yticklabels=[f"dim{d}" for d in dims],
            cmap="Blues")
plt.title("AUC heat-map"); plt.tight_layout(); plt.show()

y_true,y_score,same,diff = raw_pairs[(best_layer,best_dim)]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(same,bins=30,alpha=.6,label="same",density=True)
plt.hist(diff,bins=30,alpha=.6,label="diff",density=True)
plt.title(f"{best_layer} dim{best_dim}  AUC={best_auc:.3f}"); plt.legend()
plt.subplot(1,2,2)
fpr,tpr,_ = roc_curve(y_true,y_score)
plt.plot(fpr,tpr,label=f"AUC={best_auc:.3f}")
plt.plot([0,1],[0,1],'--',c='grey'); plt.xlabel("FPR");plt.ylabel("TPR")
plt.legend(); plt.tight_layout(); plt.show()
