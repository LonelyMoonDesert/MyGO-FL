# topo_distance_cifar10_optimized.py
"""
CIFAR‑10 similarity via topological signatures — **Extended playground**
=======================================================================
This version bundles **multiple optimisations** you might want to try:

1. **Deeper features** – pick any layer of ResNet‑18.
2. **Channel handling**
   • `mean`   : old behaviour (quickest)
   • `pca-K`  : keep top‑K principal components then PH per component
3. **Topological signature**
   • Raw barcodes + **bottleneck**  (default)
   • Raw barcodes + **p‑Wasserstein** (p=2) ➀
   • **Persistence Image** vector  + Euclidean / Cosine distance ➁
4. **Scalability knobs** – sample #images/pairs, down‑sampling size, etc.
5. **Full visualisation** – heat‑map of AUCs + ROC/Histogram for best combo.

➀ Needs `pot`; will auto‑install via pip if missing.
➁ Needs `scikit‑tda[persim]`; auto‑install if missing.

Runtime hint (Mac M2, 200 img, 2000 pair):
• mean + bottleneck   ≈ 3 min
• pca‑8 + bottleneck  ≈ 5 min
• persistence image   ≈ 2 min (no PH distance in loop)
"""
import os
import sys
import subprocess
import itertools
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision
import gudhi as gd
from gudhi import bottleneck_distance
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- optional deps ----------
try:
    from gudhi.wasserstein import wasserstein_distance as wdist
except ImportError:
    wdist = None

try:
    from persim import PersistenceImager
except ImportError:
    PersistenceImager = None

# ---------- 0. CONFIG --------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT     = os.path.expanduser("../../data/")
N_PER_CLASS   = 20      # images per CIFAR‑10 class
N_PAIRS       = 1_000   # pos & neg pairs each
DOWN_SIZE     = 32      # H,W fed to PH

LAYER_CHOICES = ["conv1", "layer2", "layer3", "layer4"]
PH_DIMS       = [0, 1]  # which homology dimensions

CHANNEL_MODE  = "pca-8"    # "mean"  |  "pca-8"  (top‑K PCs) | "raw-4" (first 4 channels)
DISTANCE_MODE = "bottleneck"   # "bottleneck" | "wasserstein" | "pimage"

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- 1. DATA ---------------
transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                       download=True, transform=transform)
cls2idx = {c: [] for c in range(10)}
for idx, (_, lab) in enumerate(dataset):
    cls2idx[lab].append(idx)
subset_idx = list(itertools.chain.from_iterable(
    random.sample(v, N_PER_CLASS) for v in cls2idx.values()))
subset = Subset(dataset, subset_idx)
loader = DataLoader(subset, batch_size=64, shuffle=False)
labels_global: list[int] = []

# ---------- 2. MODEL --------------
resnet = torchvision.models.resnet18(weights="DEFAULT").to(DEVICE).eval()
layer_map: OrderedDict[str, torch.nn.Module] = OrderedDict({
    "conv1": torch.nn.Sequential(
        resnet.conv1, resnet.bn1, torch.nn.ReLU(inplace=False), resnet.maxpool
    ),
    "layer2": torch.nn.Sequential(
        resnet.conv1, resnet.bn1, torch.nn.ReLU(inplace=False), resnet.maxpool,
        resnet.layer1, resnet.layer2
    ),
    "layer3": torch.nn.Sequential(
        resnet.conv1, resnet.bn1, torch.nn.ReLU(inplace=False), resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3
    ),
    "layer4": torch.nn.Sequential(
        resnet.conv1, resnet.bn1, torch.nn.ReLU(inplace=False), resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    ),
})
layer_map = OrderedDict((k, v) for k, v in layer_map.items() if k in LAYER_CHOICES)

# ---------- 3. TOPOLOGY UTILS -----

def cubical_barcode(img_np: np.ndarray, dim: int):
    cub = gd.CubicalComplex(dimensions=img_np.shape, top_dimensional_cells=img_np.ravel())
    cub.persistence(homology_coeff_field=2)
    return cub.persistence_intervals_in_dimension(dim)

if DISTANCE_MODE == "pimage" and PersistenceImager is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-tda[persim]", "--quiet"])
    from persim import PersistenceImager
if DISTANCE_MODE == "wasserstein" and wdist is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pot", "--quiet"])
    from gudhi.wasserstein import wasserstein_distance as wdist

# prepare persistence imager once
pi = None
if DISTANCE_MODE == "pimage":
    pi = PersistenceImager(pixel_size=0.005)  # default settings

# ---------- 4. FEATURE EXTRACTION & BARCODE CACHE ---------
print("Extracting features & computing signatures …")
barcode_cache = {layer: {d: [] for d in PH_DIMS} for layer in layer_map}
vector_cache  = {layer: [] for layer in layer_map}  # for persistence image mode

for x, y in tqdm(loader):
    labels_global.extend(y.tolist())
    x = x.to(DEVICE)
    with torch.no_grad():
        for lname, lmodule in layer_map.items():
            feat = lmodule(x)  # [B,C,H,W]
            # --- channel handling ---
            if CHANNEL_MODE.startswith("mean"):
                feat = feat.mean(1, keepdim=True)
            elif CHANNEL_MODE.startswith("pca"):
                k = int(CHANNEL_MODE.split("-")[1])
                B, C, H, W = feat.shape
                feat_flat = feat.permute(0,2,3,1).reshape(-1, C).cpu().numpy()
                # on‑the‑fly PCA using SVD of covariance
                cov = np.cov(feat_flat, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                pcs = eigvecs[:, -k:]
                feat = torch.from_numpy((feat_flat @ pcs).reshape(B, H, W, k).transpose(0,3,1,2)).to(feat.device)
            elif CHANNEL_MODE.startswith("raw"):
                k = int(CHANNEL_MODE.split("-")[1])
                feat = feat[:, :k, :, :]
            else:
                raise ValueError("Unknown CHANNEL_MODE")

            # spatial down‑sample
            if feat.shape[-1] != DOWN_SIZE:
                feat = F.adaptive_avg_pool2d(feat, (DOWN_SIZE, DOWN_SIZE))

            B, C, H, W = feat.shape
            feat_np = feat.cpu().numpy()
            for bi in range(B):
                if DISTANCE_MODE == "pimage":
                    # combine all dims barcodes -> vector
                    bars = []
                    for c in range(C):
                        bars.extend(cubical_barcode(feat_np[bi, c], 0))
                    vec = pi.transform(np.array(bars))
                    vector_cache[lname].append(vec.ravel())
                else:
                    for dim in PH_DIMS:
                        # merge channels: avg distance across channels later
                        bar_per_ch = [cubical_barcode(feat_np[bi, c], dim) for c in range(C)]
                        barcode_cache[lname][dim].append(bar_per_ch)

# ---------- 5. DISTANCE FUNCTIONS --------------

def distance_sig(sigA, sigB, dim):
    if DISTANCE_MODE == "bottleneck":
        # average bottleneck over channels
        return np.mean([bottleneck_distance(a, b) for a, b in zip(sigA, sigB)])
    elif DISTANCE_MODE == "wasserstein":
        return np.mean([wdist(a, b, p=2) for a, b in zip(sigA, sigB)])
    else:
        raise RuntimeError

if DISTANCE_MODE == "pimage":
    from scipy.spatial.distance import cdist
    def distance_vec(v1, v2):
        return cdist(v1[None], v2[None], metric="euclidean")[0,0]

# ---------- 6. EVALUATION ----------------------
print("Sampling pairs & computing distances …")
idx_pool = list(range(len(labels_global)))
results   = {}
raw_pairs = {}

for lname in layer_map:
    results[lname] = {}
    for dim in (PH_DIMS if DISTANCE_MODE!="pimage" else [0]):
        same_scores, diff_scores = [], []
        while len(same_scores) < N_PAIRS or len(diff_scores) < N_PAIRS:
            i, j = random.sample(idx_pool, 2)
            if DISTANCE_MODE == "pimage":
                d = distance_vec(vector_cache[lname][i], vector_cache[lname][j])
            else:
                d = distance_sig(barcode_cache[lname][dim][i],
                                 barcode_cache[lname][dim][j], dim)
            sim = -d  # higher = more similar
            if labels_global[i] == labels_global[j]:
                if len(same_scores) < N_PAIRS:
                    same_scores.append(sim)
            else:
                if len(diff_scores) < N_PAIRS:
                    diff_scores.append(sim)
        y_true  = np.array([1]*N_PAIRS + [0]*N_PAIRS)
        y_score = np.array(same_scores + diff_scores)
        auc_val = roc_auc_score(y_true, y_score)
        results[lname][dim] = auc_val
        raw_pairs[(lname, dim)] = (y_true, y_score, same_scores, diff_scores)
        print(f"{lname:<6}  dim {dim}:  AUC = {auc_val:.3f}")

# ---------- 7. SUMMARY + VIS ------------------
print("\n=== Summary (AUC) ===")
for lname in results:
    print(lname, results[lname])

best_layer, best_dim, best_auc = max(
    ((l,d,a) for l,v in results.items() for d,a in v.items()), key=lambda t:t[2])
print(f"\nBest → {best_layer} dim{best_dim}  AUC={best_auc:.3f}\n")

# heatmap
plt.figure(figsize=(10,4))
ax1 = plt.subplot(1,2,1)
mat = np.full((len(PH_DIMS), len(layer_map)), np.nan)
for i,d in enumerate(PH_DIMS):
    for j,l in enumerate(layer_map):
        mat[i,j] = results[l].get(d, np.nan)
_ = sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=list(layer_map.keys()),
                yticklabels=[f"dim{d}" for d in PH_DIMS], cmap="Blues", ax=ax1)
ax1.set_title("AUC heat‑map")

# best hist & ROC
best_y_true,best_y_score,best_same,best_diff = raw_pairs[(best_layer,best_dim)]
ax2 = plt.subplot(1,2,2)
ax2.hist(best_same, bins=30, alpha=0.6,label="same", density=True)
ax2.hist(best_diff, bins=30, alpha=0.6,label="diff", density=True)
ax2.set_title(f"Best {best_layer} dim{best_dim}\nAUC={best_auc:.3f}")
ax2.legend()
plt.tight_layout(); plt.show()

fpr,tpr,_ = roc_curve(best_y_true,best_y_score)
plt.figure(figsize=(4,4))
plt.plot(fpr,tpr,label=f"AUC={best_auc:.3f}")
plt.plot([0,1],[0,1],'--',color='grey')
plt.xlabel("FPR");plt.ylabel("TPR");plt.title("ROC curve")
plt.legend();plt.tight_layout();plt.show()
