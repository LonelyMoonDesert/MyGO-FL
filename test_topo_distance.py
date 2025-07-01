# topo_distance_cifar10_fixed.py
"""
Demo: measure similarity of CIFAR‑10 samples via bottleneck distance
of cubical persistent homology on early ResNet‑18 feature maps.

Fixes compared with the previous version
---------------------------------------
1. Remove nonexistent `gw.bottleneck_distance`.
2. Use GUDHI‑top‑level `bottleneck_distance` (no POT dependency).
3. Drop the superfluous `persistence_dim_max` kwarg when calling
   `CubicalComplex.persistence()`.
"""
import os
import random
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import gudhi as gd
from gudhi import bottleneck_distance   # <‑‑ single source of truth
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ---------- 1. CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.expanduser("../data/")
N_PER_CLASS = 20  # images per class (total 200)
N_DIM = 1         # 0 → connected comp., 1 → holes
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- 2. DATA ----------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform
)

# sample a balanced subset -----------------------------------------------------
class_to_indices = {c: [] for c in range(10)}
for idx, (_, label) in enumerate(dataset):
    class_to_indices[label].append(idx)

subset_indices = list(
    itertools.chain.from_iterable(
        random.sample(v, N_PER_CLASS) for v in class_to_indices.values()
    )
)
subset = Subset(dataset, subset_indices)
loader = DataLoader(subset, batch_size=32, shuffle=False)

# ---------- 3. FEATURE EXTRACTOR ----------
resnet = torchvision.models.resnet18(weights="DEFAULT").to(DEVICE)
feature_net = torch.nn.Sequential(
    resnet.conv1, resnet.bn1, torch.nn.ReLU(inplace=False)
)  # output: 64×112×112
feature_net.eval()

def get_feature_map(batch):
    """Return numpy array [B,H,W] (channel‑mean)."""
    with torch.no_grad():
        feats = feature_net(batch.to(DEVICE)).cpu().numpy()
    return feats.mean(axis=1)

# ---------- 4. CUBICAL PH ----------

def cubical_diagram(img, dim=N_DIM):
    cub = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.ravel())
    cub.persistence(homology_coeff_field=2)
    return cub.persistence_intervals_in_dimension(dim)


def bottleneck(img_a, img_b, dim=N_DIM):
    return bottleneck_distance(cubical_diagram(img_a, dim), cubical_diagram(img_b, dim))

# ---------- 5. BARCODE CACHE ----------
barcode_cache, labels = [], []
print("Computing barcodes …")
for x, y in tqdm(loader):
    fmap = get_feature_map(x)
    for i in range(fmap.shape[0]):
        barcode_cache.append(cubical_diagram(fmap[i]))
        labels.append(int(y[i]))

# ---------- 6. PAIR SAMPLING ----------

def sample_pairs(n_pairs=1_000):
    """Return arrays of distances for same‑class and different‑class pairs."""
    same, diff = [], []
    while len(same) < n_pairs or len(diff) < n_pairs:
        i, j = random.sample(range(len(barcode_cache)), 2)
        d = bottleneck_distance(barcode_cache[i], barcode_cache[j])
        if labels[i] == labels[j]:
            if len(same) < n_pairs:
                same.append(d)
        else:
            if len(diff) < n_pairs:
                diff.append(d)
    return np.array(same), np.array(diff)

same_d, diff_d = sample_pairs()

print(f"Same‑class mean distance  : {same_d.mean():.4f}")
print(f"Different‑class mean dist.: {diff_d.mean():.4f}")

# ---------- 7. PLOT ----------
plt.hist(same_d, bins=30, alpha=0.7, label="Same class", density=True)
plt.hist(diff_d, bins=30, alpha=0.7, label="Different class", density=True)
plt.xlabel(f"Bottleneck distance (dim {N_DIM})")
plt.ylabel("Density")
plt.title("CIFAR‑10 Topological Distance")
plt.legend()
plt.tight_layout()
plt.show()
