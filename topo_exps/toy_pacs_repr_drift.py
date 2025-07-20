#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Toy experiment v4: Representation drift across PACS domains with memory-safe Rips + PI.

Usage example:
    python toy_pacs_repr_drift_v4.py \
        --data_root=../../data/pacs \
        --domain_a=photo --domain_b=sketch \
        --class_name=dog --max_imgs=200 \
        --layer=layer1 --spatial_pool=flatten \
        --flat_down=8 --rips_dim=32 --rips_q=0.5

Key options:
  * --layer: ResNet18 block to hook.
  * --spatial_pool=gap|flatten; flatten supports --flat_down spatial downsample.
  * --rips_dim: PCA dims before Rips (control cost).
  * --rips_q: distance quantile sets Rips max_edge_length.
  * --rips_max_pairs: #pairs sampled to estimate scale.
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# optional deps ---------------------------------------------------------
try:
    import gudhi as gd
except ImportError:  # pragma: no cover
    gd = None

try:
    from persim import PersistenceImager
except ImportError:  # pragma: no cover
    PersistenceImager = None

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# PACS dir helpers
# ----------------------------------------------------------------------
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
PACS_DOMAIN_ALIASES = {
    'photo':        ['photo', 'photos', 'Photo'],
    'art_painting': ['art_painting', 'art', 'painting', 'Art_Painting'],
    'cartoon':      ['cartoon', 'Cartoon'],
    'sketch':       ['sketch', 'Sketch'],
}


def resolve_pacs_root(root: str) -> str:
    """Return path that contains the 4 domain dirs."""
    cands = [
        os.path.join(root, 'pacs_data', 'pacs_data'),
        os.path.join(root, 'dct2_images', 'dct2_images'),
        root,
    ]
    for cand in cands:
        if all(os.path.isdir(os.path.join(cand, d)) for d in PACS_DOMAINS):
            return cand
    return root


def locate_pacs_domain(root: str, domain_key: str) -> str:
    for alias in PACS_DOMAIN_ALIASES.get(domain_key, [domain_key]):
        p = os.path.join(root, alias)
        if os.path.isdir(p):
            return p
    raise ValueError(f"Domain {domain_key} not found under {root}")


# ----------------------------------------------------------------------
# Dataset: 单域单类子集
# ----------------------------------------------------------------------
class PACSClassSubset(Dataset):
    def __init__(self, domain_dir, class_name, transform, max_imgs=None):
        self.transform = transform
        base = datasets.ImageFolder(domain_dir, transform=None)
        if class_name not in base.class_to_idx:
            raise ValueError(
                f"class_name '{class_name}' not in domain {domain_dir}; "
                f"available: {list(base.class_to_idx.keys())}"
            )
        cls_idx = base.class_to_idx[class_name]
        idxs = [i for i, (_, y) in enumerate(base.samples) if y == cls_idx]
        if max_imgs is not None:
            idxs = idxs[:max_imgs]
        self.paths = [base.samples[i][0] for i in idxs]
        self.targets = [cls_idx for _ in idxs]
        self.class_name = class_name
        self.domain_dir = domain_dir
        self.loader = base.loader  # PIL loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.loader(self.paths[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


# ----------------------------------------------------------------------
# Model + feature hook
# ----------------------------------------------------------------------
def build_resnet18(hook_layer='layer3', pretrained=True, device='cpu'):
    # torchvision >=0.13 weights API
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:  # pragma: no cover
            model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(weights=None)

    model.eval().to(device)
    feat_container = {}

    def _hook(m, inp, out):
        feat_container['feat'] = out.detach()

    if hasattr(model, hook_layer):
        getattr(model, hook_layer).register_forward_hook(_hook)
    else:  # fallback
        warnings.warn(f"ResNet18 has no attr {hook_layer}; using layer3.")
        model.layer3.register_forward_hook(_hook)

    return model, feat_container


def extract_features(model, feat_container, dataloader, device='cpu',
                     spatial_pool='gap', flat_down=0):
    """Forward + capture hook layer; pool to vector."""
    xs = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            _ = model(xb)
            feat = feat_container['feat']  # [B,C,H,W] or [B,C]
            if feat.dim() == 4:
                if spatial_pool == 'gap':
                    feat = torch.mean(feat, dim=(2, 3))  # [B,C]
                elif spatial_pool == 'flatten':
                    if flat_down and flat_down > 0:
                        feat = F.adaptive_avg_pool2d(feat, (flat_down, flat_down))
                    feat = feat.view(feat.size(0), -1)
                else:
                    raise ValueError("Unknown spatial_pool")
            xs.append(feat.cpu())
    return torch.cat(xs, dim=0).numpy()


# ----------------------------------------------------------------------
# Rips utilities
# ----------------------------------------------------------------------
def reduce_dim_for_rips(points, out_dim=32, random_state=0):
    """PCA -> out_dim; skip if out_dim<=0 or >=D."""
    X = np.asarray(points, dtype=np.float32)
    D = X.shape[1]
    if out_dim <= 0 or out_dim >= D:
        return X, None
    pca = PCA(n_components=out_dim, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z.astype(np.float32), pca


def estimate_max_edge_length_safe(points, q=1.0, max_pairs=50000, rng_seed=0):
    """Sampled pairwise dist quantile; memory-safe."""
    pts = np.asarray(points, dtype=np.float32)
    n = pts.shape[0]
    if n <= 1:
        return 0.0
    total_pairs = n * (n - 1) // 2
    m = min(max_pairs, total_pairs)
    rng = np.random.default_rng(rng_seed)
    idx_i = rng.integers(0, n, size=m)
    idx_j = rng.integers(0, n, size=m)
    d = np.linalg.norm(pts[idx_i] - pts[idx_j], axis=1)
    mel = np.quantile(d, q)
    return float(max(mel, 1e-8))


def rips_persistence(points, max_dim=1, max_edge_length=1.0):
    if gd is None:
        raise ImportError("gudhi not installed; pip install gudhi")
    rc = gd.RipsComplex(points=points, max_edge_length=float(max_edge_length))
    st = rc.create_simplex_tree(max_dimension=max_dim)
    st.persistence()
    out = {}
    for d in range(max_dim + 1):
        pd = st.persistence_intervals_in_dimension(d)
        out[d] = np.asarray(pd, dtype=np.float32)
    return out


def sanitize_intervals_dict(pd_dict, clip=None):
    """Merge dims; drop inf/nan; clip death<=clip."""
    arrs = []
    for _, intervals in pd_dict.items():
        if intervals is None or len(intervals) == 0:
            continue
        a = np.asarray(intervals, dtype=np.float64)
        mask = np.isfinite(a).all(axis=1)
        a = a[mask]
        if a.size == 0:
            continue
        if clip is not None:
            a[:, 1] = np.minimum(a[:, 1], clip)
        a = a[a[:, 1] > a[:, 0]]
        if a.size == 0:
            continue
        arrs.append(a.astype(np.float32))
    if not arrs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(arrs)


# ----------------------------------------------------------------------
# Persistence metrics
# ----------------------------------------------------------------------
def persistence_entropy(intervals):
    if intervals is None or intervals.size == 0:
        return 0.0
    a = np.asarray(intervals, dtype=np.float64)
    lens = a[:, 1] - a[:, 0]
    lens = lens[lens > 0]
    if lens.size == 0:
        return 0.0
    s = lens.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.0
    p = lens / s
    return float(-(p * np.log(p + 1e-12)).sum())


def build_pi_object(res=(25, 25), sigma=0.1):
    if PersistenceImager is None:
        return None, False
    try:  # new-style
        pi_obj = PersistenceImager(pixels=res, kernel_params={'sigma': sigma})
        return pi_obj, True
    except TypeError:  # pragma: no cover
        pass
    try:  # old-style
        pi_obj = PersistenceImager(resolution=res, sigma=sigma)
        return pi_obj, True
    except TypeError:  # pragma: no cover
        pass
    return None, False


def intervals_to_pi_pair(bars_a, bars_b, resolution=(30, 30)):
    """Shared-grid histogram PI for two diagrams."""
    H, W = int(resolution[0]), int(resolution[1])
    arrs = []
    for arr in (bars_a, bars_b):
        if arr is not None and arr.size:
            arr = np.asarray(arr, dtype=np.float32)
            arr = arr[arr[:, 1] > arr[:, 0]]
            if arr.size:
                arrs.append(arr)
    if not arrs:
        z = np.zeros(H * W, dtype=np.float32)
        return z, z
    all_arr = np.vstack(arrs)
    bmin, dmin = all_arr.min(axis=0)
    bmax, dmax = all_arr.max(axis=0)
    eps = 1e-6
    bx = np.linspace(bmin, bmax + eps, H + 1, dtype=np.float32)
    dy = np.linspace(dmin, dmax + eps, W + 1, dtype=np.float32)

    def _hist(arr):
        if arr is None or arr.size == 0:
            return np.zeros(H * W, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr[arr[:, 1] > arr[:, 0]]
        if arr.size == 0:
            return np.zeros(H * W, dtype=np.float32)
        h, _, _ = np.histogram2d(arr[:, 0], arr[:, 1], bins=[bx, dy])
        return h.astype(np.float32).ravel()

    return _hist(bars_a), _hist(bars_b)


# ----------------------------------------------------------------------
# Embedding (3D)
# ----------------------------------------------------------------------
def embed_3d(points, use_umap=False, random_state=0):
    if use_umap and umap is not None:
        reducer = umap.UMAP(n_components=3, random_state=random_state)
        return reducer.fit_transform(points)
    pca = PCA(n_components=3)
    return pca.fit_transform(points)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_scatter_3d(z_a, z_b, label_a, label_b, out_path=None, show=True):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_a[:, 0], z_a[:, 1], z_a[:, 2], s=20, c='C0', alpha=0.7, label=label_a)
    ax.scatter(z_b[:, 0], z_b[:, 1], z_b[:, 2], s=20, c='C1', alpha=0.7, label=label_b)
    ax.set_title(f"3D embedding: {label_a} vs {label_b}")
    ax.legend()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def _plot_barcode_1d(intervals, ax, title=""):
    if intervals.size == 0:
        ax.text(0.5, 0.5, "(empty)", ha='center', va='center')
        ax.set_axis_off()
        return
    for i, (b, d) in enumerate(intervals):
        ax.plot([b, d], [i, i], '-', color='k', lw=1)
    ax.set_xlabel("filtration")
    ax.set_ylabel("bars")
    ax.set_title(title)


def plot_barcodes(pd_a, pd_b, label_a, label_b, out_path=None, show=True):
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))  # dims 0,1
    dims = [0, 1]
    for j, d in enumerate(dims):
        ax_a = axes[0, j]
        ax_b = axes[1, j]
        _plot_barcode_1d(pd_a.get(d, np.zeros((0, 2))), ax_a, title=f"{label_a} H{d}")
        _plot_barcode_1d(pd_b.get(d, np.zeros((0, 2))), ax_b, title=f"{label_b} H{d}")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_pi_heatmaps(pi_a, pi_b, label_a, label_b, res, out_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(pi_a.reshape(res), origin='lower', cmap='viridis')
    axes[0].set_title(label_a)
    axes[1].imshow(pi_b.reshape(res), origin='lower', cmap='viridis')
    axes[1].set_title(label_b)
    for ax in axes:
        ax.axis('off')
    fig.suptitle("Persistence Images")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True,
                    help="root to pacs/ (folder containing pacs_data/ or dct2_images/)")
    ap.add_argument('--domain_a', type=str, default='photo', choices=PACS_DOMAINS)
    ap.add_argument('--domain_b', type=str, default='sketch', choices=PACS_DOMAINS)
    ap.add_argument('--class_name', type=str, default='dog')
    ap.add_argument('--max_imgs', type=int, default=200,
                    help="max images per domain to load (None=all)")
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--layer', type=str, default='layer3',
                    help="resnet18 layer to hook: conv1|layer1|layer2|layer3|layer4|avgpool")
    ap.add_argument('--spatial_pool', type=str, default='gap', choices=['gap', 'flatten'])
    ap.add_argument('--flat_down', type=int, default=16,
                    help="spatial size before flatten (only if spatial_pool=flatten; 0=disable)")
    ap.add_argument('--use_umap', type=int, default=0, help="1=UMAP,0=PCA")
    ap.add_argument('--pi_res', type=int, default=25, help="PI resolution (square)")
    ap.add_argument('--pi_sigma', type=float, default=0.1, help="PI gaussian spread (if persim)")
    ap.add_argument('--rips_dim', type=int, default=32,
                    help="PCA dim before Rips; 0=use full dim")
    ap.add_argument('--rips_q', type=float, default=1.0,
                    help="quantile in (0,1] to set max_edge_length for Rips; 1.0=max")
    ap.add_argument('--rips_max_pairs', type=int, default=50000,
                    help="max random pairs for Rips scale estimate")
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_prefix', type=str, default=None,
                    help="if set, save figures to <prefix>_*.png")
    ap.add_argument('--plot_raw_barcodes', type=int, default=0,
                    help="1=plot raw barcode (may be messy)")
    args = ap.parse_args()

    device = args.device

    # --------------------- data ---------------------
    root = resolve_pacs_root(args.data_root)
    dom_a_dir = locate_pacs_domain(root, args.domain_a)
    dom_b_dir = locate_pacs_domain(root, args.domain_b)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    ds_a = PACSClassSubset(dom_a_dir, args.class_name, tfm, max_imgs=args.max_imgs)
    ds_b = PACSClassSubset(dom_b_dir, args.class_name, tfm, max_imgs=args.max_imgs)
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=False)
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Loaded domain {args.domain_a}: {len(ds_a)} images of class '{args.class_name}'.")
    print(f"[INFO] Loaded domain {args.domain_b}: {len(ds_b)} images of class '{args.class_name}'.")

    # --------------------- model & features ---------------------
    model, feat_container = build_resnet18(pretrained=True,
                                           hook_layer=args.layer,
                                           device=device)
    feats_a = extract_features(model, feat_container, dl_a,
                               device=device, spatial_pool=args.spatial_pool,
                               flat_down=args.flat_down)
    feats_b = extract_features(model, feat_container, dl_b,
                               device=device, spatial_pool=args.spatial_pool,
                               flat_down=args.flat_down)
    print(f"[INFO] feature shapes: A {feats_a.shape}, B {feats_b.shape}")

    # --------------------- choose embedding data ---------------------
    # If dimensionality huge, embed on reduced copy; else original.
    if feats_a.shape[1] > 5000 or feats_b.shape[1] > 5000:
        emb_a_src, _ = reduce_dim_for_rips(feats_a, out_dim=min(128, feats_a.shape[1]))
        emb_b_src, _ = reduce_dim_for_rips(feats_b, out_dim=min(128, feats_b.shape[1]))
    else:
        emb_a_src, emb_b_src = feats_a, feats_b
    z_a = embed_3d(emb_a_src, use_umap=bool(args.use_umap))
    z_b = embed_3d(emb_b_src, use_umap=bool(args.use_umap))

    scatter_path = args.out_prefix + "_scatter.png" if args.out_prefix else None
    plot_scatter_3d(z_a, z_b, args.domain_a, args.domain_b,
                    out_path=scatter_path, show=True)

    # --------------------- reduce for Rips ---------------------
    feats_a_r, _ = reduce_dim_for_rips(feats_a, out_dim=args.rips_dim)
    feats_b_r, _ = reduce_dim_for_rips(feats_b, out_dim=args.rips_dim)
    print(f"[INFO] Rips features dim: A {feats_a_r.shape}, B {feats_b_r.shape}")

    # --------------------- Rips persistence ---------------------
    mel_a = estimate_max_edge_length_safe(feats_a_r, q=args.rips_q,
                                          max_pairs=args.rips_max_pairs)
    mel_b = estimate_max_edge_length_safe(feats_b_r, q=args.rips_q,
                                          max_pairs=args.rips_max_pairs)
    mel = max(mel_a, mel_b)
    print(f"[INFO] Rips max_edge_length (q={args.rips_q}): A={mel_a:.3f}, B={mel_b:.3f} -> use {mel:.3f}")

    pd_a = rips_persistence(feats_a_r, max_dim=1, max_edge_length=mel)
    pd_b = rips_persistence(feats_b_r, max_dim=1, max_edge_length=mel)

    if args.plot_raw_barcodes:
        bc_path = args.out_prefix + "_barcodes_raw.png" if args.out_prefix else None
        plot_barcodes(pd_a, pd_b, args.domain_a, args.domain_b,
                      out_path=bc_path, show=True)

    # --------------------- sanitize intervals ---------------------
    bars_a = sanitize_intervals_dict(pd_a, clip=mel)
    bars_b = sanitize_intervals_dict(pd_b, clip=mel)
    print(f"[INFO] sanitized bars: A {bars_a.shape}, B {bars_b.shape}")

    # --------------------- PI ---------------------
    res = (args.pi_res, args.pi_res)
    sigma = args.pi_sigma
    pi_obj, use_persim = build_pi_object(res=res, sigma=sigma)
    if use_persim:
        pi_obj.fit([bars_a, bars_b])
        pi_a = pi_obj.transform([bars_a])[0].ravel()
        pi_b = pi_obj.transform([bars_b])[0].ravel()
    else:
        pi_a, pi_b = intervals_to_pi_pair(bars_a, bars_b, resolution=res)

    pi_path = args.out_prefix + "_pi.png" if args.out_prefix else None
    plot_pi_heatmaps(pi_a, pi_b, args.domain_a, args.domain_b, res,
                     out_path=pi_path, show=True)

    # --------------------- metrics ---------------------
    pi_dist = float(np.linalg.norm(pi_a - pi_b))
    pe_a = persistence_entropy(bars_a)
    pe_b = persistence_entropy(bars_b)

    print("==== Topology summary ====")
    print(f"PI L2 distance ({args.domain_a} vs {args.domain_b}): {pi_dist:.4f}")
    print(f"Persistence Entropy {args.domain_a}: {pe_a:.4f}")
    print(f"Persistence Entropy {args.domain_b}: {pe_b:.4f}")
    print("==========================")


if __name__ == "__main__":
    main()
