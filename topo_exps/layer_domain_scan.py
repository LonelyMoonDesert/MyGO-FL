#!/usr/bin/env python
"""
Layer & Domain Scan Helper for FedTopo toy experiments.

Functions
---------
1. layer_scan(): sweep over conv1/layer1/layer2/layer3 on a given domain pair;
   outputs CSV and optional line plot of PI distance vs layer.

2. domain_matrix(): for a fixed layer, compute PI distance for all 4 PACS domains
   pair‑wise; outputs CSV matrix + seaborn heatmap.

Both reuse toy_pacs_repr_drift_v4 functions by import. Requires that script is on PYTHONPATH.

Example
-------
python layer_domain_scan.py layer_scan \
    --data_root ../../data/pacs \
    --domain_a photo --domain_b sketch \
    --class_name dog --max_imgs 200 \
    --spatial_pool flatten --flat_down 8 \
    --rips_dim 32 --rips_q 0.5

python layer_domain_scan.py domain_matrix \
    --data_root ../../data/pacs \
    --layer layer1 --class_name dog --max_imgs 200 \
    --spatial_pool flatten --flat_down 8 --rips_dim 32 --rips_q 0.5
"""

import argparse
import csv
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
# import helper functions from v4 script
from toy_pacs_repr_drift import (
    PACS_DOMAINS, resolve_pacs_root, locate_pacs_domain, build_resnet18,
    extract_features, reduce_dim_for_rips, sanitize_intervals_dict,
    build_pi_object, intervals_to_pi_pair, estimate_max_edge_length_safe,
    rips_persistence)


def pi_distance(features_a, features_b, rips_q, rips_dim, rips_max_pairs):
    # reduce dim
    fa_r, _ = reduce_dim_for_rips(features_a, out_dim=rips_dim)
    fb_r, _ = reduce_dim_for_rips(features_b, out_dim=rips_dim)
    # mel
    mel_a = estimate_max_edge_length_safe(fa_r, q=rips_q, max_pairs=rips_max_pairs)
    mel_b = estimate_max_edge_length_safe(fb_r, q=rips_q, max_pairs=rips_max_pairs)
    mel = max(mel_a, mel_b)
    # rips
    pd_a = rips_persistence(fa_r, max_dim=1, max_edge_length=mel)
    pd_b = rips_persistence(fb_r, max_dim=1, max_edge_length=mel)
    bars_a = sanitize_intervals_dict(pd_a, clip=mel)
    bars_b = sanitize_intervals_dict(pd_b, clip=mel)
    # PI
    pi_a, pi_b = intervals_to_pi_pair(bars_a, bars_b, resolution=(25, 25))
    return float(np.linalg.norm(pi_a - pi_b))


def compute_features(domain, class_name, args, cache, model, feat_container):
    key = (domain, class_name, args.layer)
    if key in cache:
        return cache[key]
    # dataset loading identical to toy_v4 logic
    from torchvision import transforms, datasets
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])
    root = resolve_pacs_root(args.data_root)
    dom_dir = locate_pacs_domain(root, domain)
    if class_name == 'all':
        ds = datasets.ImageFolder(dom_dir, transform=tfm)
    else:
        from toy_pacs_repr_drift import PACSClassSubset
        ds = PACSClassSubset(dom_dir, class_name, tfm, max_imgs=args.max_imgs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    features = extract_features(model, feat_container, dl, device=args.device,
                                spatial_pool=args.spatial_pool, flat_down=args.flat_down)
    cache[key] = features
    return features


def layer_scan(args):
    layers = ['conv1', 'layer1', 'layer2', 'layer3']
    out_csv = Path(args.out_prefix or 'layer_scan.csv')
    rows = [('layer', 'pi_distance')]
    # reuse model backbone for all layers (only rehook per layer)
    cache = {}
    for layer in layers:
        args.layer = layer
        model, feat_container = build_resnet18(hook_layer=layer, device=args.device)
        fa = compute_features(args.domain_a, args.class_name, args, cache, model, feat_container)
        fb = compute_features(args.domain_b, args.class_name, args, cache, model, feat_container)
        dist = pi_distance(fa, fb, args.rips_q, args.rips_dim, args.rips_max_pairs)
        print(f"{layer}: PI distance={dist:.4f}")
        rows.append((layer, f"{dist:.4f}"))
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    # plot
    xs, ys = zip(*[(r[0], float(r[1])) for r in rows[1:]])
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('PI L2 distance')
    plt.title(f"PI distance vs layer ({args.domain_a} vs {args.domain_b})")
    plt.tight_layout()
    plt.savefig(out_csv.with_suffix('.png'), dpi=200)
    print(f"Saved CSV -> {out_csv}, plot -> {out_csv.with_suffix('.png')}")


def domain_matrix(args):
    domains = PACS_DOMAINS
    n = len(domains)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    cache = {}
    model, feat_container = build_resnet18(hook_layer=args.layer, device=args.device)
    for i, da in enumerate(domains):
        for j, db in enumerate(domains):
            if j <= i:  # symmetric
                continue
            fa = compute_features(da, args.class_name, args, cache, model, feat_container)
            fb = compute_features(db, args.class_name, args, cache, model, feat_container)
            dist = pi_distance(fa, fb, args.rips_q, args.rips_dim, args.rips_max_pairs)
            dist_mat[i, j] = dist_mat[j, i] = dist
            print(f"{da} vs {db}: {dist:.2f}")
    out_csv = Path(args.out_prefix or f'domain_matrix_{args.layer}.csv')
    # save csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + domains)
        for i, d in enumerate(domains):
            w.writerow([d] + [f"{dist_mat[i,j]:.4f}" for j in range(n)])
    # heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(dist_mat, annot=True, xticklabels=domains, yticklabels=domains, cmap='Blues', fmt='.1f')
    plt.title(f"PI distance matrix ({args.layer})")
    plt.tight_layout()
    plt.savefig(out_csv.with_suffix('.png'), dpi=200)
    print(f"Saved CSV -> {out_csv}, heatmap -> {out_csv.with_suffix('.png')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Layer / Domain scan for FedTopo toy")
    sub = parser.add_subparsers(dest='cmd', required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('--data_root', type=str, required=True)
    base.add_argument('--class_name', type=str, default='dog')
    base.add_argument('--max_imgs', type=int, default=200)
    base.add_argument('--batch_size', type=int, default=32)
    base.add_argument('--spatial_pool', type=str, default='flatten', choices=['flatten', 'gap'])
    base.add_argument('--flat_down', type=int, default=8)
    base.add_argument('--rips_dim', type=int, default=32)
    base.add_argument('--rips_q', type=float, default=0.5)
    base.add_argument('--rips_max_pairs', type=int, default=50000)
    base.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    base.add_argument('--out_prefix', type=str, default=None)

    p_layer = sub.add_parser('layer_scan', parents=[base])
    p_layer.add_argument('--domain_a', type=str, required=True, choices=PACS_DOMAINS)
    p_layer.add_argument('--domain_b', type=str, required=True, choices=PACS_DOMAINS)

    p_matrix = sub.add_parser('domain_matrix', parents=[base])
    p_matrix.add_argument('--layer', type=str, default='layer1',
                          choices=['conv1', 'layer1', 'layer2', 'layer3'])

    args = parser.parse_args()

    if args.out_prefix is None:
        if args.cmd == 'layer_scan':
            args.out_prefix = f"layer_scan_{args.domain_a}_{args.domain_b}_{args.class_name}"
        else:
            args.out_prefix = f"domain_matrix_{args.layer}_{args.class_name}"

    if args.cmd == 'layer_scan':
        layer_scan(args)
    else:
        domain_matrix(args)
