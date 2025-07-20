import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from gudhi import CubicalComplex
from persim import PersistenceImager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =============== 数据集和特征 ====================
class PACSDataset(Dataset):
    def __init__(self, root, domain, classes, n_per_class=40, transform=None):
        self.samples, self.labels = [], []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            cdir = os.path.join(root, domain, c)
            img_files = os.listdir(cdir)
            if n_per_class:
                img_files = random.sample(img_files, min(n_per_class, len(img_files)))
            for f in img_files:
                self.samples.append(os.path.join(cdir, f))
                self.labels.append(self.class_to_idx[c])
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# =============== 设置 ====================
DATA_ROOT = '../../data/pacs/pacs_data/pacs_data'   # <-- 请替换为你的 PACS 根目录
DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
N_PER_CLASS = 30      # 每个域每类样本数
N_PAIR = 500          # 每种pair采样数
CLASSES = sorted(os.listdir(os.path.join(DATA_ROOT, DOMAINS[0])))
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============== 预处理 & 模型 ====================
transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()
])

resnet = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
LAYER_NAMES = ['conv1', 'layer1', 'layer2', 'layer3']

def get_layer_outputs(model, layers, x):
    hooks, outputs = [], {}
    for lname in layers:
        def _hook(mod, inp, out, name=lname):
            outputs[name] = out.detach().cpu()
        lmod = dict([*model.named_modules()])[lname]
        hooks.append(lmod.register_forward_hook(_hook))
    with torch.no_grad():
        _ = model(x.cuda())
    for h in hooks: h.remove()
    return {k: v for k,v in outputs.items()}

def extract_PI(feat, K=8, dim=1, size=32):
    if feat.shape[0] > K:
        feat = feat[:K]
    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (size, size))
    feat = feat.cpu().numpy()
    pi = PersistenceImager(pixel_size=0.05)
    vecs = []
    for ch in feat:
        cc = CubicalComplex(dimensions=ch.shape, top_dimensional_cells=ch.ravel())
        cc.compute_persistence()  # 这句是必须的！
        bar = cc.persistence_intervals_in_dimension(dim)
        v = pi.transform(bar)
        vecs.append(v.ravel())
    return np.stack(vecs, 0).mean(0)


# =============== 主流程：提取 PI 特征 ================
print("Step 1: Collecting PI features ...")
domain_class_feat = {layer: {(d, c): [] for d in DOMAINS for c in range(len(CLASSES))}
                     for layer in LAYER_NAMES}
domain_labels = {layer: [] for layer in LAYER_NAMES}
class_labels = {layer: [] for layer in LAYER_NAMES}
all_domains = {layer: [] for layer in LAYER_NAMES}
all_classes = {layer: [] for layer in LAYER_NAMES}
all_feats = {layer: [] for layer in LAYER_NAMES}

for d in DOMAINS:
    ds = PACSDataset(DATA_ROOT, d, CLASSES, n_per_class=N_PER_CLASS, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    for xb, yb in tqdm(loader, desc=f"Domain {d}"):
        feats = get_layer_outputs(resnet, LAYER_NAMES, xb)
        for li, lname in enumerate(LAYER_NAMES):
            for i in range(len(yb)):
                pi_feat = extract_PI(feats[lname][i], K=8, dim=1, size=32)
                c = int(yb[i])
                domain_class_feat[lname][(d,c)].append(pi_feat)
                domain_labels[lname].append(d)
                class_labels[lname].append(c)
                all_domains[lname].append(d)
                all_classes[lname].append(c)
                all_feats[lname].append(pi_feat)

# =============== Step 2: 采样 PI 距离 pairs ===============
from scipy.spatial.distance import euclidean

def random_pairs(domain_class_feat, N=500):
    results = {'same_class_same_domain': [],
               'same_class_diff_domain': [],
               'diff_class_same_domain': [],
               'diff_class_diff_domain': []}
    domains = list({d for d, _ in domain_class_feat})
    classes = list({c for _, c in domain_class_feat})
    for _ in range(N):
        # same_class_same_domain
        d, c = random.choice(list(domain_class_feat.keys()))
        vecs = domain_class_feat[(d, c)]
        if len(vecs) < 2: continue
        i, j = random.sample(range(len(vecs)), 2)
        results['same_class_same_domain'].append(euclidean(vecs[i], vecs[j]))
        # same_class_diff_domain
        d2 = random.choice([dd for dd in domains if dd != d])
        if (d2, c) in domain_class_feat and len(domain_class_feat[(d2, c)]) > 0:
            i1 = random.choice(range(len(vecs)))
            i2 = random.choice(range(len(domain_class_feat[(d2, c)])))
            results['same_class_diff_domain'].append(
                euclidean(vecs[i1], domain_class_feat[(d2, c)][i2]))
        # diff_class_same_domain
        c2 = random.choice([cc for cc in classes if cc != c])
        if (d, c2) in domain_class_feat and len(domain_class_feat[(d, c2)]) > 0:
            i1 = random.choice(range(len(vecs)))
            i2 = random.choice(range(len(domain_class_feat[(d, c2)])))
            results['diff_class_same_domain'].append(
                euclidean(vecs[i1], domain_class_feat[(d, c2)][i2]))
        # diff_class_diff_domain
        d2 = random.choice([dd for dd in domains if dd != d])
        c2 = random.choice([cc for cc in classes if cc != c])
        if (d2, c2) in domain_class_feat and len(domain_class_feat[(d2, c2)]) > 0:
            i1 = random.choice(range(len(vecs)))
            i2 = random.choice(range(len(domain_class_feat[(d2, c2)])))
            results['diff_class_diff_domain'].append(
                euclidean(vecs[i1], domain_class_feat[(d2, c2)][i2]))
    return results

layer_results = {}
for layer in LAYER_NAMES:
    print(f"Sampling pairs for layer {layer} ...")
    layer_results[layer] = random_pairs(domain_class_feat[layer], N=N_PAIR)

# =============== Step 3: ANOVA 统计 + 可视化 ===============
records = []
for layer, result in layer_results.items():
    for key, vals in result.items():
        for v in vals:
            records.append({'layer': layer, 'pair': key, 'distance': v})
df = pd.DataFrame(records)
sns.boxplot(data=df, x='pair', y='distance', hue='layer')
plt.title('PI distances by pair type & layer')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 方差分析
model = ols('distance ~ C(layer) * C(pair)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# =============== Step 4: t-SNE / UMAP ===============
for layer in LAYER_NAMES:
    print(f"t-SNE/UMAP for layer {layer} ...")
    X = np.array(all_feats[layer])
    dlabels = np.array(all_domains[layer])
    clabels = np.array(all_classes[layer])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2)
    embedding = reducer.fit_transform(X)
    plt.figure(figsize=(6,5))
    for d in np.unique(dlabels):
        idx = (dlabels == d)
        plt.scatter(embedding[idx,0], embedding[idx,1], label=d, alpha=0.5, s=8)
    plt.legend()
    plt.title(f"UMAP by domain - {layer}")
    plt.show()

    plt.figure(figsize=(6,5))
    for c in np.unique(clabels):
        idx = (clabels == c)
        plt.scatter(embedding[idx,0], embedding[idx,1], label=str(c), alpha=0.5, s=8)
    plt.legend()
    plt.title(f"UMAP by class - {layer}")
    plt.show()
