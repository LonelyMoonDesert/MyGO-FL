#!/usr/bin/env python
# pacs_drift_vis.py  (2025‑07)
# --------------------------------------------------------
# Visualise representation drift across PACS domains (=clients)
# Output: figs/pacs_umap_all.svg , figs/pacs_horse_nn.svg
# --------------------------------------------------------
import os, random, numpy as np, tqdm, umap, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from persim import PersistenceImager
from gudhi import CubicalComplex
sns.set_style('white'); sns.set_palette('tab10')
Path('figs').mkdir(exist_ok=True)

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PACSSlice(Dataset):
    def __init__(self, root, domain, classes, n_per):
        self.imgs, self.cls = [], []
        for c in classes:
            p = os.path.join(root, domain, c)
            files = sorted(os.listdir(p))[:n_per]
            self.imgs += [os.path.join(p,f) for f in files]
            self.cls  += [classes.index(c)]*len(files)
        self.dom = [DOMAINS.index(domain)]*len(self.imgs)
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        from PIL import Image
        return tf(Image.open(self.imgs[i]).convert('RGB')), self.cls[i], self.dom[i]
# ---------- 1. PACS mini‑subset loader ----------
if __name__ == '__main__':
    ROOT = '../../data/pacs/pacs_data/pacs_data'                       # change if needed
    DOMAINS = ['art_painting','cartoon','photo','sketch']
    CLASSES = ['dog','guitar','horse','person']   # 4/7 classes for clarity
    N_PER   = 400                         # images / domain / class

    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    subsets = [PACSSlice(ROOT,d,CLASSES,N_PER) for d in DOMAINS]
    loader  = DataLoader(torch.utils.data.ConcatDataset(subsets),
                         batch_size=32, shuffle=False, num_workers=4)

    # ---------- 2. ResNet‑18 layer1 feature ----------
    net = torchvision.models.resnet18(weights='DEFAULT').to(DEVICE).eval()
    feat_hook={}
    def _h(_,__,out): feat_hook['x']=out.detach().cpu()
    net.layer1.register_forward_hook(_h)

    pim = PersistenceImager(pixel_size=0.05)
    def feat_to_pi(feat, k=8, size=16):
        feat = torch.nn.functional.adaptive_avg_pool2d(feat,(size,size))[:k]
        vec=[]
        for ch in feat:
            cc=CubicalComplex(dimensions=ch.shape, top_dimensional_cells=ch.numpy().ravel())
            cc.compute_persistence()
            vec.append(pim.transform(cc.persistence_intervals_in_dimension(1)).ravel())
        return np.mean(vec,0)

    vecs, y_cls, y_dom = [], [], []
    for x, c, d in tqdm.tqdm(loader):
        _ = net(x.to(DEVICE))
        for i in range(x.size(0)):
            vecs.append(feat_to_pi(feat_hook['x'][i]))
            y_cls.append(int(c[i])); y_dom.append(int(d[i]))
    vecs = np.stack(vecs)

    # ---------- 3. UMAP ----------
    emb = umap.UMAP(n_neighbors=80, min_dist=0.05, random_state=SEED).fit_transform(vecs)

    # ---------- 4‑A. 全类散点 ----------
    markers=['o','s','^','D']                 # 4 domains
    plt.figure(figsize=(6,5))
    for did,m in enumerate(markers):
        idx = [i for i,d in enumerate(y_dom) if d==did]
        plt.scatter(emb[idx,0],emb[idx,1],marker=m,s=14,
                    c=[sns.color_palette()[y_cls[i]] for i in idx],alpha=.8,
                    label=DOMAINS[did])
    plt.xticks([]);plt.yticks([]);plt.axis('equal')
    legend=plt.legend(frameon=False,fontsize=8,ncol=2,bbox_to_anchor=(0.5,-0.06),loc='upper center')
    plt.title("PACS•layer1 PI(β₁)•UMAP",fontsize=14)
    plt.tight_layout(); plt.savefig('figs/pacs_umap_all.svg',dpi=400,bbox_extra_artists=(legend,))

    # ---------- 4‑B. 单类 (horse) k‑NN 连通图 ----------
    cls_id = CLASSES.index('horse')
    horse = [i for i,c in enumerate(y_cls) if c==cls_id]
    colors = ['royalblue','darkorange','seagreen','crimson']
    plt.figure(figsize=(4,4))
    from sklearn.neighbors import NearestNeighbors
    for did,m,col in zip(range(4),markers,colors):
        idx = [i for i in horse if y_dom[i]==did]
        pts = emb[idx]
        plt.scatter(pts[:,0],pts[:,1],marker=m,s=20,c=col,label=DOMAINS[did],alpha=.9)
        nbrs=NearestNeighbors(n_neighbors=4).fit(pts)
        for p,ids in enumerate(nbrs.kneighbors(return_distance=False)):
            for q in ids[1:]:
                plt.plot(pts[[p,q],0],pts[[p,q],1],c=col,alpha=.4,lw=.6)
    plt.xticks([]);plt.yticks([]);plt.axis('equal')
    plt.title("horse • client‑wise disconnected manifolds")
    plt.legend(frameon=False,fontsize=8)
    plt.tight_layout(); plt.savefig('figs/pacs_horse_nn.svg',dpi=400)

    print("Figures saved to ./figs/  (UMAP & horse‑kNN)")
