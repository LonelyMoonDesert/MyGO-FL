#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedTopo introduction figures:
A) non‑IID client drift -> UMAP
B) small pixel error -> big Betti error
C) domain vs class linear‑probe curves

Author: (your name) 2025‑07
"""

# ----------------- common imports -----------------
import os, random, itertools, math, numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns; sns.set_palette('Set2')
from pathlib import Path
Path('figs').mkdir(exist_ok=True)

SEED = 2025
random.seed(SEED); np.random.seed(SEED)

# ----------------- Experiment A --------------------
def expA_umap_drift():
    """CIFAR‑10 Dirichlet split + style‑aug  → UMAP"""
    import torch, torchvision, umap
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset

    # 1. non‑IID split --------------------------------------------------------
    alpha = 0.1
    root = '../../data'
    cifar = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                         transform=transforms.ToTensor())
    label_idx = [[] for _ in range(10)]
    for idx,(x,y) in enumerate(cifar):
        label_idx[y].append(idx)

    idx_a, idx_b = [], []
    for cls in range(10):
        idxs = np.array(label_idx[cls])
        proportion = np.random.dirichlet([alpha, alpha])
        n = len(idxs); n_a = int(proportion[0]*n)
        np.random.shuffle(idxs)
        idx_a.extend(idxs[:n_a]); idx_b.extend(idxs[n_a:])

    # 2. dataloader & cartoon style for client B ------------------------------
    make_gray = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dl_a = DataLoader(Subset(cifar, idx_a), batch_size=128, shuffle=False)
    dl_b = DataLoader(Subset(cifar, idx_b), batch_size=128, shuffle=False)

    # 3. feature extractor (ResNet‑18 layer1) ---------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet18(weights='DEFAULT').to(device).eval()
    feat_hook = {}
    def _hook(_,__ ,out): feat_hook['x']=out.detach().cpu()
    net.layer1.register_forward_hook(_hook)

    def get_pi(feat, k=8, size=16):
        from persim import PersistenceImager
        from gudhi import CubicalComplex
        pim = PersistenceImager(pixel_size=0.05)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat,(size,size))[:k]
        vec = []
        for ch in feat:
            cc = CubicalComplex(dimensions=ch.shape,
                                top_dimensional_cells=ch.numpy().ravel())
            cc.compute_persistence(); bars = cc.persistence_intervals_in_dimension(1)
            vec.append(pim.transform(bars).ravel())
        return np.mean(vec,0)

    Z, labs, doms = [], [], []
    for dom,loader,style in [('A',dl_a,False),('B',dl_b,True)]:
        for x,y in loader:
            if style: x = make_gray(x)            # cartoon‑like grayscale
            _ = net(x.to(device))
            for i in range(x.size(0)):
                Z.append(get_pi(feat_hook['x'][i]))
                labs.append(int(y[i])); doms.append(dom)

    # 4. UMAP + plotting ------------------------------------------------------
    Z = np.stack(Z)
    emb = umap.UMAP(n_neighbors=40, min_dist=0.1).fit_transform(Z)
    plt.figure(figsize=(4.5,4))
    for dom,marker in zip(['A','B'],['x','o']):
        idx = [i for i,d in enumerate(doms) if d==dom]
        plt.scatter(emb[idx,0], emb[idx,1], s=10, marker=marker, alpha=.8,
                    label=f'client {dom}')
    plt.xticks([]); plt.yticks([]); plt.axis('equal')
    plt.title("Same label – diff. clients → drift")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig('figs/umap_drift.svg')

# ----------------- Experiment B --------------------
def expB_pixel_vs_topology():
    """Blood‑vessel mask: tiny erasures -> Betti explode"""
    from skimage import io, morphology

    mask = io.imread('./data/drive_sample_mask.png')>0   # 512×512 bin mask
    H,W = mask.shape
    yy,xx = np.mgrid[:H,:W]

    # three erasure radii
    radii = [0, 3, 6, 9]
    imgs, bettis = [], []
    from gudhi import CubicalComplex
    pim = []
    for r in radii:
        m = mask.copy()
        if r>0:
            cy,cx = np.random.randint(100,400,2)
            m[(yy-cy)**2+(xx-cx)**2<=r**2]=0
        imgs.append(m)
        cc = CubicalComplex(dimensions=m.shape, top_dimensional_cells=m.ravel())
        cc.compute_persistence()
        bettis.append((len(cc.persistence_intervals_in_dimension(0)),
                       len(cc.persistence_intervals_in_dimension(1))))

    # plot grid ---------------------------------------------------------------
    fig,ax = plt.subplots(1,4,figsize=(10,2.8))
    titles=['orig']+['erase r='+str(r) for r in radii[1:]]
    for i,(im,t) in enumerate(zip(imgs,titles)):
        ax[i].imshow(im,cmap='gray')
        ax[i].set_title(f"{t}\nβ0={bettis[i][0]} β1={bettis[i][1]}", fontsize=8)
        ax[i].set_axis_off()
    plt.tight_layout(); plt.savefig('figs/vessel_betti_grid.png', dpi=300)

# ----------------- Experiment C --------------------
def expC_line_probe():
    """Office‑Home 4域 65类 → domain/class probe vs layer"""
    import torch, torchvision
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    # ——简化：随便采 Office‑Home subset (需下载)——
    data_root='./office_home_sub'
    domains=['Art','Clipart','Real_World','Product']
    trs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])
    sampler=[]
    for d in domains:
        ds=ImageFolder(os.path.join(data_root,d),transform=trs)
        idx = random.sample(range(len(ds)), 500)  # 每域 500
        sampler.extend([(ds[i][0], ds[i][1], domains.index(d)) for i in idx])
    X, y_cls, y_dom = zip(*sampler)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet50(weights='DEFAULT').to(device).eval()
    layers = ['conv1','layer1','layer2','layer3','layer4','avgpool']
    hooks, outs={}, {}
    for l in layers:
        def _h(name):
            return lambda _,__,o: outs.setdefault(name,[]).append(o.detach().cpu())
        dict([*net.named_modules()])[l].register_forward_hook(_h(l))

    from persim import PersistenceImager; from gudhi import CubicalComplex
    pim=PersistenceImager(pixel_size=.05)

    Z={l:[] for l in layers}
    for x in tqdm.tqdm(X):
        outs.clear(); _=net(x.unsqueeze(0).to(device))
        for l in layers:
            feat=outs[l][0]
            if l=='avgpool': vec=feat.flatten().cpu().numpy(); Z[l].append(vec); continue
            feat=torch.nn.functional.adaptive_avg_pool2d(feat,(16,16))[:8]
            v=[]
            for ch in feat:
                cc=CubicalComplex(dimensions=ch.shape,top_dimensional_cells=ch.numpy().ravel())
                cc.compute_persistence()
                v.append(pim.transform(cc.persistence_intervals_in_dimension(1)).ravel())
            Z[l].append(np.mean(v,0))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    acc_dom,acc_cls=[],[]
    for l in layers:
        Zl=np.stack(Z[l])
        clf=LogisticRegression(max_iter=4000,solver='lbfgs',multi_class='multinomial')
        acc_dom.append(accuracy_score(y_dom, clf.fit(Zl,y_dom).predict(Zl)))
        acc_cls.append(accuracy_score(y_cls, clf.fit(Zl,y_cls).predict(Zl)))

    # ——折线图——
    x=np.arange(len(layers))
    plt.figure(figsize=(5,3.2))
    plt.plot(x,acc_dom,'b-o',label='Domain probe')
    plt.plot(x,acc_cls,'orange',marker='s',label='Class probe')
    xi=np.argwhere(np.diff(np.sign(np.array(acc_dom)-np.array(acc_cls)))).flatten()
    if len(xi): plt.scatter(xi[0],acc_dom[xi[0]],c='r',s=60)
    plt.xticks(x,layers,rotation=30); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.savefig('figs/domain_class_line.svg')

# ----------------- main ----------------------------
if __name__ == '__main__':
    expA_umap_drift()
    expB_pixel_vs_topology()
    # expC_line_probe()   # 需要 Office‑Home，数据较大；按需取消注释
    print("All figures saved to ./figs/")
