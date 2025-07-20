"""
topo_layer_analysis_v2.py
输出:
  figs/line_probe.png   折线 + 交叉点
  figs/heat_domain_L*.png / heat_class_L*.png
依赖: torch torchvision scikit-learn matplotlib seaborn gudhi persim tqdm
"""
import os, random, numpy as np, torch, torchvision, tqdm, itertools, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from gudhi import CubicalComplex
from persim import PersistenceImager
import matplotlib.pyplot as plt
from pathlib import Path
# ---------- CONFIG ----------
DATA = '../../data/pacs/pacs_data/pacs_data'
DOMAINS = ['art_painting','cartoon','photo','sketch']
CLASSES = sorted(os.listdir(os.path.join(DATA, DOMAINS[0])))
LAYERS  = ['conv1','layer1','layer2','layer3']
N_PER   = 100                # 样本更充分
K_CH    = 8
SIZE    = 32
DIMs    = [0,1]             # β0 & β1
SEED    = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs('figs', exist_ok=True)

# ---------- Dataset ----------
from torchvision import transforms
T = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
class PACS(torch.utils.data.Dataset):
    def __init__(self, root, domain):
        self.paths, self.cls = [], []
        for c in CLASSES:
            fs = os.listdir(os.path.join(root, domain, c))[:N_PER]
            self.paths += [os.path.join(root, domain, c,f) for f in fs]
            self.cls   += [CLASSES.index(c)]*len(fs)
        self.dom = [DOMAINS.index(domain)]*len(self.paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        from PIL import Image
        return T(Image.open(self.paths[i]).convert('RGB')), self.cls[i], self.dom[i]

ds = torch.utils.data.ConcatDataset([PACS(DATA,d) for d in DOMAINS])
loader = torch.utils.data.DataLoader(ds,batch_size=16,shuffle=False)

# ---------- Model & hooks ----------
net = torchvision.models.resnet18(weights='DEFAULT').eval().cuda()
out={}
def hook(n):
    return lambda m,i,o: out.setdefault(n, []).append(o.detach().cpu())
for n in LAYERS: dict([*net.named_modules()])[n].register_forward_hook(hook(n))

# ---------- PI util ----------
pim = PersistenceImager(pixel_size=0.05)
def pi_vec(feat):
    feat = torch.nn.functional.adaptive_avg_pool2d(feat,(SIZE,SIZE))[:K_CH].cpu().numpy()
    vec=[]
    for dim in DIMs:
        for ch in feat:
            cc=CubicalComplex(dimensions=ch.shape, top_dimensional_cells=ch.ravel())
            cc.compute_persistence()
            vec.append(pim.transform(cc.persistence_intervals_in_dimension(dim)).ravel())
    return np.mean(vec,0)

# ---------- Gather ----------
Z, Y_dom, Y_cls = {l:[] for l in LAYERS}, [], []
for xb, cls, dom in tqdm.tqdm(loader):
    out.clear(); _=net(xb.cuda())
    for l in LAYERS:
        for feat in out[l]:
            Z[l].append(pi_vec(feat))
    Y_dom += dom.tolist(); Y_cls += cls.tolist()

# ---------- Metrics ----------
probe_D, probe_C, sil_D, sil_C, pi_ratio = [],[],[],[],[]
for l in LAYERS:
    Zl=np.stack(Z[l])
    # linear probe
    clf=LogisticRegression(max_iter=4000,solver='lbfgs',multi_class='multinomial')
    probe_D.append(accuracy_score(Y_dom, clf.fit(Zl,Y_dom).predict(Zl)))
    probe_C.append(accuracy_score(Y_cls, clf.fit(Zl,Y_cls).predict(Zl)))
    # silhouette
    sil_D.append(silhouette_score(Zl, Y_dom))
    sil_C.append(silhouette_score(Zl, Y_cls))
    # π‑ratio
    def avg_pair(X, lab):
        s=0;n=0
        for a,b in itertools.combinations(range(len(X)),2):
            if lab[a]!=lab[b]:
                s+=np.linalg.norm(X[a]-X[b]); n+=1
        return s/n
    pi_ratio.append(avg_pair(Zl,Y_dom)/avg_pair(Zl,Y_cls))

# ---------- 折线图 ----------
x=np.arange(len(LAYERS))
plt.figure(figsize=(6,4))
plt.plot(x, probe_D, 'b-o', label='Domain‑Acc')
plt.plot(x, probe_C, 'orange',marker='s',label='Class‑Acc')
inter=np.argwhere(np.diff(np.sign(np.array(probe_D)-np.array(probe_C)))).flatten()
if len(inter):
    xi=inter[0]; plt.scatter(xi, probe_D[xi], c='r', s=80, zorder=3)
    plt.text(xi+0.1, probe_D[xi], 'cross‑over', color='r')
plt.xticks(x, LAYERS); plt.ylabel('Linear‑Probe Acc'); plt.legend(); plt.tight_layout()
plt.savefig('figs/line_probe.png'); plt.close()

# ---------- 热图 ----------
import seaborn as sns
for l, Zl in Z.items():
    Zl = np.stack(Zl)
    # domain matrix 4×4
    m_dom=np.zeros((len(DOMAINS),len(DOMAINS)))
    for i,d1 in enumerate(DOMAINS):
        for j,d2 in enumerate(DOMAINS):
            idx1=[k for k,y in enumerate(Y_dom) if y==i]
            idx2=[k for k,y in enumerate(Y_dom) if y==j]
            m_dom[i,j]=np.linalg.norm(Zl[idx1].mean(0)-Zl[idx2].mean(0))
    sns.heatmap(m_dom,annot=True,fmt=".2f",xticklabels=DOMAINS,yticklabels=DOMAINS,cmap='Blues')
    plt.title(f'Domain distance {l}'); plt.tight_layout(); plt.savefig(f'figs/heat_domain_{l}.png'); plt.close()
    # class matrix 7×7
    m_cls=np.zeros((len(CLASSES),len(CLASSES)))
    for i,c1 in enumerate(CLASSES):
        for j,c2 in enumerate(CLASSES):
            idx1=[k for k,y in enumerate(Y_cls) if y==i]
            idx2=[k for k,y in enumerate(Y_cls) if y==j]
            m_cls[i,j]=np.linalg.norm(Zl[idx1].mean(0)-Zl[idx2].mean(0))
    sns.heatmap(m_cls,annot=False,cbar=False,cmap='Reds')
    plt.title(f'Class distance {l}'); plt.tight_layout(); plt.savefig(f'figs/heat_class_{l}.png'); plt.close()

# ---------- 打印指标 ----------
print("\nLayer | DomainAcc  ClassAcc | Sil(D)  Sil(C) | π‑ratio")
for i,l in enumerate(LAYERS):
    print(f"{l:<6}  {probe_D[i]*100:6.1f}  {probe_C[i]*100:6.1f} | "
          f"{sil_D[i]:.3f}  {sil_C[i]:.3f} | {pi_ratio[i]:.2f}")
