#!/usr/bin/env python
# pixel_vs_topology_demo.py
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from gudhi import CubicalComplex
from skimage.metrics import structural_similarity    # MSE & more
sns.set_style('white'); os.makedirs('figs', exist_ok=True)

# ---------- 1. generate masks ----------
H=W=128
yy,xx=np.mgrid[:H,:W]
disk=lambda cx,cy: (yy-cy)**2+(xx-cx)**2 <= 40**2
img_A = disk(64,64).astype(int)
img_B = img_A.copy(); img_B[61:67,61:67]=0         # 3×3 hole
img_C = disk(74,64).astype(int)                    # shift right 10px

imgs=[img_A,img_B,img_C]; titles=['A  original','B  3×3 hole','C  shift 10px']

# ---------- 2. pixel-level losses ----------
def dice(a,b):    return 1 - (2*(a&b).sum()+1)/((a.sum()+b.sum())+1)
def iou(a,b):     return 1 - ((a&b).sum()+1)/((a|b).sum()+1)
def bce(a,b):     return -(a*np.log(np.clip(b,.01,1))+(1-a)*np.log(np.clip(1-b,.01,1))).mean()
def mse(a,b):     return ((a-b)**2).mean()

metrics = {'Dice':dice,'IoU':iou,'BCE':bce,'MSE':mse}
loss_AB={k:f(img_A,img_B) for k,f in metrics.items()}
loss_AC={k:f(img_A,img_C) for k,f in metrics.items()}

# ---------- 3. Betti numbers ----------
def betti(im):
    cc=CubicalComplex(dimensions=im.shape, top_dimensional_cells=im.ravel())
    cc.compute_persistence()
    return (len(cc.persistence_intervals_in_dimension(0)),
            len(cc.persistence_intervals_in_dimension(1)))

betA,betB,betC=[betti(m) for m in imgs]
delta_B = betB[1]-betA[1]   # Δβ1
delta_C = betC[1]-betA[1]

# ---------- 4. plot ----------
fig,ax=plt.subplots(1,3,figsize=(8,2.6))
for i,(im,t) in enumerate(zip(imgs,titles)):
    ax[i].imshow(im,cmap='gray')
    ax[i].set_axis_off()
    ax[i].set_title(f"{t}\nβ0={betti(im)[0]} β1={betti(im)[1]}",fontsize=8)
plt.tight_layout(); plt.savefig('figs/mask_triplet.png',dpi=300)

plt.figure(figsize=(6,3))
pairs=['A→B','A→C']; x=np.arange(len(metrics))
bar_w=.35
plt.bar(x-bar_w/2, [loss_AB[k] for k in metrics], width=bar_w, color='steelblue', label='pixel-loss  A→B')
plt.bar(x+bar_w/2, [loss_AC[k] for k in metrics], width=bar_w, color='darkorange', label='pixel-loss  A→C')
plt.plot(x,[delta_B]*len(x),'k--',label='Δβ₁  A→B'); plt.plot(x,[delta_C]*len(x),'g--',label='Δβ₁  A→C')
plt.xticks(x,metrics.keys()); plt.ylabel('loss / Betti'); plt.legend(fontsize=8)
plt.title('Pixel losses vs Topology difference')
plt.tight_layout(); plt.savefig('figs/topo_pixel_demo.png',dpi=300)


print("done → figs/mask_triplet.png & topo_pixel_demo.png  ( <1 s )")
