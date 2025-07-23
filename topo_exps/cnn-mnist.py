import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gudhi.representations import PersistenceImage
import gudhi

# ============ 网络结构 =============
class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============ 配置参数 =============
N_CLIENTS = 5
DIRICHLET_ALPHA = 0.1
N_CLASSES = 10
N_EPOCH = 2
BATCH_SIZE = 64
LAYER = 'conv2'
N_TEST_PER_CLASS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============ 数据集 Dirichlet 分割 =============
def split_dataset_dirichlet(dataset, alpha, n_clients, n_classes):
    labels = np.array(dataset.targets)
    idx_list = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = np.array([p * (len(idx_c) - len(idx_list[i])) for i, p in enumerate(proportions)])
        proportions = (proportions / proportions.sum()) * len(idx_c)
        proportions = proportions.astype(int)
        proportions[-1] = len(idx_c) - proportions[:-1].sum()
        start = 0
        for i in range(n_clients):
            idx_list[i] += idx_c[start:start+proportions[i]].tolist()
            start += proportions[i]
    return idx_list

# ============ 数据加载 =============
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
client_indices = split_dataset_dirichlet(dataset, DIRICHLET_ALPHA, N_CLIENTS, N_CLASSES)

testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform)
test_class_idxs = [[] for _ in range(N_CLASSES)]
for idx, (img, lbl) in enumerate(testset):
    if len(test_class_idxs[lbl]) < N_TEST_PER_CLASS:
        test_class_idxs[lbl].append(idx)
test_idxs = sum(test_class_idxs, [])
public_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(testset, test_idxs),
    batch_size=100, shuffle=False
)

# ============ 获取激活 =============
def get_activations(model, images, layer=LAYER):
    feats = []
    handle = dict(model.named_modules())[layer].register_forward_hook(
        lambda m, _, out: feats.append(out.detach().cpu())
    )
    _ = model(images.to(DEVICE))
    handle.remove()
    return feats[0]  # [B, C, H, W]

# ============ 客户端本地微调 ============
def local_finetune(global_model, dataset, indices, epochs=N_EPOCH):
    model = SimpleCNNMNIST(input_dim=16*4*4, hidden_dims=[120, 84], output_dim=10)
    model.load_state_dict(global_model.state_dict())
    model.to(DEVICE)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=BATCH_SIZE, shuffle=True
    )
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

# ============ 全局模型初始化 ============
global_model = SimpleCNNMNIST(input_dim=16*4*4, hidden_dims=[120, 84], output_dim=10).to(DEVICE)
global_model.apply(lambda m: hasattr(m, 'reset_parameters') and m.reset_parameters())

# ============ 客户端训练 ============
client_models = []
for i in range(N_CLIENTS):
    print(f"训练客户端 {i+1} / {N_CLIENTS}")
    model_i = local_finetune(global_model, dataset, client_indices[i])
    client_models.append(model_i)

# ============ 激活提取 ============
for x, y in public_loader:
    test_imgs = x
    break

acts_clients = []
for i, model in enumerate(client_models):
    with torch.no_grad():
        acts = get_activations(model, test_imgs, layer=LAYER)   # [B,C,H,W]
    acts_clients.append(acts)

# ============ 统计漂移 ============
mu_list, sigma_list = [], []
for acts in acts_clients:
    A = acts.numpy()   # [B,C,H,W]
    A = A.reshape(-1, A.shape[1])  # [B*H*W, C]
    mu = A.mean(axis=0)    # [C,]
    sigma = A.std(axis=0)  # [C,]
    mu_list.append(mu)
    sigma_list.append(sigma)

mu_arr = np.stack(mu_list)
sigma_arr = np.stack(sigma_list)

# ============ PI特征 ============
pi = PersistenceImage(
    bandwidth=0.2,
    weight=lambda bd: float(bd[1] - bd[0]),
    resolution=[8,8]
)
pi.fit([np.array([[0.0, 1.0]], dtype=np.float32)])

def channel_pi(arr):
    cc = gudhi.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
    bars = cc.persistence()
    diag = [p[1] for p in bars if p[0]==0 and p[1][1]>p[1][0]]
    if len(diag) == 0:
        return np.zeros(pi.resolution[0]*pi.resolution[1], dtype=np.float32)
    return pi.transform([np.array(diag, dtype=np.float32)]).ravel()

def pooled_pi_feats(acts, n_channels=0):
    if n_channels == 0: n_channels = acts.shape[1]
    pooled = nn.AdaptiveAvgPool2d((7,7))(torch.from_numpy(acts)) # [B,C,7,7]，更适合mnist
    pooled = pooled.numpy()
    pis = []
    for c in range(n_channels):
        pi_c = np.stack([channel_pi(pooled[b, c, :, :]) for b in range(pooled.shape[0])], axis=0)
        pis.append(pi_c.mean(axis=0))
    return np.stack(pis)

pi_vecs = []
for idx, acts in enumerate(acts_clients):
    print(f'计算PI：客户端{idx+1}')
    pi_feats = pooled_pi_feats(acts.numpy())
    print('PI均值/方差:', np.mean(pi_feats), np.var(pi_feats))
    pi_vecs.append(pi_feats)

def calc_pi_dist(pi_vecs):
    D = np.zeros((N_CLIENTS, N_CLIENTS))
    for i in range(N_CLIENTS):
        for j in range(N_CLIENTS):
            d = np.linalg.norm(pi_vecs[i]-pi_vecs[j])
            D[i, j] = d
    return D

D = calc_pi_dist(pi_vecs)

mean_var = np.var(mu_arr)
pi_var = np.var(D)
print("均值散布（方差）:", mean_var)
print("PI距离散布（方差）:", pi_var)
print("统计漂移/拓扑漂移比值:", mean_var/pi_var if pi_var > 0 else "N/A")

# ============ 可视化 ============
plt.figure(figsize=(6,3))
sns.heatmap(mu_arr, cmap='viridis', cbar=True)
plt.xlabel('Channel'); plt.ylabel('Client')
plt.title('Activation mean')
plt.tight_layout()
plt.savefig('mnist_mean_heatmap.png', dpi=300)
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(D, cmap='Blues')
plt.colorbar()
plt.title('PI Euclidean distance across clients')
plt.tight_layout()
plt.savefig('mnist_pi_distance.png', dpi=300)
plt.show()

def plot_compare_channel(channel=0):
    fig, axes = plt.subplots(2, N_CLIENTS, figsize=(2.2*N_CLIENTS,4))
    for i in range(N_CLIENTS):
        act = acts_clients[i][0, channel]
        act_tensor = torch.tensor(act).unsqueeze(0).unsqueeze(0)
        pool = nn.functional.adaptive_avg_pool2d(act_tensor, (7,7))[0,0].numpy()
        axes[0,i].imshow(pool, cmap='RdBu'); axes[0,i].set_title(f"Client{i+1} ch{channel}"); axes[0,i].axis('off')
        pi_img = pi_vecs[i][channel].reshape(8,8)
        axes[1,i].imshow(pi_img, cmap='gray'); axes[1,i].set_title("PI"); axes[1,i].axis('off')
    plt.suptitle(f"Channel {channel} visual & PI (sample 0)")
    plt.tight_layout()
    plt.savefig(f'mnist_channel{channel}_comp.png', dpi=300)
    plt.show()

plot_compare_channel(channel=0)

with open('mnist_fedtopo_metrics.txt', 'w') as f:
    f.write(f"均值方差: {mean_var:.4g}\n")
    f.write(f"PI 距离方差: {pi_var:.4g}\n")
    f.write(f"漂移方差比: {mean_var/pi_var if pi_var > 0 else 'N/A'}\n")

print("所有结果已保存（mnist_mean_heatmap.png, mnist_pi_distance.png, mnist_channel*_comp.png, mnist_fedtopo_metrics.txt）")
