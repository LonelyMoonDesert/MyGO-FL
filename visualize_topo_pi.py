import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import os
import itertools

# ====== 配置参数 ======
PI_NPY_FILE = 'logs/resnet18-cifar10/label-beta0.1/topo-vis/topo_PI_records.npz'  # 你的 npz 路径
N_CLIENTS = 5                           # 客户端数量，需与训练一致
PI_SHAPE = (20, 20)                     # Persistence Image 分辨率，如 10x10
SAVE_DIR = './topo_vis_results'         # 图片保存目录

os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 工具函数 ======
def plot_pi_heatmap(pi_vec, title, save_path=None):
    """
    pi_vec: [batch, pi_dim] or [pi_dim] (mean后)，自动reshape为PI_SHAPE
    """
    arr = pi_vec.mean(axis=0) if pi_vec.ndim == 2 else pi_vec
    arr2d = arr.reshape(PI_SHAPE)
    plt.figure(figsize=(4,4))
    sns.heatmap(arr2d, cmap='viridis')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_similarity_curve(sim_dict, save_path=None):
    plt.figure(figsize=(7,5))
    for cid, sims in sim_dict.items():
        plt.plot(list(sims.keys()), list(sims.values()), marker='o', label=f'Client {cid}')
    plt.xlabel('Round')
    plt.ylabel('Cosine Similarity to Global')
    plt.title('PI Cosine Similarity: Client vs Global')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ====== 主流程 ======
# 1. 读取npz
pi_data = np.load(PI_NPY_FILE, allow_pickle=True)
key_rounds = sorted([int(k.replace('round','')) for k in pi_data.keys()])

# 2. 可视化每轮所有客户端和全局 PI 热图
for r in key_rounds:
    pi_dict = pi_data[f'round{r}'].item()
    for cid in range(N_CLIENTS):
        if cid in pi_dict:
            plot_pi_heatmap(
                pi_dict[cid],
                title=f'Round {r} Client {cid}',
                save_path=os.path.join(SAVE_DIR, f'heatmap_round{r}_client{cid}.png')
            )
    if 'global' in pi_dict:
        plot_pi_heatmap(
            pi_dict['global'],
            title=f'Round {r} Global',
            save_path=os.path.join(SAVE_DIR, f'heatmap_round{r}_global.png')
        )
print(f"所有 PI 热力图已保存至 {SAVE_DIR}")

# 3. 客户端PI与全局PI的余弦相似度趋势曲线
sim_dict = {cid: {} for cid in range(N_CLIENTS)}
for r in key_rounds:
    pi_dict = pi_data[f'round{r}'].item()
    if 'global' not in pi_dict: continue
    global_vec = pi_dict['global'].mean(axis=0)
    for cid in range(N_CLIENTS):
        if cid in pi_dict:
            cli_vec = pi_dict[cid].mean(axis=0)
            sim = 1 - cosine(cli_vec, global_vec)
            sim_dict[cid][r] = sim

plot_similarity_curve(sim_dict, save_path=os.path.join(SAVE_DIR, "similarity_trend.png"))
print(f"客户端-全局 PI 相似度趋势曲线已保存至 {SAVE_DIR}")

# 4. （可选）打印距离或其他统计
client_pairwise_l2 = {}
client_pairwise_cosine = {}

# 新增：日志保存路径
LOG_TXT = os.path.join(SAVE_DIR, "client_stats.txt")
with open(LOG_TXT, "w", encoding="utf-8") as f:

    for r in key_rounds:
        pi_dict = pi_data[f'round{r}'].item()
        if 'global' not in pi_dict: continue
        round_header = f"\n[Round {r}]"
        print(round_header)
        f.write(round_header + "\n")
        global_vec = pi_dict['global'].mean(axis=0)
        for cid in range(N_CLIENTS):
            if cid in pi_dict:
                cli_vec = pi_dict[cid].mean(axis=0)
                dist = np.linalg.norm(cli_vec - global_vec)
                sim = 1 - cosine(cli_vec, global_vec)
                line = f"Client {cid} - L2距离: {dist:.4f}, 余弦相似度: {sim:.4f}"
                print(line)
                f.write(line + "\n")

        # ======= 新增：统计client间均值L2与均值余弦相似度 =======
        client_vecs = [pi_dict[cid].mean(axis=0) for cid in range(N_CLIENTS) if cid in pi_dict]
        pairs = list(itertools.combinations(range(len(client_vecs)), 2))
        if not pairs:
            continue
        l2s = []
        cosines = []
        for i, j in pairs:
            l2 = np.linalg.norm(client_vecs[i] - client_vecs[j])
            l2s.append(l2)
            cosine_sim = 1 - cosine(client_vecs[i], client_vecs[j])
            cosines.append(cosine_sim)
        client_pairwise_l2[r] = np.mean(l2s)
        client_pairwise_cosine[r] = np.mean(cosines)

        # 同步写一行client间均值统计
        line = f"Mean L2 Distance (Clients): {client_pairwise_l2[r]:.4f}, Mean Cosine Similarity (Clients): {client_pairwise_cosine[r]:.4f}"
        print(line)
        f.write(line + "\n")

print(f"详细统计信息已保存至 {LOG_TXT}")

# ======= 合并画图：一张图两个y轴 =======

fig, ax1 = plt.subplots(figsize=(8, 5))

xs = list(client_pairwise_l2.keys())
l2_vals = list(client_pairwise_l2.values())
cos_vals = list(client_pairwise_cosine.values())

color_l2 = 'tab:blue'
color_cos = 'tab:orange'

# 左y轴：均值L2距离
ax1.set_xlabel('Round')
ax1.set_ylabel('Mean L2 Distance', color=color_l2)
ax1.plot(xs, l2_vals, marker='o', color=color_l2, label='Mean L2 Distance')
ax1.tick_params(axis='y', labelcolor=color_l2)
ax1.grid(alpha=0.3)

# 右y轴：均值余弦相似度
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Cosine Similarity', color=color_cos)
ax2.plot(xs, cos_vals, marker='s', color=color_cos, label='Mean Cosine Similarity')
ax2.tick_params(axis='y', labelcolor=color_cos)

plt.title('Client-Client Mean PI Distance & Cosine Similarity')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "client_pairwise_l2_and_cosine.png"))
plt.close()
print(f"客户端间PI均值L2距离和均值余弦相似度趋势合并图已保存至 {SAVE_DIR}")