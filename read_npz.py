import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import os
import itertools

# ====== 配置参数 ======
PI_NPY_FILE = 'logs/topo_PI_records.npz'  # 你的 npz 路径
N_CLIENTS = 5                           # 客户端数量，需与训练一致
PI_SHAPE = (20, 20)                     # Persistence Image 分辨率，如 10x10
SAVE_DIR = './topo_vis_results'         # 图片保存目录



datas = np.load(PI_NPY_FILE, allow_pickle=True)

for key, arr in datas.items():
  print(key, ": ", arr)