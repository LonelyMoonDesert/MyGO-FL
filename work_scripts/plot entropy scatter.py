import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# 更新后的 fedafg_data 数据
fedafg_data = {
    'Entropy': [
        6.631081, 6.636479, 6.683227, 6.701694, 6.716955, 6.732974, 6.739673, 6.750707, 6.759898, 6.766018
    ],
    'Train Accuracy': [
        0.5502, 0.5544, 0.6673, 0.7118, 0.7269, 0.7421, 0.7587, 0.7698, 0.7826, 0.7877
    ],
    'Test Accuracy': [
        0.4267, 0.4289, 0.5823, 0.6037, 0.6346, 0.6154, 0.6614, 0.6557, 0.6498, 0.6734
    ]
}

fedavg_data = {
    'Entropy': [
        6.614684, 6.604956, 6.594398, 6.595246, 6.597914, 6.600152, 6.608097, 6.609503, 6.608964
    ],
    'Train Accuracy': [
        0.4347, 0.6132, 0.5468, 0.6257, 0.6394, 0.6952, 0.664, 0.7167, 0.7371
    ],
    'Test Accuracy': [
        0.4242, 0.5184, 0.4622, 0.5478, 0.531, 0.593, 0.5226, 0.6476, 0.6256
    ]
}

# 创建 DataFrame
fedafg_df = pd.DataFrame(fedafg_data)
fedavg_df = pd.DataFrame(fedavg_data)

# 创建2行1列的子图
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制FedAFG的训练和测试准确度（第一个子图）
axs[0].scatter(fedafg_df['Entropy'], fedafg_df['Train Accuracy'], label='FedAFG Train Accuracy', color='blue', marker='o', s=100)
axs[0].scatter(fedafg_df['Entropy'], fedafg_df['Test Accuracy'], label='FedAFG Test Accuracy', color='blue', marker='x', s=100)
axs[0].set_title('FedAFG - Entropy vs Accuracy', fontsize=24)
axs[0].set_xlabel('Entropy Value', fontsize=22)
axs[0].set_ylabel('Accuracy', fontsize=22)
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[0].legend(fontsize=20)
axs[0].grid(True)

# 绘制FedAvg的训练和测试准确度（第二个子图）
axs[1].scatter(fedavg_df['Entropy'], fedavg_df['Train Accuracy'], label='FedAvg Train Accuracy', color='green', marker='o', s=100)
axs[1].scatter(fedavg_df['Entropy'], fedavg_df['Test Accuracy'], label='FedAvg Test Accuracy', color='green', marker='x', s=100)
axs[1].set_title('FedAvg - Entropy vs Accuracy', fontsize=24)
axs[1].set_xlabel('Entropy Value', fontsize=22)
axs[1].set_ylabel('Accuracy', fontsize=22)
axs[1].tick_params(axis='x', which='major', labelsize=18)
axs[1].tick_params(axis='y', which='major', labelsize=20)

# 格式化FedAvg的横轴刻度，使其显示两位小数
axs[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

axs[1].legend(fontsize=20)
axs[1].grid(True)

# 调整布局以确保图形不会被裁剪
plt.tight_layout()

# 显示图形
plt.show()
