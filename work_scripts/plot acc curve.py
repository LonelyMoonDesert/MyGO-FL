import matplotlib.pyplot as plt

# 数据

# 数据
rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FedAFG = [0.3436, 0.4477, 0.5054, 0.5604, 0.6059, 0.6404, 0.6571, 0.6714, 0.6808, 0.6975]
FedAvg = [0.2224, 0.3252, 0.3936, 0.4667, 0.4939, 0.5191, 0.5716, 0.5835, 0.6021, 0.6347]
FedNova = [0.2011, 0.3624, 0.4332, 0.4605, 0.5064, 0.5321, 0.5374, 0.5814, 0.6046, 0.6277]
FedProx = [0.2359, 0.3479, 0.4198, 0.4557, 0.4913, 0.5035, 0.5185, 0.5561, 0.5707, 0.5956]
SCAFFOLD = [0.2286, 0.3230, 0.4181, 0.4361, 0.4973, 0.5350, 0.5645, 0.5798, 0.6150, 0.6235]
# 创建图形
plt.figure(figsize=(10, 6))

# 设置更和谐的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 绘制每条曲线
plt.plot(rounds, FedAFG, label='FedAFG', marker='o', color=colors[0], linewidth=2)
plt.plot(rounds, FedAvg, label='FedAvg', marker='s', color=colors[1], linewidth=2)
plt.plot(rounds, FedNova, label='FedNova', marker='^', color=colors[2], linewidth=2)
plt.plot(rounds, FedProx, label='FedProx', marker='D', color=colors[3], linewidth=2)
plt.plot(rounds, SCAFFOLD, label='SCAFFOLD', marker='x', color=colors[4], linewidth=2)
# 设置字体大小（更大且不加粗）
plt.rcParams.update({'font.size': 24})  # 全局字体大小
plt.title('Global Model Test Accuracy per Round', fontsize=24)  # 标题字体大小
plt.xlabel('Round', fontsize=24)  # X轴标签字体大小
plt.ylabel('Test Accuracy', fontsize=24)  # Y轴标签字体大小

# 调整坐标轴刻度标签的字体大小
plt.xticks(fontsize=22)  # X轴刻度标签字体大小
plt.yticks(fontsize=22)  # Y轴刻度标签字体大小

# 添加图例
plt.legend(fontsize=24, frameon=True, shadow=True, loc='best')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()