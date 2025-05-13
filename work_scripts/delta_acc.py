import pandas as pd
import matplotlib.pyplot as plt

# 假设你的数据已经加载到一个DataFrame中，命名为df
df = pd.read_csv('data.csv')

# 提取相关列
df_normal = df[['network', 'round', 'normal_pre_training_train_acc', 'normal_training_train_acc']]
df_adversarial = df[['network', 'round', 'adversarial_pre_training_train_acc', 'adversarial_training_train_acc']]

# 计算精度变化
df_normal['normal_train_improvement'] = df_normal['normal_training_train_acc'] - df_normal['normal_pre_training_train_acc']
df_adversarial['adversarial_train_improvement'] = df_adversarial['adversarial_training_train_acc'] - df_adversarial['adversarial_pre_training_train_acc']

# 计算每个 round 内的平均提升（以 network 为分组）
normal_improvement_per_round = df_normal.groupby('round')['normal_train_improvement'].mean()
adversarial_improvement_per_round = df_adversarial.groupby('round')['adversarial_train_improvement'].mean()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(normal_improvement_per_round, label='Normal Training Improvement', marker='o')
plt.plot(adversarial_improvement_per_round, label='Adversarial Training Improvement', marker='o')

plt.title('Training Accuracy Improvement Per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy Improvement')
plt.legend()
plt.grid(True)
plt.show()
