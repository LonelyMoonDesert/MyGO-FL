import matplotlib.pyplot as plt
import pandas as pd

loss_df = pd.read_csv('client_losses.csv')

# 画三种loss，每种一张图
for loss_type in ['total_loss', 'ce_loss', 'topo_loss']:
    plt.figure(figsize=(8,5))
    for client in sorted(loss_df['client'].unique()):
        df_c = loss_df[loss_df['client']==client]
        plt.plot(df_c['global_epoch'], df_c[loss_type], label=f'Client {client}', linewidth=2)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel(loss_type.replace('_', ' ').title(), fontsize=18)
    plt.title(f'{loss_type.replace("_", " ").title()} vs Epoch', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'loss_client_{loss_type}.png', dpi=350)
    plt.close()
print('Loss曲线已保存：loss_client_*.png')

