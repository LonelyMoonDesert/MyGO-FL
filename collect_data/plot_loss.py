import matplotlib.pyplot as plt
import pandas as pd

acc_df = pd.read_csv('client_global_test_acc.csv')

plt.figure(figsize=(8,5))
for client in sorted(acc_df['client'].unique(), key=lambda x: (str(x)!='global', x)):
    sub = acc_df[acc_df['client']==client]
    plt.plot(sub['round'], sub['test_acc'], marker='o', label=f'{"Global" if client=="global" else "Client "+str(client)}')
plt.xlabel('Communication Round', fontsize=18)
plt.ylabel('Test Accuracy', fontsize=18)
plt.title('Test Accuracy of Clients and Global Model', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('testacc_all.png', dpi=350)
plt.close()
print('Test accuracy曲线已保存：testacc_all.png')
