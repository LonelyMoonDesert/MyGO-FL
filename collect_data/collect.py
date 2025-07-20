import re
import pandas as pd

logfile = '../logs/resnet18-pacs/domain-3clients/fedtopo-layer1-2025-07-18-171327(88.89%).log'  # 换成你的日志文件名

# === 提取 loss 信息 ===
loss_rows = []
cur_client = None
cur_round = 0
epoch_in_round = 0

with open(logfile, encoding='utf8') as f:
    for line in f:
        # 识别当前client编号
        m_client = re.search(r'Training network (\d+)', line)
        if m_client:
            cur_client = int(m_client.group(1))
            epoch_in_round = 0
            continue
        # 识别epoch和loss
        m_loss = re.search(r'Epoch: (\d+) Total: ([\d\.]+) CE: ([\d\.]+) Topo: ([\d\.]+)', line)
        if m_loss and cur_client is not None:
            epoch, total, ce, topo = m_loss.groups()
            loss_rows.append({
                'client': cur_client,
                'global_epoch': cur_round*5 + int(epoch),  # 5为每轮epoch数，如有变更这里需同步改
                'comm_round': cur_round,
                'epoch_in_round': int(epoch),
                'total_loss': float(total),
                'ce_loss': float(ce),
                'topo_loss': float(topo)
            })
            epoch_in_round += 1
            # 如果一个client到5个epoch，说明进到下一轮
            if epoch_in_round == 5:
                cur_round += 1

loss_df = pd.DataFrame(loss_rows)
loss_df.to_csv('client_losses.csv', index=False)
print('client_losses.csv已保存，共', len(loss_df), '条记录')
# === 提取 test acc（client） ===
pattern_testacc = re.compile(r"net (\d+) final test acc ([\d\.]+)")
acc_rows = []
round_idx, client_counts, client_seen = -1, 0, set()
with open(logfile, encoding='utf8') as f:
    for line in f:
        m = pattern_testacc.search(line)
        if m:
            cid, acc = int(m.group(1)), float(m.group(2))
            if cid == 0:
                round_idx += 1
                client_seen = set()
            acc_rows.append({'round': round_idx, 'client': cid, 'test_acc': acc})
            client_seen.add(cid)
acc_df = pd.DataFrame(acc_rows)

# === 提取 global acc ===
pattern_global = re.compile(r">> Global Model Test accuracy: ([\d\.]+)")
global_acc_rows = []
round_idx = -1
with open(logfile, encoding='utf8') as f:
    for line in f:
        if "Global Model Test accuracy" in line:
            round_idx += 1
            acc = float(pattern_global.search(line).group(1))
            global_acc_rows.append({'round': round_idx, 'client': 'global', 'test_acc': acc})
acc_df = pd.concat([acc_df, pd.DataFrame(global_acc_rows)], ignore_index=True)
acc_df.to_csv('client_global_test_acc.csv', index=False)
print('数据已保存：client_losses.csv, client_global_test_acc.csv')
