import re
import csv

# 读取日志文件内容
log_file_path = '1.txt'  # 修改为你的日志文件路径

with open(log_file_path, 'r') as file:
    log_data = file.read()

# 使用正则表达式提取网络编号和对应的熵值
network_numbers = re.findall(r"Training network (\d+). n_training: ", log_data)
entropy_values = re.findall(r">> Entropy: ([\d.]+)", log_data)
train_accuracies = re.findall(r">> Training accuracy: ([\d.]+)", log_data)
test_accuracies = re.findall(r">> Test accuracy: ([\d.]+)", log_data)

# 检查提取的数据
if len(entropy_values) != len(network_numbers):
    print(len(entropy_values), len(network_numbers))
    print("提取的熵值和网络编号数量不匹配，请检查日志文件的格式！")
else:
    # 将数据写入CSV文件
    output_file = "entropy_values.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Network", "Entropy Value", "Train Acc", "Test Acc"])  # 写入标题
        for i in range(len(entropy_values)):
            writer.writerow([f"Network {network_numbers[i]}", entropy_values[i], train_accuracies[i], test_accuracies[i]])  # 按照正确的编号写入熵值

    print(f"Entropy values have been written to {output_file}")
