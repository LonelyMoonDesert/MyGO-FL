import pandas as pd
import matplotlib.pyplot as plt

# Sample data to represent the provided Excel data
data = {
    "round": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "FedAvg": [0.1047, 0.2095, 0.2878, 0.3357, 0.3599, 0.3749, 0.4119, 0.4119, 0.4364, 0.4442],
    "FedAFG": [0.1855, 0.3251, 0.3708, 0.4057, 0.4364, 0.4682, 0.4799, 0.4995, 0.5274, 0.5383],
    "FedNova": [0.1771, 0.2214, 0.2864, 0.3255, 0.353, 0.3759, 0.4113, 0.4113, 0.4482, 0.4524],
    "FedProx": [0.1779, 0.2091, 0.2907, 0.3405, 0.363, 0.3711, 0.4132, 0.4143, 0.4332, 0.4405],
    "SCAFFOLD": [0.1, 0.2171, 0.2969, 0.3343, 0.3952, 0.4055, 0.4639, 0.4664, 0.5018, 0.5321]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# Plot each method
for column in df.columns[1:]:
    plt.plot(df['round'], df[column], marker='o', label=column)

# Chart title and labels
plt.title('Global Model Test Accuracy per Round', fontsize=16)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.xticks(df['round'])
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Legend
plt.legend(title='Methods', fontsize=12, title_fontsize=14)

# Save and display plot
plt.tight_layout()
plt.savefig('global_model_test_accuracy_plot.png')
plt.show()
