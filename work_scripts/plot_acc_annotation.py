import pandas as pd
import matplotlib.pyplot as plt

# Sample data to represent the provided Excel data
# Sample data to represent the provided Excel data
data = {
    "round": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "FedAvg": [0.1025, 0.3811, 0.5121, 0.5426, 0.5859, 0.6179, 0.6166, 0.6394, 0.6162, 0.651],
    "FedAFG": [0.3916, 0.5513, 0.5847, 0.6163, 0.6361, 0.649, 0.6627, 0.6537, 0.6834, 0.6887],
    "FedNova": [0.1099, 0.3802, 0.5125, 0.5627, 0.6097, 0.6254, 0.6235, 0.6424, 0.6238, 0.649],
    "FedProx": [0.1132, 0.3513, 0.4833, 0.5136, 0.5887, 0.604, 0.6132, 0.6259, 0.6176, 0.6347],
    "MOON": [0.0998, 0.2569, 0.4119, 0.4759, 0.4683, 0.4939, 0.5216, 0.4942, 0.5339, 0.5445],
    "SCAFFOLD": [0.1297, 0.2642, 0.4425, 0.5155, 0.5785, 0.5955, 0.6058, 0.6016, 0.6193, 0.6333]
}

# Create DataFrame
df = pd.DataFrame(data)

# # Plotting
# plt.figure(figsize=(10, 6))
#
# # Plot each method
# for column in df.columns[1:]:
#     plt.plot(df['round'], df[column], marker='o', label=column)
#
# # Chart title and labels
# plt.title('Global Model Test Accuracy per Round', fontsize=16)
# plt.xlabel('Round', fontsize=14)
# plt.ylabel('Test Accuracy', fontsize=14)
# plt.xticks(df['round'])
# plt.yticks(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
#
# # Legend
# plt.legend(title='Methods', fontsize=12, title_fontsize=14)
#
# # Save and display plot
# plt.tight_layout()
# plt.savefig('global_model_test_accuracy_plot.png')
# plt.show()

# Plotting with final accuracy annotations
plt.figure(figsize=(10, 6))

# Define offsets for annotations to avoid overlapping text
offsets = {
    "FedAvg": (0.15, 0.04),
    "FedAFG": (0.15, -0.04),
    "FedNova": (0.15, 0.06),
    "FedProx": (0.15, -0.06),
    "MOON": (0.15, 0.08),
    "SCAFFOLD": (0.15, -0.08)
}


# Plot each method and add text annotation for final value
for column in df.columns[1:]:
    plt.plot(df['round'], df[column], marker='o', label=column)
    offset_x, offset_y = offsets[column]
    plt.text(df['round'].iloc[-1] + offset_x, df[column].iloc[-1] + offset_y,
             f'{df[column].iloc[-1]:.3f}', fontsize=10,
             verticalalignment='center', color=plt.gca().lines[-1].get_color())

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
plt.savefig('global_model_test_accuracy_with_annotations.png')
plt.show()
