# Redraw the boxplots with fully English labels for better compatibility
plt.figure(figsize=(12, 8))

# Network 2 Boxplot (Train Accuracy Change vs Entropy Change Group)
plt.subplot(2, 1, 1)
sns.boxplot(
    x='Entropy Group',
    y='Train_accuracy_change',
    data=filtered_df[filtered_df['Network'] == 'Network 2'],
    palette='Set2'
)
plt.title("Network 2: Entropy Change Group vs Train Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change Group", fontsize=12)
plt.ylabel("Train Accuracy Change", fontsize=12)
plt.grid(axis='y')

# Network 3 Boxplot (Train Accuracy Change vs Entropy Change Group)
plt.subplot(2, 1, 2)
sns.boxplot(
    x='Entropy Group',
    y='Train_accuracy_change',
    data=filtered_df[filtered_df['Network'] == 'Network 3'],
    palette='Set3'
)
plt.title("Network 3: Entropy Change Group vs Train Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change Group", fontsize=12)
plt.ylabel("Train Accuracy Change", fontsize=12)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
