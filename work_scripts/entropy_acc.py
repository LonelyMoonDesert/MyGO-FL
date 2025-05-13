# Redraw the scatter plots with fully English labels for the earlier example (Network 2 and Network 3 Entropy Change vs Accuracy Changes)

plt.figure(figsize=(14, 8))

# Network 2: Entropy Change vs Train Accuracy Change
plt.subplot(2, 2, 1)
sns.scatterplot(
    x=network_2_data['Entropy_change'],
    y=network_2_data['Train_accuracy_change'],
    color='blue', label='Train Accuracy Δ'
)
sns.regplot(
    x=network_2_data['Entropy_change'],
    y=network_2_data['Train_accuracy_change'],
    scatter=False, color='red'
)
plt.title("Network 2: Entropy Change vs Train Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change (ΔEntropy)", fontsize=12)
plt.ylabel("Train Accuracy Change (ΔTrain Accuracy)", fontsize=12)
plt.grid()

# Network 2: Entropy Change vs Test Accuracy Change
plt.subplot(2, 2, 2)
sns.scatterplot(
    x=network_2_data['Entropy_change'],
    y=network_2_data['Test_accuracy_change'],
    color='green', label='Test Accuracy Δ'
)
sns.regplot(
    x=network_2_data['Entropy_change'],
    y=network_2_data['Test_accuracy_change'],
    scatter=False, color='red'
)
plt.title("Network 2: Entropy Change vs Test Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change (ΔEntropy)", fontsize=12)
plt.ylabel("Test Accuracy Change (ΔTest Accuracy)", fontsize=12)
plt.grid()

# Network 3: Entropy Change vs Train Accuracy Change
plt.subplot(2, 2, 3)
sns.scatterplot(
    x=network_3_data['Entropy_change'],
    y=network_3_data['Train_accuracy_change'],
    color='blue', label='Train Accuracy Δ'
)
sns.regplot(
    x=network_3_data['Entropy_change'],
    y=network_3_data['Train_accuracy_change'],
    scatter=False, color='red'
)
plt.title("Network 3: Entropy Change vs Train Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change (ΔEntropy)", fontsize=12)
plt.ylabel("Train Accuracy Change (ΔTrain Accuracy)", fontsize=12)
plt.grid()

# Network 3: Entropy Change vs Test Accuracy Change
plt.subplot(2, 2, 4)
sns.scatterplot(
    x=network_3_data['Entropy_change'],
    y=network_3_data['Test_accuracy_change'],
    color='green', label='Test Accuracy Δ'
)
sns.regplot(
    x=network_3_data['Entropy_change'],
    y=network_3_data['Test_accuracy_change'],
    scatter=False, color='red'
)
plt.title("Network 3: Entropy Change vs Test Accuracy Change", fontsize=14)
plt.xlabel("Entropy Change (ΔEntropy)", fontsize=12)
plt.ylabel("Test Accuracy Change (ΔTest Accuracy)", fontsize=12)
plt.grid()

plt.tight_layout()
plt.show()
