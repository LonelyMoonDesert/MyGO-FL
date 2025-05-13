import matplotlib.pyplot as plt
import pandas as pd

# Data for FedAFG and FedAvg (could be loaded from CSV or parsed directly)
# Example structure based on your data (adjust this as per your format)
fedafg_data = {
    'Round': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Entropy': [6.610919, 6.634334, 6.663237, 6.665559, 6.6944, 6.691469, 6.712406, 6.731045, 6.742549, 6.752489],
    'Train Accuracy': [0.6781, 0.6454, 0.7472, 0.7743, 0.8113, 0.818, 0.8657, 0.8582, 0.9075, 0.9188],
    'Test Accuracy': [0.6484, 0.634, 0.7013, 0.7313, 0.7499, 0.7643, 0.7991, 0.7863, 0.8226, 0.8247]
}

fedavg_data = {
    'Round': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Entropy': [6.621871, 6.609939, 6.603573, 6.60646, 6.608447, 6.610233, 6.610618, 6.613698, 6.617594, 6.618586],
    'Train Accuracy': [0.5659, 0.635, 0.7237, 0.7275, 0.7416, 0.8014, 0.8299, 0.8369, 0.8657, 0.8722],
    'Test Accuracy': [0.5548, 0.6207, 0.6843, 0.6982, 0.6911, 0.7515, 0.7625, 0.7706, 0.7914, 0.7905]
}

# Create DataFrames
fedafg_df = pd.DataFrame(fedafg_data)
fedavg_df = pd.DataFrame(fedavg_data)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot Entropy comparison
axs[0].plot(fedafg_df['Round'], fedafg_df['Entropy'], label='FedAFG Entropy', color='blue', marker='o')
axs[0].plot(fedavg_df['Round'], fedavg_df['Entropy'], label='FedAvg Entropy', color='red', marker='x')
axs[0].set_title('Entropy Comparison (FedAFG vs FedAvg)', fontsize=24)
axs[0].set_xlabel('Round', fontsize=22)
axs[0].set_ylabel('Entropy', fontsize=22)
axs[0].tick_params(axis='both', which='major', labelsize=22)
axs[0].legend(fontsize=24)
axs[0].grid(True)  # Ensure grid is on

# Plot Train Accuracy comparison
axs[1].plot(fedafg_df['Round'], fedafg_df['Train Accuracy'], label='FedAFG Train Accuracy', color='blue', marker='o')
axs[1].plot(fedavg_df['Round'], fedavg_df['Train Accuracy'], label='FedAvg Train Accuracy', color='red', marker='x')
axs[1].set_title('Train Accuracy Comparison (FedAFG vs FedAvg)', fontsize=24)
axs[1].set_xlabel('Round', fontsize=22)
axs[1].set_ylabel('Train Accuracy', fontsize=22)
axs[1].tick_params(axis='both', which='major', labelsize=22)
axs[1].legend(fontsize=24)
axs[1].grid(True)  # Ensure grid is on

# Plot Test Accuracy comparison
axs[2].plot(fedafg_df['Round'], fedafg_df['Test Accuracy'], label='FedAFG Test Accuracy', color='blue', marker='o')
axs[2].plot(fedavg_df['Round'], fedavg_df['Test Accuracy'], label='FedAvg Test Accuracy', color='red', marker='x')
axs[2].set_title('Test Accuracy Comparison (FedAFG vs FedAvg)', fontsize=24)
axs[2].set_xlabel('Round', fontsize=22)
axs[2].set_ylabel('Test Accuracy', fontsize=22)
axs[2].tick_params(axis='both', which='major', labelsize=22)
axs[2].legend(fontsize=24)
axs[2].grid(True)  # Ensure grid is on

# Adjust layout to ensure content is not cut off
plt.tight_layout()

# Show the plot
plt.show()
