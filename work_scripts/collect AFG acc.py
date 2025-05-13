import re
import csv
from typing import List, Dict


def extract_accuracies(log_file: str, output_csv: str) -> None:
    """
    Extract accuracy information from the log file and save it in a CSV file.

    Args:
        log_file (str): Path to the input log file.
        output_csv (str): Path to the output CSV file.
    """
    # Regular expressions to extract the relevant data
    start_training_pattern = re.compile(r"Starting adversarial training of clients...")
    pre_train_train_pattern = re.compile(r">> Pre-Training Training accuracy: ([\d.]+)")
    pre_train_test_pattern = re.compile(r">> Pre-Training Test accuracy: ([\d.]+)")
    train_train_pattern = re.compile(r">> Training accuracy: ([\d.]+)")
    train_test_pattern = re.compile(r">> Test accuracy: ([\d.]+)")

    # Regular expressions to extract normal training information
    normal_train_start_pattern = re.compile(r"Training network (\d+). n_training: (\d+)")

    # Data storage
    data: List[Dict[str, float]] = []

    try:
        with open(log_file, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{log_file}' was not found.")
        return
    except IOError as e:
        print(f"Error reading file '{log_file}': {e}")
        return

    capture_adversarial = False
    capture_normal = False
    current_entry = {}

    for line in lines:
        # Handling adversarial training
        if start_training_pattern.search(line):
            capture_adversarial = True
            capture_normal = False  # Stop capturing normal training data
            current_entry = {}
            continue

        if capture_adversarial:
            # Extract adversarial accuracies
            pre_train_train_match = pre_train_train_pattern.search(line)
            pre_train_test_match = pre_train_test_pattern.search(line)
            train_train_match = train_train_pattern.search(line)
            train_test_match = train_test_pattern.search(line)

            if pre_train_train_match:
                current_entry["adversarial_pre_training_train_acc"] = float(pre_train_train_match.group(1))
            if pre_train_test_match:
                current_entry["adversarial_pre_training_test_acc"] = float(pre_train_test_match.group(1))
            if train_train_match:
                current_entry["adversarial_training_train_acc"] = float(train_train_match.group(1))
            if train_test_match:
                current_entry["adversarial_training_test_acc"] = float(train_test_match.group(1))

            # If all fields for adversarial training are captured, save the entry
            if len(current_entry) == 4:
                data.append(current_entry)
                capture_adversarial = False  # Stop capturing until the next start marker

        # Handling normal training
        normal_train_match = normal_train_start_pattern.search(line)
        if normal_train_match:
            capture_normal = True
            capture_adversarial = False  # Stop capturing adversarial training data
            current_entry = {
                "network": normal_train_match.group(1),
                "round": normal_train_match.group(2)
            }
            continue

        if capture_normal:
            # Extract normal training accuracies
            pre_train_train_match = pre_train_train_pattern.search(line)
            pre_train_test_match = pre_train_test_pattern.search(line)
            train_train_match = train_train_pattern.search(line)
            train_test_match = train_test_pattern.search(line)

            if pre_train_train_match:
                current_entry["normal_pre_training_train_acc"] = float(pre_train_train_match.group(1))
            if pre_train_test_match:
                current_entry["normal_pre_training_test_acc"] = float(pre_train_test_match.group(1))
            if train_train_match:
                current_entry["normal_training_train_acc"] = float(train_train_match.group(1))
            if train_test_match:
                current_entry["normal_training_test_acc"] = float(train_test_match.group(1))

            # If all fields for normal training are captured, save the entry
            if len(current_entry) == 6:  # 2 (network, round) + 4 (acc)
                data.append(current_entry)
                capture_normal = False  # Stop capturing until the next start marker

    if data:
        try:
            # Write the data to a CSV file
            with open(output_csv, 'w', newline='') as csvfile:
                fieldnames = [
                    "network", "round",
                    "normal_pre_training_train_acc", "normal_pre_training_test_acc",
                    "normal_training_train_acc", "normal_training_test_acc",
                    "adversarial_pre_training_train_acc", "adversarial_pre_training_test_acc",
                    "adversarial_training_train_acc", "adversarial_training_test_acc"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            print(f"Data has been extracted and saved to {output_csv}.")
        except IOError as e:
            print(f"Error writing to CSV file '{output_csv}': {e}")
    else:
        print("No data found to extract.")

# Example usage
log_file_path = "before_after/fedavg.log"  # Replace with your log file path
output_csv_path = "client_accuracies.csv"  # Replace with your desired output CSV file path
extract_accuracies(log_file_path, output_csv_path)

print(f"Data has been extracted and saved to {output_csv_path}.")
