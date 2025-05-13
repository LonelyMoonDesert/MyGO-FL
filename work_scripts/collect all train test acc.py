import os
import re
import pandas as pd

def extract_accuracies_from_logs(directory):
    # This dictionary will hold the data with keys as (network, round) and values as dicts of accuracies
    results = {}

    # Regular expressions to capture the accuracy values
    training_pattern = re.compile(r'>> Training accuracy: ([0-9\.]+)')
    test_pattern = re.compile(r'>> Test accuracy: ([0-9\.]+)')

    # Loop through each file in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it is a file and ends with .log
        if os.path.isfile(file_path) and filename.endswith('.log'):
            # Extract the method name (e.g., network name) from the filename
            method_name = filename.split('-')[0]
            print(f"Processing file: {filename}")

            # Initialize network_number and round_number to None at the start of each file
            network_number = None
            round_number = None

            with open(file_path, 'r') as file:
                # Loop through each line in the file
                for line in file:
                    # Look for training and test accuracy patterns
                    training_match = training_pattern.search(line)
                    test_match = test_pattern.search(line)

                    # Get the round number and network ID from the line
                    if 'comm round' in line:
                        round_number = int(re.search(r'comm round:(\d+)', line).group(1))
                    if 'Training network' in line:
                        network_number = int(re.search(r'Training network (\d+)', line).group(1))

                    # Skip if either round_number or network_number is not initialized
                    if network_number is None or round_number is None:
                        continue

                    # Ensure that the (network_number, round_number) key exists in the results dictionary
                    if (network_number, round_number) not in results:
                        results[(network_number, round_number)] = {'training_accuracy': [], 'test_accuracy': []}

                    # If we found a training accuracy
                    if training_match:
                        results[(network_number, round_number)]['training_accuracy'].append(
                            float(training_match.group(1)))

                    # If we found a test accuracy
                    if test_match:
                        results[(network_number, round_number)]['test_accuracy'].append(float(test_match.group(1)))

    # Flatten the dictionary into a list of rows for the DataFrame
    rows = []
    for (network, round_number), accuracies in results.items():
        rows.append({
            'network': network,
            'round': round_number,
            'training_accuracy': accuracies.get('training_accuracy', []),
            'test_accuracy': accuracies.get('test_accuracy', [])
        })

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    output_path = os.path.join(directory, 'network_accuracies_by_round.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


# Example usage
directory_path = r'before_after'  # Update this to the directory containing your log files
extract_accuracies_from_logs(directory_path)
