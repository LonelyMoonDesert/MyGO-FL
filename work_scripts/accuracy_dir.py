import os
import re
import pandas as pd

def extract_accuracy_from_logs(directory):
    # This dictionary will hold the data with keys as method names and values as lists of accuracies
    results = {}

    # Regular expression to capture the accuracy value following the target phrase in log files
    accuracy_pattern = re.compile(r'>> Global Model Test accuracy: ([0-9\.]+)')

    # Loop through each file in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.log'):
            # Extract the method name from the filename using naming convention before the date
            method_name = filename.split('-')[0]
            with open(os.path.join(directory, filename), 'r') as file:
                # List to store accuracies for this particular method
                accuracies = []

                # Read the file line by line
                for line in file:
                    # Search for the accuracy pattern
                    match = accuracy_pattern.search(line)
                    if match:
                        # Append the accuracy to the list
                        accuracies.append(float(match.group(1)))

                # Append the list of accuracies to the results dictionary under the method name
                if method_name in results:
                    results[method_name].extend(accuracies)  # Append to existing method's list
                else:
                    results[method_name] = accuracies  # Start a new list for new methods

    # Create a DataFrame from the results dictionary
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(directory, 'global_model_test_accuracies.csv'), index=False)

# Example usage
directory_path = '../logs/resnet18-cifar10/label+feature_skew(noise0.5 beta0.5)-5clients'  # Change this to your log files directory
extract_accuracy_from_logs(directory_path)
