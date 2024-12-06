import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
contrast_set_dirs = [
    "eval_output/contrast_sets/contrast_set_10",
    "eval_output/contrast_sets/contrast_set_20",
    "eval_output/contrast_sets/contrast_set_30",
    "eval_output/contrast_sets/contrast_set_50",
    "eval_output/contrast_sets/contrast_set_100"
]

# Function to load predictions
def load_predictions(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

# Initialize summary lists
contrast_set_sizes = []
accuracies = []
misclassification_stats = []

# Initialize lists for misclassified examples
contrast_misclassified_examples = []

# Analyze each contrast set
for contrast_set_dir in contrast_set_dirs:
    eval_predictions_path = os.path.join(contrast_set_dir, 'eval_predictions.jsonl')
    
    # Load predictions
    predictions = load_predictions(eval_predictions_path)
    
    # Identify misclassified examples
    misclassified = predictions[predictions['label'] != predictions['predicted_label']]
    # Add contrast set information to misclassified examples
    misclassified['contrast_set'] = os.path.basename(contrast_set_dir)

    
    # Compute accuracy
    accuracy = (predictions['label'] == predictions['predicted_label']).mean()
    accuracies.append(accuracy)
    
    # Misclassification analysis
    misclassified = predictions[predictions['label'] != predictions['predicted_label']]
    misclassified_stats = misclassified.groupby('label').size() / predictions.groupby('label').size()
    misclassified_stats = misclassified_stats.fillna(0).reset_index()
    misclassified_stats.columns = ['True Label', 'Misclassification Rate']
    misclassified_stats['Contrast Set'] = os.path.basename(contrast_set_dir)
    misclassification_stats.append(misclassified_stats)
    contrast_misclassified_examples.append(misclassified)
    
    # Store contrast set size
    contrast_set_sizes.append(len(predictions))



# Combine misclassification statistics
misclassification_df = pd.concat(misclassification_stats, ignore_index=True)

# Save to CSV
misclassification_df.to_csv('contrast_set_misclassification_statistics.csv', index=False)

# Plot accuracy vs. contrast set size
plt.figure(figsize=(8, 6))
plt.plot(contrast_set_sizes, accuracies, marker='o')
plt.xlabel('Contrast Set Size (Number of Examples)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Contrast Set Size')
plt.grid(True)
plt.savefig('accuracy_vs_contrast_set_size.png')

print("Misclassification statistics saved to 'contrast_set_misclassification_statistics.csv'")
print("Accuracy plot saved to 'accuracy_vs_contrast_set_size.png'")

# Combine all misclassified examples into a single DataFrame
contrast_misclassified_df = pd.concat(contrast_misclassified_examples, ignore_index=True)


# Display a random sample of misclassified examples
contrast_misclassified_df = misclassified.sample(5)
# Save misclassified examples to CSV
contrast_misclassified_df.to_csv('contrast_misclassified_examples.csv', index=False)

print("Misclassified examples for contrast sets saved to 'contrast_misclassified_examples.csv'")
