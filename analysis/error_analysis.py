import pandas as pd

# Load the JSONL file to examine its contents
file_path = 'eval_output/eval_predictions.jsonl'
predictions = pd.read_json(file_path, lines=True)

# Display the first few rows to understand the structure of the data
predictions.head()


# Identify misclassified examples
misclassified = predictions[predictions['label'] != predictions['predicted_label']]

# Compute the proportion of misclassifications by true label
misclassification_stats = misclassified.groupby('label').size() / predictions.groupby('label').size()

# Display misclassification statistics
misclassification_stats.name = 'Misclassification Rate'
misclassification_stats.index.name = 'True Label'
misclassification_stats = misclassification_stats.reset_index()

# Save misclassification statistics to a CSV file for review
misclassification_stats.to_csv('misclassification_statistics.csv', index=False)

# Display a random sample of misclassified examples
misclassified_examples = misclassified.sample(5)
misclassified_examples.to_csv('misclassified_examples.csv', index=False)

print("Misclassification statistics saved to 'misclassification_statistics.csv'")
print("Sample misclassified examples saved to 'misclassified_examples.csv'")
