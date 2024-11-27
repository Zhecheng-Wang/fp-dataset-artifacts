import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from helpers import prepare_dataset_nli, compute_accuracy

NUM_PREPROCESSING_WORKERS = 1  # Lower number for smaller datasets

def evaluate_contrast_set(model_path, contrast_set_path, output_dir, max_length=128, max_eval_samples=None):
    # Load dataset from JSON
    with open(contrast_set_path, "r") as f:
        data = json.load(f)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)

    # If max_eval_samples is set, select a subset
    if max_eval_samples is not None and max_eval_samples < len(dataset):
        dataset = dataset.select(range(max_eval_samples))

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    eval_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,  # Lower to avoid parallelization overhead for small datasets
    )

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        evaluation_strategy="no",
        logging_dir=os.path.join(output_dir, "logs")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )

    # Run evaluation
    print(f"Evaluating contrast set: {contrast_set_path}")
    results = trainer.evaluate()
    print("Results:", results)

    # Save evaluation results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(results, f)

    print(f"Evaluation completed. Results saved to {output_dir}/eval_metrics.json")



if __name__ == "__main__":
    # Parameters
    trained_model_dir = "trained_model"  # Directory with the trained model
    contrast_sets_dir = "data/contrast_sets"  # Directory with contrast set files
    eval_output_dir = "eval_output/contrast_sets"  # Where to save evaluation results
    max_length = 128  # Maximum sequence length for tokenization
    max_eval_samples = 100  # Limit the number of evaluation samples

    os.makedirs(eval_output_dir, exist_ok=True)
    contrast_sets = [f for f in os.listdir(contrast_sets_dir) if f.endswith('.json')]

    for contrast_set in contrast_sets:
        contrast_set_path = os.path.join(contrast_sets_dir, contrast_set)
        output_dir = os.path.join(eval_output_dir, os.path.splitext(contrast_set)[0])
        evaluate_contrast_set(trained_model_dir, contrast_set_path, output_dir, max_length, max_eval_samples)
    
    # Collect contrast set sizes and their accuracies
    contrast_set_sizes = []
    accuracies = []

    for contrast_set in contrast_sets:
        contrast_set_path = os.path.join(contrast_sets_dir, contrast_set)
        output_dir = os.path.join(eval_output_dir, os.path.splitext(contrast_set)[0])
        
        # Get the size of the contrast set
        with open(contrast_set_path, 'r') as f:
            data = json.load(f)
            size = len(data['premise'])  # Assuming premise represents the number of examples

        # Get the evaluation accuracy
        eval_metrics_path = os.path.join(output_dir, 'eval_metrics.json')
        with open(eval_metrics_path, 'r') as f:
            results = json.load(f)
            accuracy = results.get('eval_accuracy', 0)  # Default to 0 if not found
        
        # Store for plotting
        contrast_set_sizes.append(size)
        accuracies.append(accuracy)

        # Sort by contrast set size
        sorted_indices = sorted(range(len(contrast_set_sizes)), key=lambda i: contrast_set_sizes[i])
        contrast_set_sizes = [contrast_set_sizes[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]

        # Plot accuracy vs. contrast set size
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(contrast_set_sizes, accuracies, marker='o')
        plt.xlabel('Contrast Set Size (Number of Examples)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Contrast Set Size')
        plt.grid(True)
        plt.savefig('accuracy_vs_contrast_set_size.png')