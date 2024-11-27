from datasets import load_dataset
from openai import OpenAI
import os
import json
from tqdm import tqdm

# Load the SNLI dataset
dataset = load_dataset("snli")

def sample_test_set(n_samples):
    # Randomly sample n samples from the test set
    test_samples = dataset["test"].shuffle().select(range(n_samples))
    test_samples_list = test_samples.to_dict()
    return test_samples_list

# Set your OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def alter_with_chatgpt(premise, hypothesis, label):
    jsonified_example = {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label
    }
    prompt = f"""Given the following premise, hypothesis, and label, please generate a new hypothesis with a slight perturbation (e.g., negate the hypothesis, add irrelevant details, change numerical values) while keeping it meaningful. Also, suggest an updated label if needed. Return the altered premise, altered hypothesis, and altered label in JSON.
    
    {jsonified_example}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={ "type": "json_object" },
        messages=[
            { "role": "system", "content": prompt }
        ]
    )
    
    # Extract and return the altered example
    altered = json.loads(response.choices[0].message.content)

    return altered['premise'], altered['hypothesis'], altered['label']

def generate_contrast_set(n_samples):
    test_samples_list = sample_test_set(n_samples)
    altered_samples = {
        'premise': [],
        'hypothesis': [],
        'label': []
    }
    premise = test_samples_list['premise']
    hypothesis = test_samples_list['hypothesis']
    label = test_samples_list['label']
    for i in tqdm(range(n_samples)):
        altered_premise, altered_hypothesis, altered_label = alter_with_chatgpt(premise[i], hypothesis[i], label[i])
        altered_samples['premise'].append(altered_premise)
        altered_samples['hypothesis'].append(altered_hypothesis)
        altered_samples['label'].append(altered_label)
    return altered_samples

# Generate contrast sets of different sizes
n_samples = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
# n_samples = [10]

for n in n_samples:
    contrast_set = generate_contrast_set(n)
    with open(f"contrast_set_{n}.json", "w+") as f:
        json.dump(contrast_set, f)