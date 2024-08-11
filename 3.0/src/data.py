
# data.py

import json
import numpy as np
from datasets import Dataset
from datasets import load_dataset

def load_n_lines_from_jsonl(jsonl_file, n):
    data = []
    with open(jsonl_file, 'r') as file:
        for i, line in enumerate(file):
            if i >= n:
                break
            data.append(json.loads(line))
    return data

def load_and_prepare_dataset(jsonl_file, tokenizer, eos_token, n):
    # Load the first N lines from the JSONL file
    data = load_n_lines_from_jsonl(jsonl_file, n)

    # Convert the data into a Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # Define the prompt format
    prediction_prompt = """Predict the order of magnitude of the following video based on the video title:

    ### Title:
    {}

    ### Views:
    {}"""

    # Function to format and apply log10 transformation
    def formatting_prompts_func(examples):
        titles = examples["title"]
        view_counts = examples["view_count"]
        texts = []
        for title, view_count in zip(titles, view_counts):
            log_views = np.log10(view_count)
            text = prediction_prompt.format(title, log_views) + eos_token
            texts.append(text)
        return {"text": texts}

    # Apply the formatting function to the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset
