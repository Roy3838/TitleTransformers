
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import Dataset
from tqdm import tqdm
import json

max_seq_length = 2048
dtype = None  # Switch to FP32 for better compatibility
load_in_4bit = False   # Disable 4-bit quantization if not needed

# Load model without memory-efficient attention
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Skip LoRA and additional configurations if not necessary for inference
FastLanguageModel.for_inference(model)

# Function to load a subset of dataset
def load_dataset(file_path):
    titles = []
    view_counts = []
    with open(file_path, 'r') as f:
        for (i, line) in enumerate(tqdm(f)):
            if i > 10000:
                break
            else:
                data = json.loads(line.strip())
                titles.append(data["title"])
                view_counts.append(data["view_count"])

    dataset_dict = {"title": titles, "view_count": view_counts}
    return Dataset.from_dict(dataset_dict)

# Load your dataset
dataset = load_dataset("/mnt/datassd/train_data.jsonl")

# Define the prompt format
prediction_prompt = """Predict the order magnitude views of the following video based on the video title:

### Title:
{}

### Log Views:
{}"""

# Function to format and apply log10 transformation
def formatting_prompts_func(examples):
    titles = examples["title"]
    view_counts = examples["view_count"]
    texts = []
    for title, view_count in zip(titles, view_counts):
        log_views = int(np.log10(view_count))
        text = prediction_prompt.format(title, log_views) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Apply the formatting function to the dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

# Convert the dataset to a pandas DataFrame
df = dataset.to_pandas()

# Print the first few rows of the DataFrame
print(df['text'][0])

# Prepare the input for inference
inputs = tokenizer(
    [
        prediction_prompt.format(
            "Placeholder",  # example title
            "",  # placeholder for Log Views, should be empty
        )
    ], return_tensors="pt").to("cuda")

# Run inference without using memory-efficient attention
with torch.no_grad():
    model_output = model(**inputs)
    logits = model_output.logits

# Apply softmax to get the probability distribution
probabilities = torch.softmax(logits, dim=-1)

# Get the predicted token and its probability
last_token_probabilities = probabilities[0, -1, :]
predicted_token_id = torch.argmax(last_token_probabilities).item()
predicted_token = tokenizer.decode(predicted_token_id)
predicted_token_probability = last_token_probabilities[predicted_token_id].item()

print(f"Predicted Token: {predicted_token}")
print(f"Predicted Token Probability: {predicted_token_probability:.4f}")

# View the entire distribution for top-k tokens
top_k = 10
top_k_probabilities, top_k_indices = torch.topk(last_token_probabilities, top_k)

for i in range(top_k):
    token = tokenizer.decode(top_k_indices[i].item())
    prob = top_k_probabilities[i].item()
    print(f"Token: {token}, Probability: {prob:.4f}")
