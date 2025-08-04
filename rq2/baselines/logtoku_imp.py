import pandas as pd
from sklearn.metrics import roc_auc_score
from rq1.core.env_loader import load_env
import json
import numpy as np
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
import math


model_name = sys.argv[1]
dataset = sys.argv[2]
env = load_env()

def arithmetic_mean_full_response_reliability(token_data, K=10):
    valid_tokens = [entry for entry in token_data if math.isfinite(entry["reliability"])]

    sorted_tokens = sorted(valid_tokens, key=lambda x: x["reliability"])

    worst_tokens = sorted_tokens[:min(K, len(sorted_tokens))]

    if not worst_tokens:
        return -math.log(K)

    avg_reliability = sum(entry["reliability"] for entry in worst_tokens) / len(worst_tokens)
    return avg_reliability

def negative_product_reliability(au, eu):
    return -au * eu

if dataset.endswith(".csv"):
    df = pd.read_csv(dataset)
elif dataset.endswith(".parquet"):
    df = pd.read_parquet(dataset)
elif dataset.endswith(".json"):
  df = pd.read_json(dataset)
else:
    raise ValueError("Unsupported dataset format. Use .csv or .parquet")


if "gemma" in model_name.lower():
    if env['hf_token'] is None:
        print("Error: hf_token is required for Gemma models.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=env['hf_token'])
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

def extract_label(x):
    try:
        if isinstance(x, str):
            x = json.loads(x)
        return x.get("label")
    except Exception as e:
        return None


def extract_answer(x):
  try:
      if isinstance(x, str):
          x = json.loads(x)
      return x.get("exact_answer")
  except Exception as e:
      return None

def extract_token_data_subset(row, tokenizer):
  exact_answer = row['exact_answer']
  token_data = row['token_data']

  # Skip if exact_answer is missing or not a string
  if not isinstance(exact_answer, str) or not exact_answer.strip():
      print(f"Skipping row {row.name} because exact_answer is missing or not a string")
      return token_data

  if not isinstance(token_data, list):
      print(f"Skipping row {row.name} because token data is not a list")
      return []

  # Build a mapping from token_id to full token data entry (assumes no duplicate token_ids)
  token_id_to_data = {entry['token_id']: entry for entry in token_data if 'token_id' in entry}

  # Tokenize the exact answer
  token_ids = tokenizer.encode(exact_answer, add_special_tokens=False)

  # Retrieve full token_data entries matching token_ids
  matched_token_data = [token_id_to_data.get(tid) for tid in token_ids if tid in token_id_to_data]
  return matched_token_data

df['true_label'] = df['gpt_output'].apply(extract_label)
df['exact_answer'] = df['gpt_output'].apply(extract_answer)
df['relevant_tokens'] = df.apply(lambda row: extract_token_data_subset(row, tokenizer), axis=1)

token_data = df['relevant_tokens']
for i, response in tqdm(enumerate(token_data)):
    for j, token in enumerate(response):
        token_data[i][j]['reliability'] = negative_product_reliability(token['au'], token['eu'])
df['relevant_tokens'] = token_data

df['response_reliability'] = df['relevant_tokens'].apply(
    lambda tokens: arithmetic_mean_full_response_reliability(tokens) if tokens else None
)

# Drop rows where true_label is NaN or response_reliability is None
valid = df.dropna(subset=['true_label', 'response_reliability'])

auroc = roc_auc_score(valid['true_label'], valid['response_reliability'])
print('AUROC: ', auroc)


    

    
    
