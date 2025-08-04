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

df['true_label'] = df['gpt_output'].apply(extract_label)

token_data = df['token_data']
for i, response in tqdm(enumerate(token_data)):
    for j, token in enumerate(response):
        token_data[i][j]['reliability'] = negative_product_reliability(token['au'], token['eu'])
df['token_data'] = token_data


for i, row in tqdm(df.iterrows()):
    df.loc[i, 'response_reliability'] = arithmetic_mean_full_response_reliability(row['token_data'])

valid = df.dropna(subset=['true_label'])

auroc = roc_auc_score(valid['true_label'], valid['response_reliability'])
print('AUROC: ', auroc)


