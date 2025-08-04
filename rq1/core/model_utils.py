import torch
import torch.nn.functional as F
from scipy.special import digamma
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def au_eu(logits, K=10):
    topk_values, _ = torch.topk(logits, K)
    alpha_k = F.relu(topk_values)
    alpha_0 = torch.sum(alpha_k)

    if alpha_0.item() == 0.0:
        return math.log(K), 1.0

    au = -torch.sum((alpha_k / alpha_0) * (digamma(alpha_k + 1) - digamma(alpha_0 + 1)))
    eu = K / torch.sum(alpha_k + 1)
    return au.item(), eu.item()


def arithmetic_mean_u(token_data, u, K=10, rev=False):
    valid_tokens = [entry for entry in token_data if math.isfinite(entry[u])]
    sorted_tokens = sorted(valid_tokens, key=lambda x: x[u], reverse=rev)
    worst_tokens = sorted_tokens[:min(K, len(sorted_tokens))]
    avg_u = sum(entry[u] for entry in worst_tokens) / len(worst_tokens)
    return avg_u

def parse_response_tokens(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
        return list(map(int, s.split()))
    else:
        return list(map(int, s.split(',')))

def parse_list_of_arrays(s):
    try:
        return eval(s, {"array": np.array, "float32": np.float32})
    except Exception as e:
        print(f"Failed to parse array: {s}\nError: {e}")
        return None

def generate_token_data(df, tokenizer, K=10):
    df_token_data = []

    for i, row in tqdm(df.iterrows(), desc="Token-wise AU/EU"):
        token_data = []
        try:
            response_tokens = parse_response_tokens(row['response_token']) if isinstance(row['response_token'], str) else row['response_token']
            logits_list = row['logits_v']

            for j, logits in enumerate(logits_list):
                token_id = response_tokens[j]
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                au, eu = au_eu(torch.tensor(logits[0]), K=K)
                token_data.append({
                    "token": token_str,
                    "au": au,
                    "eu": eu,
                    "token_id": token_id
                })
        except Exception as e:
            print(f"Row {i} failed: {e}")
            token_data = []

        df_token_data.append(token_data)

    return df_token_data

def error_type_classifier(df):
    unique_ids = df['id'].unique()
    for id in unique_ids:
        for context in [0, 1]:
            id_df = df[(df['id'] == id) & (df['with_context'] == context)]
            ratio = id_df['correctness_label'].astype(int).mean()
            if ratio > 0.95:
                label = 'C1'
            elif ratio > 0.6:
                label = 'C2'
            elif ratio > 0.4:
                label = 'EQ'
            elif ratio > 0.05:
                label = 'E2'
            else:
                label = 'E1'
            df.loc[id_df.index, 'error_type'] = label
    return df
