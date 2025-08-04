from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import ast
import os

def parse_list_of_arrays(s):
    try:
        return eval(s, {"array": np.array, "float32": np.float32})
    except Exception as e:
        print(f"Failed to parse:\n{s}\nError: {e}\n")
        return None

def negative_product_reliability(au, eu):
    return -au * eu

def arithmetic_mean_full_response_reliability(token_data, K=10):
    valid_tokens = [entry for entry in token_data if math.isfinite(entry["reliability"])]

    sorted_tokens = sorted(valid_tokens, key=lambda x: x["reliability"])

    worst_tokens = sorted_tokens[:min(K, len(sorted_tokens))]

    avg_reliability = sum(entry["reliability"] for entry in worst_tokens) / len(worst_tokens)
    return avg_reliability

def plot_one_distribution_on_ax(from_cat, to_cat, wc_subset, woc_df, start, end, dist='all',
                                feature='eu', token_data_col='token_data', mean_col=None,
                                ax=None, label_wc=None, label_woc=None):

    wc_ids = wc_subset['id'].unique()
    woc_subset = woc_df[woc_df['id'].isin(wc_ids)]

    if dist == 'all':
        x_wc = [float(tok[feature]) for _, row in wc_subset.iterrows() for tok in row[token_data_col] if tok[feature] > start]
        x_woc = [float(tok[feature]) for _, row in woc_subset.iterrows() for tok in row[token_data_col] if tok[feature] > start]
    else:
        x_wc = wc_subset[mean_col][wc_subset[mean_col] > start]
        x_woc = woc_subset[mean_col][woc_subset[mean_col] > start]

    sns.kdeplot(x_woc, fill=True, color='crimson', label=label_woc or f'WOC ({from_cat})', alpha=0.3, ax=ax)
    sns.kdeplot(x_wc, fill=True, color='royalblue', label=label_wc or f'WC ({to_cat})', alpha=0.3, ax=ax)

    ax.set_xlim(start, end)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True)


file_paths = [
    "/rq1/data/fanar_hotpotqa_wcc.parquet",
    "/rq1/data/fanar_hotpotqa_wic.parquet",
    "/rq1/data/fanar_nq_wcc.parquet",
    "/rq1/data/fanar_nq_wic.parquet",
    "/rq1/data/qwen_hotpotqa_wcc.parquet",
    "/rq1/data/qwen_hotpotqa_wic.parquet",
    "/rq1/data/qwen_nq_wcc.parquet",
    "/rq1/data/qwen_nq_wic.parquet",
    "/rq1/data/gemma_hotpotqa_wcc.parquet",
    "/rq1/data/gemma_hotpotqa_wic.parquet",
    "/rq1/data/gemma_nq_wcc.parquet",
    "/rq1/data/gemma_nq_wic.parquet",
]

dfs = {}

for path in file_paths:
    name = os.path.splitext(os.path.basename(path))[0]  
    df = pd.read_parquet(path)
    df["token_data"] = df["token_data"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["logits_v"] = df["logits_v"].apply(parse_list_of_arrays)
    dfs[name] = df


dfs = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
for d in dfs:
    token_data = d['token_data'].apply(lambda x: copy.deepcopy(x))

    for i, response in tqdm(enumerate(token_data)):
        for j, token in enumerate(response):
            token_data[i][j]['reliability'] = negative_product_reliability(token['au'], token['eu'])
    d['token_data_ip'] = token_data

    for i, row in tqdm(d.iterrows()):
        d.loc[i, 'response_reliability_am'] = arithmetic_mean_full_response_reliability(row['token_data_ip'])


def plot_figure(from_cat, to_cat, start, end, dist='all',
                                feature='eu', token_data_col='token_data', mean_col=None, filename=None):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Liberation Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14,
    })

    
    models_1 = {"Fanar-9B": (fanar_hotpotqa_wcc, fanar_hotpotqa_wic),
                "Qwen-7B": (qwen_hotpotqa_wcc, qwen_hotpotqa_wic),
                "Gemma-12B": (gemma_hotpotqa_wcc, gemma_hotpotqa_wic)}

    models_2 = {"Fanar-9B": (fanar_nq_wcc, fanar_nw_wic),
        "Qwen-7B": (qwen_nq_wcc, qwen_nq_wic),
        "Gemma-12B": (gemma_nq_wcc, gemma_nq_wic)}

    model_names = list(models_1.keys())
    dataset_titles = ["HotpotQA", "Google Natural Questions"]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, model_name in enumerate(model_names):
        df1_cc, df1_ic = models_1[model_name]
        df2_cc, df2_ic = models_2[model_name]

        # --- Dataset 1: Correct Context ---
        woc_E = df1_cc[df1_cc['with_context'] == 0 & df1_cc['error_type'].isin(['E1', 'E2'])]
        wc_C = df1_cc[df1_cc['with_context'] == 1 & df1_cc['error_type'].isin(['C1', 'C2'])]
        wc_C_filtered = wc_C[wc_C['id'].isin(woc_E['id'].unique())]
        plot_one_distribution_on_ax(from_cat, to_cat, wc_C_filtered, woc_E, start, end,
            dist, feature, token_data_col, mean_col, axes[i][0],
            label_woc='WOC (E)', label_wc='WCC (C)')

        # --- Dataset 1: Incorrect Context ---
        woc_C = df1_cc[df1_cc['with_context'] == 0 & df1_cc['error_type'].isin(['C1', 'C2'])]
        wc_E = df1_ic[df1_ic['with_context'] == 1 & df1_ic['error_type'].isin(['E1', 'E2'])]
        wc_E_filtered = wc_E[wc_E['id'].isin(woc_C['id'].unique())]
        plot_one_distribution_on_ax(from_cat, to_cat, wc_E_filtered, woc_C, start, end,
            dist, feature, token_data_col, mean_col, axes[i][1],
            label_woc='WOC (C)', label_wc='WIC (E)')

        # --- Dataset 2: Correct Context ---
        woc_E2 = df2_cc[df2_cc['with_context'] == 0 & df2_cc['error_type'].isin(['E1', 'E2'])]
        wc_C2 = df2_cc[df2_cc['with_context'] == 1 & df2_cc['error_type'].isin(['C1', 'C2'])]
        wc_C_filtered2 = wc_C2[wc_C2['id'].isin(woc_E2['id'].unique())]
        plot_one_distribution_on_ax(from_cat, to_cat, wc_C_filtered2, woc_E2, start, end,
            dist, feature, token_data_col, mean_col, axes[i][2],
            label_woc='WOC (E)', label_wc='WCC (C)')

        # --- Dataset 2: Incorrect Context ---
        woc_C2 = df2_cc[df2_cc['with_context'] == 0 & df2_cc['error_type'].isin(['C1', 'C2'])]
        wc_E2 = df2_ic[df2_ic['with_context'] == 1 & df2_ic['error_type'].isin(['E1', 'E2'])]
        wc_E_filtered2 = wc_E2[wc_E2['id'].isin(woc_C2['id'].unique())]
        plot_one_distribution_on_ax(from_cat, to_cat, wc_E_filtered2, woc_C2, start, end,
            dist, feature, token_data_col, mean_col, axes[i][3],
            label_woc='WOC (C)', label_wc='WIC (E)')


        row_pos = axes[i][0].get_position()
        y_center = ((row_pos.y0 + row_pos.y1) / 2) + 0.03
        fig.text(0.03, y_center, model_name, ha='right', va='center',
                fontsize=18, rotation=90)

    # === Dataset titles centered across two columns ===
    fig.text(0.28, 0.999, "HotpotQA", ha='center', fontsize=18)
    fig.text(0.75, 0.999, "Google Natural Questions", ha='center', fontsize=18)

    # === Sub-column context descriptions ===
    fig.text(0.170, 0.965, "Correct Context", ha='center', fontsize=16)
    fig.text(0.400, 0.965, "Incorrect Context", ha='center', fontsize=16)
    fig.text(0.650, 0.965, "Correct Context", ha='center', fontsize=16)
    fig.text(0.875, 0.965, "Incorrect Context", ha='center', fontsize=16)

    # === Common labels ===
    fig.text(0.52, 0.08, feature.upper(), ha='center', fontsize=16)
    fig.text(0.99, 0.53, 'Density', va='center', rotation=270, fontsize=16)

        # === Add per-column legends ===
    col_legend_locs = [0.165, 0.40, 0.64, 0.875] 

    for col in range(4):
        handles, labels = axes[0][col].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='upper center',
                   bbox_to_anchor=(col_legend_locs[col], 0.96),  
                   frameon=False, ncol=2, fontsize=16)

    plt.tight_layout(rect=[0.03, 0.1, 1, 0.93])  
    
    if filename:
        fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved to {filename}")
    else:
        plt.show()

    plt.close(fig) 
    
plot_figure(from_cat='C', to_cat='E', start=0, end=0.1, dist='mean', feature='eu', mean_col='mean_eu_low',)