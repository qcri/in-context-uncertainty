import pandas as pd
from sklearn.metrics import roc_auc_score
import json


datasets = [
    'fanar_math.json', 'fanar_triviaqa.json', 'fanar_truthfulqa.json',
    'qwen_math.json', 'qwen_triviaqa.json', 'qwen_truthfulqa.json',
    'gemma_math.json', 'gemma_triviaqa.json', 'gemma_truthfulqa.json'
]

for filename in datasets:
    var_name = filename.replace('.json', '')
    df = pd.read_json(f'/path_to_folder/{filename}')
    
    def extract_label(x):
        try:
            if isinstance(x, str):
                x = json.loads(x)
            return x.get("label")
        except Exception as e:
            return None

    df['true_label'] = df['gpt_output'].apply(extract_label)

    print(f"\n--- {var_name} ---")
    print("Predicted label counts:")
    print(df['ptrue_label'].value_counts(dropna=False))
    
    print("\nGround truth label counts:")
    print(df['true_label'].value_counts(dropna=False))
    
    # Drop NaNs before calculating AUROC
    valid = df.dropna(subset=['ptrue_label', 'true_label'])
    
    # Check if ground truth has more than one class
    if valid['true_label'].nunique() < 2:
        print("AUROC cannot be computed: only one class present in ground truth.")
        continue

    try:
        score = roc_auc_score(valid['true_label'], valid['ptrue_label'])
        print(f"\nAUROC: {score:.4f}")
    except Exception as e:
        print(f"\nError computing AUROC: {e}")
