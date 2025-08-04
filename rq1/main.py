import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from core.env_loader import load_env
from core.model_utils import arithmetic_mean_u, parse_list_of_arrays, generate_token_data, error_type_classifier
from tqdm import tqdm
from core.prompt_utils import build_prompt_with_sources, correctness_labeler, incorrect_context_generator
import torch
import torch.nn.functional as F


# ------------- LOAD VARIABLES -------------
model_name = sys.argv[1]
dataset = sys.argv[2]
output_path = sys.argv[3]
in_context = sys.argv[4]
env = load_env()

# ------------- LOAD DATASET -------------
if dataset.endswith(".csv"):
    df = pd.read_csv(dataset)
elif dataset.endswith(".parquet"):
    df = pd.read_parquet(dataset)
else:
    raise ValueError("Unsupported dataset format. Use .csv or .parquet")


# ------------- LOAD MODEL -------------
device = 'cuda:0'

if "gemma" in model_name.lower():
    if env['hf_token'] is None:
        print("Error: hf_token is required for Gemma models.")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=env['hf_token'])
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # default threshold
    llm_int8_enable_fp32_cpu_offload=True  # useful if GPU is very tight on memory
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map=device, 
    torch_dtype=torch.bfloat16
)

try:
    print(model.device)
    print(model.hf_device_map)
except:
    pass


# ------------- FUNCTIONS -------------

def generate_response(question, sources=None, num_samples=15):
    model.eval()
    if sources:
        messages = build_prompt_with_sources(question, sources)
    else:
        messages = [
            {"role": "system", "content": "Answer the question directly, without additional explanation, and be as concise as possible."},
            {"role": "user", "content": question},
        ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    results = []
   
    for i in range(num_samples):
        output = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                                temperature=1,
                                top_k=20,
                                top_p=0.9,
                                # repetition_penalty=1.2,
                                # eos_token_id=tokenizer.eos_token_id,
                                output_logits =True,
                                return_dict_in_generate=True,
                                disable_compile=True,
                                output_hidden_states=True,
                                output_attentions=True)

        # 1. get response and token
        response_token = output.sequences[0][input_len:].cpu().numpy()
        response = tokenizer.decode(response_token,skip_special_tokens=True)

        
        # 2. get logits
        logits =  output.logits
        # 3. get log_probs
        probs = [F.softmax(r, dim=-1) for r in logits]  
        
        def obtain_topk(lst,k=100):
            value, index = [],[]
            for r in lst:
                v,idx = torch.topk(r,k)
                value.append(v.cpu().numpy())
                index.append(idx.cpu().numpy())
            return value,index
            
        logits_v,logits_idx = obtain_topk(logits)
        prob_v,prob_idx = obtain_topk(probs)
        result_dict ={"response":response,"response_token":response_token,"logits_v":logits_v,"logits_idx":logits_idx ,"prob_v":prob_v,"prob_idx": prob_idx, "k": i+1} 
        results.append(result_dict)

    return results


# ------------- EXPERIMENT -------------
sample_size = 15
all_rows = []

if in_context == 'wic':
  incorrect_contexts = []
  for i, row in tqdm(df.iterrows()):
      prompt = f'''Q: {row['question']} \nA: {row['answer']}'''
      context = incorrect_context_generator(prompt)
      incorrect_contexts.append(context)

  df['context'] = incorrect_contexts

for i, row in tqdm(df.iterrows(), total=len(df)):
    # No-context
    try:
        samples = generate_response(row['question'])
        for result_dict in samples:
            all_rows.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'context': row['context'],
                'context_len': row['context_len'],
                'answer_len': row['answer_len'],
                'response': result_dict['response'],
                'response_token': result_dict['response_token'],
                'logits_v': result_dict['logits_v'],
                'logits_idx': result_dict['logits_idx'],
                'prob_v': result_dict['prob_v'],
                'prob_idx': result_dict['prob_idx'],
                'k': result_dict['k'],
                'with_context': 0
            })
    except Exception as e:
        print(e)
        for j in range(sample_size):
            all_rows.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'context': row['context'],
                'context_len': row['context_len'],
                'answer_len': row['answer_len'],
                'response': None,
                'response_token': None,
                'logits_v': None,
                'logits_idx': None,
                'prob_v': None,
                'prob_idx': None,
                'k': j,
                'with_context': 0
            })

    # With-context
    try:
        samples = generate_response(row['question'], row['context'])
        for result_dict in samples:
            all_rows.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'context': row['context'],
                'context_len': row['context_len'],
                'answer_len': row['answer_len'],
                'response': result_dict['response'],
                'response_token': result_dict['response_token'],
                'logits_v': result_dict['logits_v'],
                'logits_idx': result_dict['logits_idx'],
                'prob_v': result_dict['prob_v'],
                'prob_idx': result_dict['prob_idx'],
                'k': result_dict['k'],
                'with_context': 1
            })
    except Exception as e:
        print(e)
        for j in range(sample_size):
            all_rows.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'context': row['context'],
                'context_len': row['context_len'],
                'answer_len': row['answer_len'],
                'response': None,
                'response_token': None,
                'logits_v': None,
                'logits_idx': None,
                'prob_v': None,
                'prob_idx': None,
                'k': j,
                'with_context': 1
            })

df_out = pd.DataFrame(all_rows)
df_out = df_out[~df_out['response'].isna()].reset_index(drop=True)

token_data = generate_token_data(df_out, tokenizer)
df_out['token_data'] = token_data


valid_labels = {"0", "1"}
for i, row in tqdm(df_out.iterrows(), desc="Labeling"):
    try:
        prompt = f"Question: {row['question']}\nAnswer: {row['answer']}\nResponse: {row['response']}"
        label = correctness_labeler(prompt)
        if label in valid_labels:
            df_out.loc[i, 'correctness_label'] = int(label)
    except Exception as e:
        print(f"Labeling failed at {i}: {e}")


df_out = df_out[df_out['correctness_label'].isin([0, 1])].reset_index(drop=True)


df_out['mean_eu_high'] = df_out['token_data'].apply(lambda x: arithmetic_mean_u(x, 'eu', rev=True))
df_out['mean_eu_low'] = df_out['token_data'].apply(lambda x: arithmetic_mean_u(x, 'eu'))
df_out['mean_au_high'] = df_out['token_data'].apply(lambda x: arithmetic_mean_u(x, 'au', rev=True))
df_out['mean_au_low'] = df_out['token_data'].apply(lambda x: arithmetic_mean_u(x, 'au'))

df_out = error_type_classifier(df_out)

df_out.to_csv(output_path)
