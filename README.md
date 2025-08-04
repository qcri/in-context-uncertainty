# Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models

The code in this paper can be used to reproduce the results in the paper.

Huggingface models used in the experiments: 
- Fanar1-9B https://huggingface.co/QCRI/Fanar-1-9B-Instruct
- Qwen2.5-7B https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Gemma3-12B https://huggingface.co/google/gemma-3-12b-it

## Preliminary step for all: Generating model's answers with relevant metadata
```
rq1/main.py model_name=[HF mdoel name] dataset=[path_to_dataset] output_path=[output_path] in_context=[wcc/wic]
```

Additionally, .env needs to be set up with the organizational/personal access keys to Azure OpenAI and Huggingface
```
AZURE_OPENAI_ENDPOINT=your-endpoint-here
AZURE_OPENAI_API_KEY=your-key-here
AZURE_DEPLOYMENT_NAME=gpt-4.1-mini-llm-reliability
AZURE_API_VERSION=2024-12-01-preview
HF_TOKEN=your-hftoken-here
```


