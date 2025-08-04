from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()
    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment": os.getenv("AZURE_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_API_VERSION"),
        "hf_token": os.getenv('HF_TOKEN')
    }
