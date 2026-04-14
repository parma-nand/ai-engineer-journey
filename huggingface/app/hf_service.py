import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

def summarize_text(text: str):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 1300,
            "min_length": 1,
            "do_sample": False
        }
    }

    for _ in range(5):  # retry 5 times
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            return {"error": "API request failed"}

        result = response.json()

        # ✅ Handle model loading
        if isinstance(result, dict) and "error" in result:
            if "loading" in result["error"].lower():
                print("Model loading... retrying")
                time.sleep(3)
                continue
            else:
                return result

        return result

    return {"error": "Model took too long to load"}