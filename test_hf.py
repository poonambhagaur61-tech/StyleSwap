import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN was not found. Check your .env file.")

client = InferenceClient(
    provider="featherless-ai",
    api_key=token
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Say hello to StyleSwap in one short sentence."
        }
    ],
    max_tokens=100,
)

print("\nAI RESPONSE:")
print(response.choices[0].message.content)