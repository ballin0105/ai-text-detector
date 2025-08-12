import requests
import json

# 配置
API_KEY = "sk-CkeD2M8VvmLtzBClduppBPw1ySKm51PefqLnKsOo4wg5g0zS"  # 您的密钥
BASE_URL = "https://www.chataiapi.com"
MODEL = "gemini-2.0-flash"

url = BASE_URL + "/v1/chat/completions"
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Hello, write a short test response."}
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")