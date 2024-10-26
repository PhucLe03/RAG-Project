import requests
import json

url ="http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

payload = {
            "model": "llama3",
            "prompt": "Why is the sky blue?",
            "stream": False
        }
response = requests.post(url, headers=headers, data=json.dumps(payload))

print(response.text)