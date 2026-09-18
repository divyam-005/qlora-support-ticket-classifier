import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen-support"

def classify_ticket(ticket: str) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": ticket, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"].strip())

def main():
    ticket = input("Enter support ticket: ").strip()
    print(json.dumps(classify_ticket(ticket), indent=2))

if __name__ == "__main__":
    main()
