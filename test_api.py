import requests
import json
import os

# Replace these values with your LiteLLM credentials
ENDPOINT_URL: str = os.getenv("LITELLM_PROXY_BASE"+"/chat/completions", "http://ai-tools.cz.intinfra.com:4004/chat/completions")
API_KEY: str = os.getenv("LITELLM_PROXY_API_KEY", os.getenv("LITELLM_API_KEY", ""))

# Model and payload details
#MODEL_NAME = "gpt-4.1-2025-04-14"
#MODEL_NAME = "gpt-4o"
#MODEL_NAME = "claude-sonnet-4-5-20250929"
#MODEL_NAME = "claude-haiku-4-5-20251001"
MODEL_NAME = "gpt-5-mini"
#MODEL_NAME = "pool/gpt-5.1"
#MODEL_NAME = "pool/gpt-5.2"


# Prepare chat-based messages
MESSAGES = [
    {"role": "system", "content": "You are an AI assistant skilled at answering questions."},
    {"role": "user", "content": "How does the process of photosynthesis work in plants?"}
]

def test_litellm():
    """Test LiteLLM API using the chat-based `messages` API format."""
    
    # Construct headers and payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": MESSAGES,
        "max_tokens": 150,  # Adjust for longer or shorter responses
        "temperature": 0.7  # Controls randomness: lower is more deterministic, higher is more creative
    }
    
    try:
        # Send a POST request to the LiteLLM API
        response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(result["choices"][0]["message"]["content"])
        else:
            print(f"Error {response.status_code}: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to LiteLLM API: {e}")

# Run the script
if __name__ == "__main__":
    test_litellm()
