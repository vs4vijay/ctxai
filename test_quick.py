#!/usr/bin/env python3
"""Quick test of DeepSeek chat model."""

import json
import sys
from pathlib import Path

import requests

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Get API key
keystore_path = Path.home() / ".ctxai" / "keys.json"
with open(keystore_path) as f:
    api_key = json.load(f).get("openrouter")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/ctxai",
    "X-Title": "ctxai",
    "Content-Type": "application/json",
}

# Test 1: Simple chat
print("Test 1: Simple Chat with deepseek/deepseek-chat")
print("=" * 60)

body = {
    "model": "deepseek/deepseek-chat",
    "messages": [{"role": "user", "content": "Say 'Hello!' and nothing else."}],
    "temperature": 0.7,
    "max_tokens": 100,
}

response = requests.post(url, headers=headers, json=body, timeout=30)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    print(f"[OK] Response: {content}")
else:
    print(f"[ERROR] {response.text}")
    sys.exit(1)

# Test 2: Function calling
print("\n\nTest 2: Function Calling")
print("=" * 60)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "The city name"}},
                "required": ["location"],
            },
        },
    }
]

body = {
    "model": "deepseek/deepseek-chat",
    "messages": [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}],
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0.7,
    "max_tokens": 500,
}

response = requests.post(url, headers=headers, json=body, timeout=30)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    message = data["choices"][0]["message"]

    if message.get("tool_calls"):
        print("[OK] Model requested tool call:")
        for tc in message["tool_calls"]:
            print(f"  Function: {tc['function']['name']}")
            print(f"  Arguments: {tc['function']['arguments']}")

        # Now send the tool result back
        print("\n\nTest 3: Sending tool result back")
        print("=" * 60)

        messages = [
            {"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."},
            {"role": "assistant", "content": message.get("content", ""), "tool_calls": message["tool_calls"]},
            {
                "role": "tool",
                "tool_call_id": message["tool_calls"][0]["id"],
                "name": "get_weather",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        body2 = {
            "model": "deepseek/deepseek-chat",
            "messages": messages,
            "tools": tools,
            "temperature": 0.7,
            "max_tokens": 500,
        }

        response2 = requests.post(url, headers=headers, json=body2, timeout=30)
        print(f"Status: {response2.status_code}")

        if response2.status_code == 200:
            data2 = response2.json()
            final_message = data2["choices"][0]["message"]
            print(f"[OK] Final response: {final_message.get('content')}")

            if final_message.get("tool_calls"):
                print("[WARN] Model requested more tool calls (loop detected!)")
            else:
                print("[OK] No more tool calls - conversation complete!")
        else:
            print(f"[ERROR] {response2.text}")
    else:
        print(f"[WARN] No tool calls. Response: {message.get('content')}")
else:
    print(f"[ERROR] {response.text}")

print("\n" + "=" * 60)
print("Tests complete!")
print("=" * 60)
