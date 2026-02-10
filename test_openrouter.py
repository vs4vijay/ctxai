#!/usr/bin/env python3
"""
Test script to verify OpenRouter API connectivity and functionality.
"""

import os
import sys
import json
import requests
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def get_api_key():
    """Get OpenRouter API key from environment or keystore."""
    # Try environment variable first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Try keystore (keys.json)
    keystore_path = Path.home() / ".ctxai" / "keys.json"
    if keystore_path.exists():
        with open(keystore_path) as f:
            keystore = json.load(f)
            # Get the openrouter value
            openrouter_val = keystore.get("openrouter")
            if isinstance(openrouter_val, str):
                # Direct API key
                return openrouter_val
            elif isinstance(openrouter_val, dict):
                # Nested format
                return openrouter_val.get("api_key")
            return None

    return None


def test_simple_chat(api_key):
    """Test a simple chat completion."""
    print("\n=== Testing Simple Chat ===")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ctxai",
        "X-Title": "ctxai",
        "Content-Type": "application/json",
    }

    body = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "user", "content": "Say 'Hello!' and nothing else."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    print(f"Request URL: {url}")
    print(f"Model: {body['model']}")
    print(f"Message: {body['messages'][0]['content']}")

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"[OK] Response: {content}")
            return True
        else:
            print(f"[ERROR] Error: {response.text}")
            return False

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False


def test_function_calling(api_key):
    """Test function calling with tools."""
    print("\n=== Testing Function Calling ===")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ctxai",
        "X-Title": "ctxai",
        "Content-Type": "application/json",
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    body = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 500,
    }

    print(f"Request URL: {url}")
    print(f"Model: {body['model']}")
    print(f"Message: {body['messages'][0]['content']}")
    print(f"Tools: {len(tools)} tool(s) provided")

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            message = data["choices"][0]["message"]

            print(f"Content: {message.get('content', '(empty)')}")

            if message.get("tool_calls"):
                print(f"[OK] Tool calls detected: {len(message['tool_calls'])}")
                for tc in message["tool_calls"]:
                    print(f"  - Function: {tc['function']['name']}")
                    print(f"    Arguments: {tc['function']['arguments']}")
                return True
            else:
                print("[WARN] No tool calls (model may not support function calling)")
                print(f"Response content: {message.get('content')}")
                return False

        else:
            print(f"[ERROR] Error: {response.text}")
            return False

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False


def test_with_tool_result(api_key):
    """Test a complete tool calling cycle with result."""
    print("\n=== Testing Tool Call + Result Cycle ===")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ctxai",
        "X-Title": "ctxai",
        "Content-Type": "application/json",
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Step 1: Initial request
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]

    body = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 500,
    }

    print("Step 1: Initial request")
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)

        if response.status_code != 200:
            print(f"[ERROR] Error: {response.text}")
            return False

        data = response.json()
        message = data["choices"][0]["message"]

        if not message.get("tool_calls"):
            print("[WARN] Model did not request tool call")
            return False

        tool_call = message["tool_calls"][0]
        print(f"[OK] Model requested tool: {tool_call['function']['name']}")

        # Step 2: Provide tool result
        messages.append({
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message["tool_calls"]
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": tool_call["function"]["name"],
            "content": "The weather in Paris is sunny, 22°C"
        })

        print("Step 2: Sending tool result back")
        body["messages"] = messages

        response2 = requests.post(url, headers=headers, json=body, timeout=30)

        if response2.status_code != 200:
            print(f"[ERROR] Error on second request: {response2.text}")
            return False

        data2 = response2.json()
        final_message = data2["choices"][0]["message"]

        print(f"[OK] Final response: {final_message.get('content', '(empty)')}")

        # Check if it has more tool calls
        if final_message.get("tool_calls"):
            print("[WARN] Model requested more tool calls (potential loop)")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False


def main():
    print("=" * 60)
    print("OpenRouter API Test Script")
    print("=" * 60)

    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("\n[ERROR] No API key found!")
        print("Run: ctxai login openrouter")
        print("Or set: OPENROUTER_API_KEY environment variable")
        return

    print(f"[OK] API key found: {api_key[:20]}...")

    # Run tests
    results = []

    results.append(("Simple Chat", test_simple_chat(api_key)))
    results.append(("Function Calling", test_function_calling(api_key)))
    results.append(("Tool Result Cycle", test_with_tool_result(api_key)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "[OK] PASS" if passed else "[ERROR] FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n[OK] All tests passed! OpenRouter API is working correctly.")
    else:
        print("\n[WARN] Some tests failed. Check the output above for details.")
        print("\nPossible issues:")
        print("- Model doesn't support function calling")
        print("- API key has insufficient credits")
        print("- Network connectivity issues")
        print("- Rate limiting")


if __name__ == "__main__":
    main()
