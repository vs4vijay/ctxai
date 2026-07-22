#!/usr/bin/env python3
"""Check available OpenRouter models."""

import json
import sys
from pathlib import Path

import requests

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def get_api_key():
    """Get OpenRouter API key."""
    keystore_path = Path.home() / ".ctxai" / "keys.json"
    if keystore_path.exists():
        with open(keystore_path) as f:
            keystore = json.load(f)
            openrouter_val = keystore.get("openrouter")
            if isinstance(openrouter_val, str):
                return openrouter_val
            elif isinstance(openrouter_val, dict):
                return openrouter_val.get("api_key")
    return None


api_key = get_api_key()
if not api_key:
    print("[ERROR] No API key found!")
    sys.exit(1)

# Fetch available models
url = "https://openrouter.ai/api/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}",
}

print("Fetching available models from OpenRouter...")
response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"[ERROR] Failed to fetch models: {response.text}")
    sys.exit(1)

data = response.json()
models = data.get("data", [])

print(f"\nFound {len(models)} models")

# Filter for free models with function calling support
print("\n" + "=" * 80)
print("FREE MODELS WITH FUNCTION CALLING:")
print("=" * 80)

free_models_with_tools = []

for model in models:
    model_id = model.get("id", "")
    pricing = model.get("pricing", {})
    prompt_price = float(pricing.get("prompt", "1"))
    completion_price = float(pricing.get("completion", "1"))

    # Check if free
    is_free = prompt_price == 0 and completion_price == 0

    # Check if supports tools
    architecture = model.get("architecture", {})
    supports_tools = architecture.get("modality") == "text->text"

    if is_free and "free" in model_id.lower():
        name = model.get("name", model_id)
        context_length = model.get("context_length", 0)

        print(f"\n  Model ID: {model_id}")
        print(f"  Name: {name}")
        print(f"  Context: {context_length:,} tokens")

        free_models_with_tools.append(model_id)

print("\n" + "=" * 80)
print("POPULAR MODELS (may have cost):")
print("=" * 80)

for model in models:
    model_id = model.get("id", "")

    if any(x in model_id.lower() for x in ["gpt-4", "claude", "gemini", "deepseek-chat"]):
        name = model.get("name", model_id)
        pricing = model.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0"))

        print(f"\n  Model ID: {model_id}")
        print(f"  Name: {name}")
        print(f"  Cost: ${prompt_price:.6f} per 1K tokens")

print("\n" + "=" * 80)
print(f"Total free models found: {len(free_models_with_tools)}")
print("=" * 80)
