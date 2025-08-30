"""Simple script to test direct Gemini generateContent REST API call.

Usage (PowerShell):
  $env:GOOGLE_API_KEY="<your_key>"; python test_gemini_api.py
or ensure .env contains GOOGLE_API_KEY and python-dotenv is installed.
"""

from __future__ import annotations
import os
import json
import sys
import textwrap
from typing import Any, Dict

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set in environment or .env file.", file=sys.stderr)
    sys.exit(1)

import requests  # noqa: E402

MODEL = "gemini-2.0-flash"  # endpoint expects short name before colon
ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
)

payload: Dict[str, Any] = {
    "contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]
}

headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}

print(f"POST {ENDPOINT}\n")
resp = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60)
print(f"Status: {resp.status_code}")
try:
    data = resp.json()
except ValueError:
    print(resp.text)
    sys.exit(2)

# Pretty print full response for debugging
print("Raw response JSON:\n" + json.dumps(data, indent=2) + "\n")

# Attempt to extract first text output (Gemini returns candidates list)
text_out = None
if isinstance(data, dict):
    for cand in data.get("candidates", []) or []:
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                text_out = p["text"]
                break
        if text_out:
            break

if text_out:
    print("Extracted answer:\n" + textwrap.fill(text_out, width=88))
else:
    print("Could not extract answer text from response.")
