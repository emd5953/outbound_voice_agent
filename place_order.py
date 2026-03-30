#!/usr/bin/env python3
"""Post test_payload.json to the running server's /place-order endpoint."""

import json
import sys

import httpx

PAYLOAD_FILE = "test_payload.json"
DEFAULT_URL = "http://localhost:8000/place-order"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    with open(PAYLOAD_FILE) as f:
        payload = json.load(f)

    print(f"Posting {PAYLOAD_FILE} to {url} ...")
    print(f"Customer: {payload.get('customer_name')}")
    print(f"Phone:    {payload.get('phone_number')}")
    print(f"Pizza:    {payload.get('pizza', {}).get('size')} {payload.get('pizza', {}).get('crust')} "
          f"w/ {', '.join(payload.get('pizza', {}).get('toppings', []))}")
    print()

    try:
        resp = httpx.post(url, json=payload, timeout=300)
    except httpx.ConnectError:
        print(f"Could not connect to {url} — is the server running?")
        print("Start it with:  uvicorn app.server:app --reload")
        sys.exit(1)

    if resp.status_code == 200:
        result = resp.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
