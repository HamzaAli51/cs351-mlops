import os

import requests


def notify(message: str) -> None:
    print(message)
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("(no DISCORD_WEBHOOK_URL set, message printed only)")
        return
    try:
        response = requests.post(url, json={"content": message[:1900]}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Could not send notification: {exc}")