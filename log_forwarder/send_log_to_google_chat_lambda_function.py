import json
import urllib.request
import os

WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK")


def lambda_handler(event, context):
    for record in event.get("records", []):
        message = record.get("message", "")

        if "ERROR" in message or "Traceback" in message:
            send_to_google_chat("🚨 *HARMONY SEEKER ERROR* 🚨", message)

        elif "INFO" in message:
            send_to_google_chat("📘 *HARMONY SEEKER INFO* 📘", message)

    return {"status": "ok"}


def send_to_google_chat(title, message):
    url = WEBHOOK_URL
    headers = {"Content-Type": "application/json"}
    data = {"text": f"{title}\n```{message}```"}
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
    urllib.request.urlopen(req)
