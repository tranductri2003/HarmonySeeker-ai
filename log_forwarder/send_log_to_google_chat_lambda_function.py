import json
import urllib.request
import os
import base64
import gzip
import io

WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK")


def lambda_handler(event, context):
    print("📦 RAW EVENT:")
    print(json.dumps(event))

    compressed_payload = base64.b64decode(event["awslogs"]["data"])
    uncompressed_payload = gzip.GzipFile(fileobj=io.BytesIO(compressed_payload)).read()
    payload = json.loads(uncompressed_payload)

    print("📝 PARSED LOG EVENTS:")
    print(json.dumps(payload))

    for log_event in payload["logEvents"]:
        message = log_event.get("message", "")
        print(f"🔍 Log message: {message}")

        if any(
            level in message
            for level in ["ERROR", "INFO", "DEBUG", "WARNING", "TRACEBACK"]
        ):
            send_to_google_chat(message)

    return {"status": "ok"}


def send_to_google_chat(message):
    headers = {"Content-Type": "application/json"}
    data = {
        "cards": [
            {
                "header": {"title": "🎯 HarmonySeeker Log"},
                "sections": [
                    {
                        "widgets": [
                            {
                                "textParagraph": {
                                    "text": f"<b>Log Message:</b><br><pre>{message}</pre>"
                                }
                            }
                        ]
                    }
                ],
            }
        ]
    }

    req = urllib.request.Request(
        WEBHOOK_URL, data=json.dumps(data).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(req) as resp:
        print(f"✅ Google Chat response: {resp.status}")
