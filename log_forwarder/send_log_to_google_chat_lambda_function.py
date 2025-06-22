import json
import urllib.request
import os

WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK")


def lambda_handler(event, context):
    for record in event.get("records", []):
        message = record.get("message", "")

        # Skip empty or system messages
        if not message.strip():
            continue

        level = get_log_level(message)
        card_payload = build_google_chat_card(message, level)
        send_to_google_chat(card_payload)

    return {"status": "ok"}


def get_log_level(message: str) -> str:
    """Detect log level from the log line"""
    levels = ["ERROR", "WARNING", "INFO", "DEBUG", "TRACEBACK"]
    for lvl in levels:
        if lvl in message.upper():
            return lvl
    return "INFO"


def build_google_chat_card(message: str, level: str):
    """Constructs a Google Chat card payload based on log level"""
    color_map = {
        "ERROR": "#D93025",
        "WARNING": "#F9AB00",
        "INFO": "#1A73E8",
        "DEBUG": "#34A853",
        "TRACEBACK": "#D93025",
    }
    emoji_map = {
        "ERROR": "🚨",
        "WARNING": "⚠️",
        "INFO": "ℹ️",
        "DEBUG": "🐛",
        "TRACEBACK": "💥",
    }

    header_text = f"{emoji_map.get(level, '📘')} HARMONY SEEKER [{level}]"

    return {
        "cards": [
            {
                "header": {
                    "title": header_text,
                    "subtitle": "From CloudWatch Lambda Log Forwarder",
                    "imageUrl": "https://cdn-icons-png.flaticon.com/512/3306/3306623.png",
                    "imageStyle": "AVATAR",
                },
                "sections": [
                    {
                        "widgets": [
                            {
                                "textParagraph": {
                                    "text": f"<b>Log Message:</b><br><pre>{message.strip()}</pre>"
                                }
                            }
                        ]
                    }
                ],
            }
        ]
    }


def send_to_google_chat(payload: dict):
    """Send card to Google Chat"""
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(WEBHOOK_URL, data=data, headers=headers)
    urllib.request.urlopen(req)
