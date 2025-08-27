#!/usr/bin/env python3
# ==========================================================
# 📣 WEPS Telegram Notifier — Real-Time Spiral Decision Alerts
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Sends alerts to Telegram bot for spiral decisions & trade executions
# ==========================================================

import requests
import logging

# === Telegram Bot Config ===
BOT_TOKEN = "7506200971:AAFrXkyYW5ZFCwY28p8UjJbLlpoZ_TfEMhg"
CHAT_ID = "7356982554"  # Replace with your actual Telegram user or group chat ID

logger = logging.getLogger("WEPS.TelegramNotifier")
logging.basicConfig(level=logging.INFO)

def send_telegram_alert(message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info("✅ Telegram alert sent successfully.")
            return True
        else:
            logger.warning(f"⚠️ Telegram failed: {response.status_code} — {response.text}")
            return False
    except Exception as e:
        logger.error(f"🔥 Telegram Exception: {e}")
        return False
