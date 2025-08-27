#!/usr/bin/env python3
# =============================================================
# 📡 WEPS Telegram Trade Alert Bot
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Sends formatted trade alerts to Telegram via bot API
# =============================================================

import requests
import logging

# ✅ Configure Your Bot Token and Chat ID
TELEGRAM_BOT_TOKEN = "7506200971:AAFrXkyYW5ZFCwY28p8UjJbLlpoZ_TfEMhg"
TELEGRAM_CHAT_ID = "7356982554"

logger = logging.getLogger("WEPS.TelegramBot")

def send_trade_alert(symbol: str, action: str, confidence: float):
    try:
        message = (
            f"📈 *WEPS TRADE ALERT*\n"
            f"• Organism: `{symbol}`\n"
            f"• Action: *{action.upper()}*\n"
            f"• Confidence: `{confidence:.2f}`\n"
            f"• Timestamp: `{__utc_now()}`"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logger.error(f"Telegram send failed: {response.text}")
        else:
            logger.info(f"📨 Telegram alert sent: {action} on {symbol}")
    except Exception as e:
        logger.error(f"Telegram send error: {str(e)}")


def __utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
