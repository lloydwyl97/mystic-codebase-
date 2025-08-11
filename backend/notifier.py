import os

import requests

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_alert(message: str):
    if DISCORD_WEBHOOK:
        try:
            requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
        except Exception as e:
            print(f"[Notifier] Discord error: {e}")

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            print(f"[Notifier] Telegram error: {e}")


def send_trade_alert(symbol: str, action: str, price: float, profit: float):
    message = f"🤖 AI Trade: {action} {symbol} @ ${price:.2f} | Profit: ${profit:.2f}"
    send_alert(message)


def send_performance_alert(avg_profit: float, total_trades: int):
    message = f"📊 AI Performance: {total_trades} trades | Avg Profit: ${avg_profit:.2f}"
    send_alert(message)
