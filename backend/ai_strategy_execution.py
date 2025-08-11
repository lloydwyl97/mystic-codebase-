import hashlib
import hmac
import json
import logging
import os
import time
from base64 import b64encode
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv  # type: ignore
from config import settings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_execution.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_strategy_execution")

BINANCE_KEY = settings.exchange.binance_us_api_key
BINANCE_SECRET = settings.exchange.binance_us_secret_key
COINBASE_KEY = settings.exchange.coinbase_api_key
COINBASE_SECRET = os.getenv("COINBASE_API_SECRET")
COINBASE_PASSPHRASE = os.getenv("COINBASE_PASSPHRASE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send_alert(msg: str) -> None:
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            )
        if DISCORD_WEBHOOK_URL and DISCORD_WEBHOOK_URL != "your_discord_webhook_url_here":
            requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        logger.info(f"Alert sent: {msg}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


def get_binance_price(symbol: str = "ETHUSDT") -> Optional[float]:
    try:
        url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        logger.error(f"Failed to get Binance US price for {symbol}: {e}")
        return None


def get_coinbase_price(product_id: str = "ETH-USD") -> Optional[float]:
    try:
        url = f"https://api.coinbase.com/v2/prices/{product_id}/spot"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return float(response.json()["data"]["amount"])
    except Exception as e:
        logger.error(f"Failed to get Coinbase price for {product_id}: {e}")
        return None


def binance_market_buy(symbol: str = "ETHUSDT", quoteOrderQty: float = 50) -> dict[str, Any]:
    try:
        base_url = "https://api.binance.us"
        endpoint = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": quoteOrderQty,
            "timestamp": timestamp,
        }
        query = urlencode(params)
        if not BINANCE_SECRET:
            logger.error("BINANCE_SECRET is not configured")
            return {"error": "BINANCE_SECRET not configured"}
        signature = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{base_url}{endpoint}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": BINANCE_KEY}
        res = requests.post(url, headers=headers, timeout=30)
        res.raise_for_status()
        result = res.json()
        send_alert(f"SUCCESS: Binance US Buy Executed: {symbol} | ${quoteOrderQty}\n{result}")
        logger.info(f"Binance US buy executed: {symbol} ${quoteOrderQty}")
        return result
    except Exception as e:
        error_msg = f"ERROR: Binance US Buy Error: {e}"
        send_alert(error_msg)
        logger.error(f"Binance US buy failed: {e}")
        return {"error": str(e)}


def coinbase_market_buy(product_id: str = "ETH-USD", funds: str = "50") -> dict[str, Any]:
    try:
        base_url = "https://api.coinbase.com"
        url_path = "/orders"
        timestamp = str(time.time())
        body = {
            "type": "market",
            "side": "buy",
            "product_id": product_id,
            "funds": funds,
        }
        body_str = json.dumps(body)
        message = f"{timestamp}POST{url_path}{body_str}"
        if not COINBASE_SECRET:
            logger.error("COINBASE_SECRET is not configured")
            return {"error": "COINBASE_SECRET not configured"}
        signature = hmac.new(
            b64encode(COINBASE_SECRET.encode()),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        sig_b64 = b64encode(signature.digest()).decode()
        headers = {
            "CB-ACCESS-KEY": COINBASE_KEY,
            "CB-ACCESS-SIGN": sig_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": COINBASE_PASSPHRASE,
            "Content-Type": "application/json",
        }
        response = requests.post(base_url + url_path, headers=headers, data=body_str, timeout=30)
        response.raise_for_status()
        result = response.json()
        send_alert(f"SUCCESS: Coinbase Buy Executed: {product_id} | ${funds}\n{result}")
        logger.info(f"Coinbase buy executed: {product_id} ${funds}")
        return result
    except Exception as e:
        error_msg = f"ERROR: Coinbase Buy Error: {e}"
        send_alert(error_msg)
        logger.error(f"Coinbase buy failed: {e}")
        return {"error": str(e)}


def execute_ai_strategy_signal(
    symbol_binance: str, symbol_coinbase: str, usd_amount: float, signal: bool
) -> Optional[dict[str, Any]]:
    try:
        if not signal:
            logger.info("No signal provided, skipping execution")
            return None

        logger.info(
            f"Executing AI strategy signal: {symbol_binance}/{symbol_coinbase} ${usd_amount}"
        )

        b_price = get_binance_price(symbol_binance)
        c_price = get_coinbase_price(symbol_coinbase)

        if b_price is None or c_price is None:
            logger.error("Failed to get prices from exchanges")
            return None

        if b_price < c_price:
            logger.info(f"Binance price ${b_price} < Coinbase price ${c_price}, using Binance")
            return binance_market_buy(symbol_binance, usd_amount)
        else:
            logger.info(f"Coinbase price ${c_price} <= Binance price ${b_price}, using Coinbase")
            return coinbase_market_buy(symbol_coinbase, str(usd_amount))
    except Exception as e:
        error_msg = f"ERROR: AI Strategy Buy Error: {e}"
        send_alert(error_msg)
        logger.error(f"AI strategy execution failed: {e}")
        return {"error": str(e)}


def test_connections():
    """Test API connections and permissions"""
    logger.info("Testing exchange connections...")

    # Test Binance
    try:
        b_price = get_binance_price("ETHUSDT")
        if b_price:
            logger.info(f"SUCCESS: Binance connection successful - ETH price: ${b_price}")
        else:
            logger.error("FAILED: Binance connection failed")
    except Exception as e:
        logger.error(f"FAILED: Binance connection error: {e}")

    # Test Coinbase
    try:
        c_price = get_coinbase_price("ETH-USD")
        if c_price:
            logger.info(f"SUCCESS: Coinbase connection successful - ETH price: ${c_price}")
        else:
            logger.error("FAILED: Coinbase connection failed")
    except Exception as e:
        logger.error(f"FAILED: Coinbase connection error: {e}")

    # Test notifications
    send_alert("AI Strategy Execution System - Connection Test")


if __name__ == "__main__":
    test_connections()
