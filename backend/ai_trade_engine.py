import hashlib
import hmac
import json
import logging
import os
import time
from base64 import b64encode
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

from backend.config import settings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_trade_engine.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_trade_engine")

BINANCE_KEY = settings.exchange.binance_us_api_key
BINANCE_SECRET = settings.exchange.binance_us_secret_key
COINBASE_KEY = settings.exchange.coinbase_api_key
COINBASE_SECRET = os.getenv("COINBASE_API_SECRET")
COINBASE_PASSPHRASE = os.getenv("COINBASE_PASSPHRASE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
TP_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.15))
TS_PERCENT = float(os.getenv("TRAILING_STOP_PERCENTAGE", 0.03))
BINANCE_SYMBOL = os.getenv("SYMBOL_PAIR_BINANCE", "ETHUSDT")
COINBASE_SYMBOL = os.getenv("SYMBOL_PAIR_COINBASE", "ETH-USD")
USD_AMOUNT = float(os.getenv("USD_TRADE_AMOUNT", 50))

position = None


def save_position_file(data):
    with open("position.json", "w") as f:
        json.dump(data, f, indent=2)


def update_position(current_price):
    global position
    if position:
        position["current_price"] = current_price
        save_position_file(position)
    else:
        save_position_file({})


def send_alert(msg):
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


def get_price_binance(symbol=BINANCE_SYMBOL):
    try:
        r = requests.get(
            f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}",
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.error(f"Failed to get Binance price for {symbol}: {e}")
        return None


def get_price_coinbase(product_id=COINBASE_SYMBOL):
    try:
        r = requests.get(f"https://api.coinbase.com/v2/prices/{product_id}/spot", timeout=10)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception as e:
        logger.error(f"Failed to get Coinbase price for {product_id}: {e}")
        return None


def binance_market_order(symbol, side, quote_qty):
    try:
        url = "https://api.binance.us/api/v3/order"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quoteOrderQty": quote_qty,
            "timestamp": timestamp,
        }
        query = urlencode(params)
        signature = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{url}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": BINANCE_KEY}
        res = requests.post(url, headers=headers, timeout=30)
        res.raise_for_status()
        result = res.json()
        send_alert(f"{side} Binance: ${quote_qty}\n{result}")
        logger.info(f"Binance {side} order executed: {symbol} ${quote_qty}")
        return result
    except Exception as e:
        error_msg = f"ERROR: Binance {side} Error: {e}"
        send_alert(error_msg)
        logger.error(f"Binance {side} order failed: {e}")
        return {"error": str(e)}


def coinbase_market_order(product_id, side, funds):
    try:
        url = "/orders"
        base = "https://api.coinbase.com"
        timestamp = str(time.time())
        body = json.dumps(
            {
                "type": "market",
                "side": side,
                "product_id": product_id,
                "funds": str(funds),
            }
        )
        message = f"{timestamp}POST{url}{body}"
        sig = hmac.new(
            b64encode(COINBASE_SECRET.encode()),
            message.encode(),
            hashlib.sha256,
        )
        headers = {
            "CB-ACCESS-KEY": COINBASE_KEY,
            "CB-ACCESS-SIGN": b64encode(sig.digest()).decode(),
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": COINBASE_PASSPHRASE,
            "Content-Type": "application/json",
        }
        res = requests.post(base + url, headers=headers, data=body, timeout=30)
        res.raise_for_status()
        result = res.json()
        send_alert(f"{side} Coinbase: ${funds}\n{result}")
        logger.info(f"Coinbase {side} order executed: {product_id} ${funds}")
        return result
    except Exception as e:
        error_msg = f"ERROR: Coinbase {side} Error: {e}"
        send_alert(error_msg)
        logger.error(f"Coinbase {side} order failed: {e}")
        return {"error": str(e)}


def check_for_buy_signal():
    """Live buy signal logic using RSI and MACD from Binance price data"""
    try:
        import pandas as pd

        # Fetch recent price data from Binance
        url = f"https://api.binance.us/api/v3/klines?symbol={BINANCE_SYMBOL}&interval=1m&limit=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        klines = response.json()
        closes = [float(k[4]) for k in klines]
        df = pd.DataFrame({"close": closes})
        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        # Calculate MACD
        ema_fast = df["close"].ewm(span=12).mean()
        ema_slow = df["close"].ewm(span=26).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        # Buy signal: RSI < 30 and MACD crosses above signal
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        rsi_buy = latest["rsi"] < 30
        macd_cross = prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]
        return rsi_buy and macd_cross
    except Exception as e:
        logger.error(f"Error in live buy signal logic: {e}")
        return False


def get_position_status():
    """Get current position status for dashboard"""
    global position
    if position:
        current_price = (
            get_price_binance() if position["exchange"] == "binance" else get_price_coinbase()
        )
        if current_price:
            entry_price = position["entry_price"]
            peak_price = position["peak_price"]
            current_pnl = ((current_price - entry_price) / entry_price) * 100
            return {
                "active": True,
                "exchange": position["exchange"],
                "entry_price": entry_price,
                "current_price": current_price,
                "peak_price": peak_price,
                "pnl_percent": current_pnl,
                "take_profit_target": entry_price * (1 + TP_PERCENT),
                "trailing_stop": peak_price * (1 - TS_PERCENT),
            }
    return {"active": False}


def trade_loop():
    global position
    logger.info("Starting AI trade engine...")
    logger.info(f"Configuration: {BINANCE_SYMBOL}/{COINBASE_SYMBOL} ${USD_AMOUNT}")
    logger.info(f"Take Profit: {TP_PERCENT*100}%, Trailing Stop: {TS_PERCENT*100}%")

    while True:
        try:
            signal = check_for_buy_signal()
            b_price = get_price_binance()
            c_price = get_price_coinbase()

            if b_price is None or c_price is None:
                logger.warning("Failed to get prices, skipping iteration")
                time.sleep(60)
                update_position(None)
                continue

            if not position and signal:
                logger.info("Buy signal detected, executing order...")
                if b_price < c_price:
                    result = binance_market_order(BINANCE_SYMBOL, "BUY", USD_AMOUNT)
                    if "error" not in result:
                        position = {
                            "entry_price": b_price,
                            "exchange": "binance",
                            "peak_price": b_price,
                        }
                        logger.info(f"Position opened: Binance {BINANCE_SYMBOL} @ ${b_price}")
                else:
                    result = coinbase_market_order(COINBASE_SYMBOL, "buy", USD_AMOUNT)
                    if "error" not in result:
                        position = {
                            "entry_price": c_price,
                            "exchange": "coinbase",
                            "peak_price": c_price,
                        }
                        logger.info(f"Position opened: Coinbase {COINBASE_SYMBOL} @ ${c_price}")

            elif position:
                price = (
                    get_price_binance()
                    if position["exchange"] == "binance"
                    else get_price_coinbase()
                )
                if price is None:
                    logger.warning("Failed to get current price for position")
                    update_position(None)
                    time.sleep(60)
                    continue

                if price > position["peak_price"]:
                    position["peak_price"] = price
                    logger.info(f"New peak price: ${price}")

                # Check take profit
                if price >= position["entry_price"] * (1 + TP_PERCENT):
                    logger.info("Take profit target reached!")
                    send_alert("TAKE PROFIT HIT")
                    if position["exchange"] == "binance":
                        binance_market_order(BINANCE_SYMBOL, "SELL", USD_AMOUNT)
                    else:
                        coinbase_market_order(COINBASE_SYMBOL, "sell", USD_AMOUNT)
                    position = None
                    logger.info("Position closed: Take profit")
                    update_position(None)
                    continue

                # Check trailing stop
                elif price <= position["peak_price"] * (1 - TS_PERCENT):
                    logger.info("Trailing stop triggered!")
                    send_alert("TRAILING STOP HIT")
                    if position["exchange"] == "binance":
                        binance_market_order(BINANCE_SYMBOL, "SELL", USD_AMOUNT)
                    else:
                        coinbase_market_order(COINBASE_SYMBOL, "sell", USD_AMOUNT)
                    position = None
                    logger.info("Position closed: Trailing stop")
                    update_position(None)
                    continue

            # Dashboard output
            status = get_position_status()
            if status["active"]:
                print(
                    f"Position: {status['exchange']} | Entry: ${status['entry_price']:.2f} | "
                    f"Current: ${status['current_price']:.2f} | PnL: {status['pnl_percent']:.2f}% | "
                    f"Peak: ${status['peak_price']:.2f}"
                )
            else:
                print(f"No Position | Binance: ${b_price:.2f} | Coinbase: ${c_price:.2f}")

            # Update position file for dashboard
            if position:
                update_position(price)
            else:
                update_position(None)

        except Exception as e:
            error_msg = f"ERROR: Trade loop error: {e}"
            send_alert(error_msg)
            logger.error(f"Trade loop error: {e}")
            update_position(None)

        time.sleep(60)  # Check every minute


def test_connections():
    """Test all connections and configurations"""
    logger.info("Testing AI trade engine connections...")

    # Test price feeds
    b_price = get_price_binance()
    c_price = get_price_coinbase()

    if b_price:
        logger.info(f"SUCCESS: Binance price: ${b_price}")
    else:
        logger.error("FAILED: Binance price feed failed")

    if c_price:
        logger.info(f"SUCCESS: Coinbase price: ${c_price}")
    else:
        logger.error("FAILED: Coinbase price feed failed")

    # Test notifications
    send_alert("AI Trade Engine - Connection Test")

    # Test configuration
    logger.info(f"Configuration: TP={TP_PERCENT*100}%, TS={TS_PERCENT*100}%, Amount=${USD_AMOUNT}")


if __name__ == "__main__":
    test_connections()
    trade_loop()


