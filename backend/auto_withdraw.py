#!/usr/bin/env python3
"""
Mystic Auto-Withdraw System
Automatically withdraws funds to cold wallet when threshold is reached
Supports Binance and Coinbase with notifications and logging
"""

import hashlib
import hmac
import json
import logging
import os
import time
from base64 import b64encode
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from backend.config import settings

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/auto_withdraw.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("auto_withdraw")


class AutoWithdrawSystem:
    """Unified auto-withdraw system for multiple exchanges"""

    def __init__(self):
        self.exchange = os.getenv("EXCHANGE", "binance").lower()
        self.cold_wallet_address = os.getenv("COLD_WALLET_ADDRESS")
        self.cold_wallet_threshold = float(os.getenv("COLD_WALLET_THRESHOLD", 250.00))
        self.check_interval = int(os.getenv("CHECK_INTERVAL", 60))  # seconds

        # API Keys
        self.binance_api_key = settings.exchange.binance_us_api_key
        self.binance_api_secret = settings.exchange.binance_us_secret_key
        self.coinbase_api_key = settings.exchange.coinbase_api_key
        self.coinbase_api_secret = os.getenv("COINBASE_API_SECRET")
        self.coinbase_passphrase = os.getenv("COINBASE_PASSPHRASE")

        # Notification settings
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK")
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Validate configuration
        self._validate_config()

        # Statistics
        self.stats = {
            "total_withdrawals": 0,
            "total_amount_withdrawn": 0.0,
            "last_withdrawal": None,
            "last_check": None,
        }

        logger.info(f"Auto-withdraw system initialized for {self.exchange}")
        logger.info(f"Cold wallet threshold: ${self.cold_wallet_threshold}")
        logger.info(f"Check interval: {self.check_interval} seconds")

    def _validate_config(self):
        """Validate required configuration"""
        if not self.cold_wallet_address:
            raise ValueError("COLD_WALLET_ADDRESS not configured")

        if self.exchange == "binance":
            if not self.binance_api_key or not self.binance_api_secret:
                raise ValueError("Binance API keys not configured")
        elif self.exchange == "coinbase":
            if (
                not self.coinbase_api_key
                or not self.coinbase_api_secret
                or not self.coinbase_passphrase
            ):
                raise ValueError("Coinbase API keys not configured")
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange}")

    def _send_notification(self, message: str, level: str = "INFO"):
        """Send notification via Discord and/or Telegram"""
        try:
            # Discord notification
            if self.discord_webhook:
                discord_payload = {
                    "content": f"ðŸ” **Mystic Auto-Withdraw**\n{message}",
                    "username": "Mystic Trading Bot",
                }
                requests.post(self.discord_webhook, json=discord_payload, timeout=10)

            # Telegram notification
            if self.telegram_token and self.telegram_chat_id:
                telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                telegram_payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"ðŸ” Mystic Auto-Withdraw\n{message}",
                    "parse_mode": "HTML",
                }
                requests.post(telegram_url, json=telegram_payload, timeout=10)

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def _log_withdrawal(
        self,
        amount: float,
        exchange: str,
        status: str,
        details: Dict[str, Any],
    ):
        """Log withdrawal to database and file"""
        try:
            # Log to file
            log_entry = {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "exchange": exchange,
                "amount": amount,
                "status": status,
                "details": details,
                "cold_wallet_address": self.cold_wallet_address[:10] + "...",
            }

            with open("logs/withdrawals.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Update statistics
            if status == "success":
                self.stats["total_withdrawals"] += 1
                self.stats["total_amount_withdrawn"] += amount
                self.stats["last_withdrawal"] = datetime.timezone.utcnow().isoformat()

            self.stats["last_check"] = datetime.timezone.utcnow().isoformat()

        except Exception as e:
            logger.error(f"Failed to log withdrawal: {e}")

    def binance_withdraw(self) -> Dict[str, Any]:
        """Handle Binance withdrawal"""
        try:
            base_url = "https://api.binance.us"

            # Get account balance
            timestamp = int(time.time() * 1000)
            query = f"timestamp={timestamp}"
            signature = hmac.new(
                self.binance_api_secret.encode(),
                query.encode(),
                hashlib.sha256,
            ).hexdigest()

            url = f"{base_url}/api/v3/account?{query}&signature={signature}"
            headers = {"X-MBX-APIKEY": self.binance_api_key}

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Find USDT balance
            usdt_balance = 0.0
            for balance in data.get("balances", []):
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break

            logger.info(f"[BINANCE] Current USDT balance: ${usdt_balance:.2f}")

            if usdt_balance <= self.cold_wallet_threshold:
                logger.info(
                    f"[BINANCE] Balance ${usdt_balance:.2f} below threshold ${self.cold_wallet_threshold}"
                )
                return {"status": "below_threshold", "balance": usdt_balance}

            # Calculate withdrawal amount (leave threshold amount)
            withdrawal_amount = round(usdt_balance - self.cold_wallet_threshold, 2)

            # Execute withdrawal
            timestamp = int(time.time() * 1000)
            params = {
                "asset": "USDT",
                "address": self.cold_wallet_address,
                "amount": withdrawal_amount,
                "network": "ETH",  # Default to ETH network
                "timestamp": timestamp,
            }

            query_string = urlencode(params)
            signature = hmac.new(
                self.binance_api_secret.encode(),
                query_string.encode(),
                hashlib.sha256,
            ).hexdigest()

            url = f"{base_url}/sapi/v1/capital/withdraw/apply?{query_string}&signature={signature}"
            headers = {"X-MBX-APIKEY": self.binance_api_key}

            response = requests.post(url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("id"):
                message = f"âœ… Binance withdrawal successful!\nAmount: ${withdrawal_amount:.2f}\nTransaction ID: {result['id']}"
                self._send_notification(message, "SUCCESS")
                self._log_withdrawal(withdrawal_amount, "binance", "success", result)
                logger.info(f"[BINANCE] Withdrawal successful: ${withdrawal_amount:.2f}")
                return {
                    "status": "success",
                    "amount": withdrawal_amount,
                    "tx_id": result["id"],
                }
            else:
                message = f"âŒ Binance withdrawal failed: {result}"
                self._send_notification(message, "ERROR")
                self._log_withdrawal(withdrawal_amount, "binance", "failed", result)
                logger.error(f"[BINANCE] Withdrawal failed: {result}")
                return {"status": "failed", "error": result}

        except Exception as e:
            error_msg = f"Binance withdrawal error: {str(e)}"
            self._send_notification(error_msg, "ERROR")
            logger.error(f"[BINANCE] {error_msg}")
            return {"status": "error", "error": str(e)}

    def coinbase_withdraw(self) -> Dict[str, Any]:
        """Handle Coinbase withdrawal"""
        try:
            base_url = "https://api.coinbase.com"

            def get_headers(method: str, request_path: str, body: str = "") -> Dict[str, str]:
                """Generate Coinbase API headers"""
                timestamp = str(time.time())
                message = timestamp + method + request_path + body
                secret_decoded = b64encode(self.coinbase_api_secret.encode())
                signature = hmac.new(secret_decoded, message.encode("utf-8"), hashlib.sha256)
                sig_b64 = b64encode(signature.digest()).decode()

                return {
                    "CB-ACCESS-KEY": self.coinbase_api_key,
                    "CB-ACCESS-SIGN": sig_b64,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "CB-ACCESS-PASSPHRASE": self.coinbase_passphrase,
                    "Content-Type": "application/json",
                }

            # Get account balance
            url = "/accounts"
            headers = get_headers("GET", url)
            response = requests.get(base_url + url, headers=headers, timeout=30)
            response.raise_for_status()
            accounts = response.json()

            # Find USDT balance
            usdt_balance = 0.0
            for account in accounts:
                if account["currency"] == "USDT":
                    usdt_balance = float(account["available"])
                    break

            logger.info(f"[COINBASE] Current USDT balance: ${usdt_balance:.2f}")

            if usdt_balance <= self.cold_wallet_threshold:
                logger.info(
                    f"[COINBASE] Balance ${usdt_balance:.2f} below threshold ${self.cold_wallet_threshold}"
                )
                return {"status": "below_threshold", "balance": usdt_balance}

            # Calculate withdrawal amount
            withdrawal_amount = round(usdt_balance - self.cold_wallet_threshold, 2)

            # Execute withdrawal
            url = "/withdrawals/crypto"
            body = {
                "amount": str(withdrawal_amount),
                "currency": "USDT",
                "crypto_address": self.cold_wallet_address,
            }
            body_str = json.dumps(body)

            headers = get_headers("POST", url, body_str)
            response = requests.post(base_url + url, headers=headers, data=body_str, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("id"):
                message = f"âœ… Coinbase withdrawal successful!\nAmount: ${withdrawal_amount:.2f}\nTransaction ID: {result['id']}"
                self._send_notification(message, "SUCCESS")
                self._log_withdrawal(withdrawal_amount, "coinbase", "success", result)
                logger.info(f"[COINBASE] Withdrawal successful: ${withdrawal_amount:.2f}")
                return {
                    "status": "success",
                    "amount": withdrawal_amount,
                    "tx_id": result["id"],
                }
            else:
                message = f"âŒ Coinbase withdrawal failed: {result}"
                self._send_notification(message, "ERROR")
                self._log_withdrawal(withdrawal_amount, "coinbase", "failed", result)
                logger.error(f"[COINBASE] Withdrawal failed: {result}")
                return {"status": "failed", "error": result}

        except Exception as e:
            error_msg = f"Coinbase withdrawal error: {str(e)}"
            self._send_notification(error_msg, "ERROR")
            logger.error(f"[COINBASE] {error_msg}")
            return {"status": "error", "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get withdrawal statistics"""
        return {
            "exchange": self.exchange,
            "cold_wallet_address": self.cold_wallet_address[:10] + "...",
            "threshold": self.cold_wallet_threshold,
            "check_interval": self.check_interval,
            "statistics": self.stats,
        }

    def run_once(self) -> Dict[str, Any]:
        """Run withdrawal check once"""
        logger.info(f"Checking {self.exchange} for withdrawal opportunity...")

        if self.exchange == "binance":
            return self.binance_withdraw()
        elif self.exchange == "coinbase":
            return self.coinbase_withdraw()
        else:
            error_msg = f"Unsupported exchange: {self.exchange}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    def run_continuous(self):
        """Run continuous withdrawal monitoring"""
        logger.info("Starting continuous auto-withdraw monitoring...")

        while True:
            try:
                result = self.run_once()

                if result.get("status") == "success":
                    logger.info(f"Withdrawal completed: {result}")
                elif result.get("status") == "below_threshold":
                    logger.info(f"Balance below threshold: {result}")
                else:
                    logger.warning(f"Withdrawal check result: {result}")

                # Wait before next check
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Auto-withdraw monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous monitoring: {e}")
                time.sleep(self.check_interval)


def main():
    """Main entry point"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Initialize auto-withdraw system
        auto_withdraw = AutoWithdrawSystem()

        # Run continuous monitoring
        auto_withdraw.run_continuous()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()


