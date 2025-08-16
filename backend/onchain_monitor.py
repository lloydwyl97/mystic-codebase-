import requests


def fetch_eth_gas_price():
    r = requests.get("https://api.etherscan.io/api?module=gastracker&action=gasoracle")
    result = r.json()["result"]
    return int(result["ProposeGasPrice"])


def fetch_whale_alerts():
    r = requests.get(
        "https://api.whale-alert.io/v1/transactions?api_key=YOUR_KEY&min_value=1000000"
    )
    return r.json().get("transactions", [])


def onchain_signal_check():
    gas = fetch_eth_gas_price()
    whales = fetch_whale_alerts()
    print(f"[ONCHAIN] Gas: {gas} gwei | Whale TXs: {len(whales)}")
    return {"gas": gas, "whales": len(whales)}


