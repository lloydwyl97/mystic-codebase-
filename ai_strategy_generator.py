import requests

# Binance API for live data
response = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10)
