## Smoke Tests for Mystic Platform

What this covers
- Backend endpoints: system health, AI heartbeat/status, advanced-tech performance/events, clear-cache
- Market router: prices (ticker), OHLCV recency, basic orderbook sanity via bid/ask preview, trades endpoint
- Autobuy flow: start, poll heartbeat, stop, poll heartbeat
- Dashboard client: import `dashboard/data_client.py` and call `get_ticker`, `get_ohlcv`, `get_autobuy_heartbeat`

Prereqs
- Windows 11, Python 3.10
- Virtual environment activated
- Backend accessible at BASE_URL (default http://127.0.0.1:9000)
- Live internet connectivity for public market endpoints

Env vars
- COINBASE_*, BINANCEUS_*, KRAKEN_*, COINGECKO_* (optional but recommended)
- REDIS_URL (optional)
- BASE_URL (optional, override default)

Run locally
1) Activate your venv
2) Run:
   `powershell -ExecutionPolicy Bypass -File scripts/run_smoke_tests.ps1 -BaseUrl http://127.0.0.1:9000`

What to expect
- Backend starts and /api/system/health-check returns 200
- Tests exercise endpoints with small backoff and timeouts
- Total run typically under a few minutes depending on rate limits

Common failures
- Missing env keys: some services may degrade; ensure required keys are present for your setup
- Rate limiting: Coin APIs may slow responses; the suite includes retries and sleeps
- Adapter off: If an exchange is disabled, the market tests will select available exchanges automatically


