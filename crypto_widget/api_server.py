# api_server.py
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable frontend access from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SHARED_DATA = "shared_data.json"
LOG_FILE = "price_log.csv"

# Track startup time for uptime calculation
startup_time = datetime.now(timezone.utc)


@app.get("/prices")
def get_prices():
    try:
        with open(SHARED_DATA) as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
def get_health() -> dict[str, Any]:
    try:
        last_modified = os.path.getmtime(SHARED_DATA)
        return {
            "status": "ok",
            "last_updated_timezone.utc": (
                datetime.fromtimestamp(last_modified, timezone.utc).isoformat()
            ),
            "log_file_size_bytes": os.path.getsize(LOG_FILE),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/coins")
def get_coins() -> dict[str, list[str]]:
    return {
        "Binance": ["BTC", "ETH", "SOL", "ADA"],
        "Coinbase": ["BTC", "ETH", "SOL", "ADA"],
    }


@app.get("/history")
def get_history(
    symbol: str = Query(..., min_length=3),
    source: str = Query(...),
    limit: int | None = 50,
):
    rows: list[dict[str, Any]] = []
    try:
        with open(LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reversed(list(reader)):  # reverse for newest-first
                if row["symbol"] == symbol and row["source"] == source:
                    rows.append(row)
                    if limit and len(rows) >= limit:
                        break
        return rows
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
