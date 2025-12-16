from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Literal, Optional

import boto3
import httpx
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_combined_signal, generate_rule_based_signal
from src.utils.data_loader import load_price_data
from src.utils.s3_store import (
    cache_bucket,
    cache_key,
    read_latest_signal,
    write_latest_signal,
)

app = FastAPI(title="Intellpulse API", version="0.2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CloudWatch Metrics
# -------------------------
_cw = boto3.client("cloudwatch")
METRICS_NS = os.getenv("METRICS_NAMESPACE", "Intellpulse/MVP1")
SERVICE_NAME = os.getenv(
    "SERVICE_NAME",
    os.getenv("AWS_LAMBDA_FUNCTION_NAME", "intellpulse-api"),
)

def _emit_metric(name: str, value: float = 1.0, unit: str = "Count", **dims) -> None:
    try:
        dimensions = [{"Name": "Service", "Value": SERVICE_NAME}]
        for k, v in dims.items():
            if v is not None:
                dimensions.append({"Name": str(k), "Value": str(v)})

        _cw.put_metric_data(
            Namespace=METRICS_NS,
            MetricData=[{
                "MetricName": name,
                "Value": float(value),
                "Unit": unit,
                "Dimensions": dimensions,
            }],
        )
    except Exception as e:
        print(f"METRICS_WARN — {e}")

# -------------------------
# API Key protection
# -------------------------
PUBLIC_PATHS = {"/health"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    api_key = os.getenv("API_KEY")
    if not api_key:
        return await call_next(request)

    if request.headers.get("x-api-key") != api_key:
        return JSONResponse({"detail": "Invalid API key"}, status_code=401)

    return await call_next(request)

# -------------------------
# Helpers
# -------------------------
def _signal_to_text(sig: int) -> str:
    return "BUY" if sig > 0 else "SELL" if sig < 0 else "HOLD"

def _cache_ttl_seconds() -> int:
    return int(os.getenv("SIGNAL_CACHE_TTL_SECONDS", "900"))

def _age_seconds(iso: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None

def _is_fresh(iso: str, ttl: int) -> bool:
    age = _age_seconds(iso)
    return age is not None and age <= ttl

def _load_price_pipeline(asset: str):
    """
    Load price data using the correct loader signature.
    If missing, return 404 (never 500).
    """
    symbol_filter = asset.replace("-", "_")  # BTC-USD -> BTC_USD
    try:
        price = load_price_data(symbol_filter=symbol_filter)
    except FileNotFoundError as e:
        _emit_metric("PriceDataMissing", 1, asset=asset)
        return JSONResponse(
            status_code=404,
            content={"detail": f"Price data not found for {asset} ({symbol_filter}). {str(e)}"},
        )

    feat = build_price_feature_set(price)
    return generate_rule_based_signal(feat)

# -------------------------
# Sentiment
# -------------------------
FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"

def _get_fear_greed_score() -> float:
    try:
        r = httpx.get(FNG_URL, timeout=5.0)
        r.raise_for_status()
        v = float(r.json()["data"][0]["value"])
        return max(0.0, min(1.0, v / 100.0))
    except Exception:
        return 0.5

def _load_aligned_sentiment(asset: str, price_df):
    return pd.DataFrame(
        index=price_df.index,
        data={"sentiment_score": _get_fear_greed_score()},
    )

# -------------------------
# Schemas
# -------------------------
class SignalResponse(BaseModel):
    asset: str
    mode: str
    latest_timestamp: str
    latest_signal: int
    latest_signal_text: Literal["BUY", "HOLD", "SELL"]
    latest_sentiment: Optional[float] = None
    cached_at_utc: Optional[str] = None

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/signal", response_model=SignalResponse)
def get_signal(asset: str = "BTC-USD", mode: Literal["price_only", "combined"] = "combined"):
    t0 = time.time()
    _emit_metric("SignalRequest", 1, asset=asset, mode=mode)

    ttl = _cache_ttl_seconds()

    # Cache read (best effort)
    try:
        bucket = cache_bucket()
        cached = read_latest_signal(asset, mode)
    except Exception:
        bucket = None
        cached = None

    if cached and _is_fresh(cached.get("cached_at", ""), ttl):
        _emit_metric("CacheHit", 1, asset=asset, mode=mode)
        _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000, unit="Milliseconds", asset=asset, mode=mode)
        return SignalResponse(
            asset=asset,
            mode=mode,
            latest_timestamp=cached["latest_timestamp"],
            latest_signal=int(cached["latest_signal"]),
            latest_signal_text=_signal_to_text(int(cached["latest_signal"])),
            latest_sentiment=cached.get("latest_sentiment"),
            cached_at_utc=cached.get("cached_at"),
        )

    _emit_metric("CacheMiss", 1, asset=asset, mode=mode)

    # Compute (and handle missing price data cleanly)
    price_sig = _load_price_pipeline(asset)
    if isinstance(price_sig, JSONResponse):
        return price_sig  # 404 response

    latest_ts = price_sig.index[-1]
    latest_signal = int(price_sig["signal"].iloc[-1])
    latest_sentiment: Optional[float] = None

    if mode == "combined":
        sent = _load_aligned_sentiment(asset, price_sig)
        combined = generate_combined_signal(price_sig, sent)
        latest_signal = int(combined["signal_combined"].iloc[-1])
        latest_sentiment = float(combined["sentiment_score"].iloc[-1])

    cached_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    resp = SignalResponse(
        asset=asset,
        mode=mode,
        latest_timestamp=latest_ts.isoformat(),
        latest_signal=latest_signal,
        latest_signal_text=_signal_to_text(latest_signal),
        latest_sentiment=latest_sentiment,
        cached_at_utc=cached_at,
    )

    # Cache write (best effort)
    if bucket:
        payload = {
            "asset": resp.asset,
            "mode": resp.mode,
            "latest_timestamp": resp.latest_timestamp,
            "latest_signal": resp.latest_signal,
            "latest_sentiment": resp.latest_sentiment,
            "cached_at": cached_at,  # IMPORTANT: reader expects cached_at
        }
        try:
            write_latest_signal(asset, mode, payload)
            _emit_metric("CacheWrite", 1, asset=asset, mode=mode)
        except Exception:
            _emit_metric("CacheWriteError", 1, asset=asset, mode=mode)

    _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000, unit="Milliseconds", asset=asset, mode=mode)
    return resp

@app.get("/debug/cache/read")
def debug_cache_read(asset: str = "BTC-USD", mode: str = "combined"):
    try:
        bucket = cache_bucket()
    except Exception:
        return {"enabled": False, "reason": "SIGNALS_BUCKET not set"}

    key = cache_key(asset, mode)
    ttl = _cache_ttl_seconds()
    cached = read_latest_signal(asset, mode)

    if not cached:
        return {
            "enabled": True,
            "found": False,
            "bucket": bucket,
            "key": key,
            "ttl_seconds": ttl,
        }

    age = _age_seconds(cached.get("cached_at", ""))

    return {
        "enabled": True,
        "found": True,
        "bucket": bucket,
        "key": key,
        "ttl_seconds": ttl,
        "cached_at": cached.get("cached_at"),
        "age_seconds": age,
        "fresh": age is not None and age <= ttl,
        "payload": cached,
    }

handler = Mangum(app)
