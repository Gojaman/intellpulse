from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Literal, Optional, Any, Dict

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
from src.utils.s3_store import read_latest_signal, write_latest_signal

app = FastAPI(title="Intellpulse API", version="0.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev. later lock to your Vercel/Lovable domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CloudWatch Metrics (MVP)
# -------------------------
_cw = boto3.client("cloudwatch")
METRICS_NS = os.getenv("METRICS_NAMESPACE", "Intellpulse/MVP1")
SERVICE_NAME = os.getenv("SERVICE_NAME", os.getenv("AWS_LAMBDA_FUNCTION_NAME", "intellpulse-api"))

def _emit_metric(name: str, value: float = 1.0, unit: str = "Count", **dims) -> None:
    """Best-effort metric emit (never break requests)."""
    try:
        dimensions = [{"Name": "Service", "Value": SERVICE_NAME}]
        for k, v in dims.items():
            if v is None:
                continue
            dimensions.append({"Name": str(k), "Value": str(v)})

        _cw.put_metric_data(
            Namespace=METRICS_NS,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": float(value),
                    "Unit": unit,
                    "Dimensions": dimensions,
                }
            ],
        )
    except Exception as e:
        print(f"METRICS_WARN — put_metric_data failed: {e}")

# -------------------------
# API Key protection (MVP)
# -------------------------
PUBLIC_PATHS = {"/health"}  # keep public

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    # read key at request-time (so env updates don't require a cold start)
    api_key = os.getenv("API_KEY")

    # if API key not configured, allow (dev mode)
    if not api_key:
        return await call_next(request)

    sent = request.headers.get("x-api-key")
    if sent != api_key:
        return JSONResponse({"detail": "Invalid API key"}, status_code=401)

    return await call_next(request)

# -------------------------
# Helpers
# -------------------------
def _signal_to_text(sig: int) -> str:
    if sig > 0:
        return "BUY"
    if sig < 0:
        return "SELL"
    return "HOLD"

def _load_price_pipeline(asset: str):
    symbol_filter = asset.replace("-", "_")  # BTC-USD -> BTC_USD
    price = load_price_data(symbol_filter=symbol_filter)
    feat = build_price_feature_set(price)
    price_sig = generate_rule_based_signal(feat)
    return price_sig

def _cache_ttl_seconds() -> int:
    return int(os.getenv("SIGNAL_CACHE_TTL_SECONDS", "900"))  # 15 min

def _age_seconds(cached_at_iso: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(cached_at_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None

def _is_fresh(cached_at_iso: str, ttl_seconds: int) -> bool:
    age = _age_seconds(cached_at_iso)
    return bool(age is not None and age <= ttl_seconds)

def _cache_bucket() -> Optional[str]:
    return os.getenv("SIGNALS_BUCKET")

def _cache_key(asset: str, mode: str) -> str:
    asset_norm = asset.replace("/", "_").replace("-", "_")
    return f"signals/latest/{asset_norm}/{mode}.json"

# --- LIVE SENTIMENT (Fear & Greed Index) ---
FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"

def _get_fear_greed_score() -> float:
    """Returns 0..1 sentiment score derived from Fear & Greed Index (0..100)."""
    try:
        r = httpx.get(FNG_URL, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        value_0_100 = float(data["data"][0]["value"])
        return max(0.0, min(1.0, value_0_100 / 100.0))
    except Exception:
        return 0.5

def _load_aligned_sentiment(asset: str, price_df):
    score = _get_fear_greed_score()
    return pd.DataFrame(index=price_df.index, data={"sentiment_score": score})

# -------------------------
# Schemas
# -------------------------
class SentimentScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    asset: Optional[str] = None

class SentimentScoreResponse(BaseModel):
    score: float
    engine: str

class SignalResponse(BaseModel):
    asset: str
    mode: str
    latest_timestamp: str
    latest_signal: int
    latest_signal_text: Literal["BUY", "HOLD", "SELL"]
    latest_sentiment: Optional[float] = None
    cached_at_utc: Optional[str] = None

class ExplainRequest(BaseModel):
    asset: str = "BTC-USD"
    mode: Literal["price_only", "combined"] = "combined"

class ExplainResponse(BaseModel):
    explanation: str

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sentiment/latest")
def sentiment_latest():
    score = _get_fear_greed_score()
    return {"source": "alternative.me_fng", "score": score}

@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
):
    t0 = time.time()
    _emit_metric("SignalRequest", 1, asset=asset, mode=mode)

    ttl = _cache_ttl_seconds()
    bucket = _cache_bucket()

    # 1) Try cache
    if bucket:
        cached = read_latest_signal(asset=asset, mode=mode)
        if cached and _is_fresh(cached.get("cached_at", ""), ttl):
            _emit_metric("CacheHit", 1, asset=asset, mode=mode)
            _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000.0, unit="Milliseconds", asset=asset, mode=mode)
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

    # 2) Compute fresh
    t_pipe0 = time.time()
    price_sig = _load_price_pipeline(asset)
    latest_ts = price_sig.index[-1]
    latest_signal = int(price_sig["signal"].iloc[-1])
    latest_sentiment: Optional[float] = None
    t_pipe1 = time.time()
    _emit_metric("PipelineComputeMs", (t_pipe1 - t_pipe0) * 1000.0, unit="Milliseconds", asset=asset, mode=mode)

    if mode == "combined":
        t_comb0 = time.time()
        sent_aligned = _load_aligned_sentiment(asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        latest_signal = int(combined["signal_combined"].iloc[-1])
        latest_sentiment = float(combined["sentiment_score"].iloc[-1])
        t_comb1 = time.time()
        _emit_metric("CombinedSignalMs", (t_comb1 - t_comb0) * 1000.0, unit="Milliseconds", asset=asset, mode=mode)

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

    # 3) Write cache (best effort)
    if bucket:
        try:
            write_latest_signal(
                asset=asset,
                mode=mode,
                payload={
                    "asset": resp.asset,
                    "mode": resp.mode,
                    "latest_timestamp": resp.latest_timestamp,
                    "latest_signal": resp.latest_signal,
                    "latest_sentiment": resp.latest_sentiment,
                    "cached_at": cached_at,
                },
            )
            _emit_metric("CacheWrite", 1, asset=asset, mode=mode)
        except Exception:
            _emit_metric("CacheWriteError", 1, asset=asset, mode=mode)

    _emit_metric("SignalComputed", 1, asset=asset, mode=mode)
    _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000.0, unit="Milliseconds", asset=asset, mode=mode)
    return resp

@app.get("/debug/cache")
def debug_cache(asset: str = "BTC-USD", mode: str = "combined"):
    bucket = _cache_bucket()
    return {
        "signals_bucket_env": bucket,
        "key_effective": _cache_key(asset, mode) if bucket else None,
        "ttl_seconds": _cache_ttl_seconds(),
        "asset": asset,
        "mode": mode,
    }

@app.get("/debug/cache/read")
def debug_cache_read(asset: str = "BTC-USD", mode: str = "combined"):
    """
    Returns the raw cached JSON (if present) + age/freshness.
    Protected by API key middleware.
    """
    bucket = _cache_bucket()
    if not bucket:
        return {"enabled": False, "reason": "SIGNALS_BUCKET not set"}

    ttl = _cache_ttl_seconds()
    key = _cache_key(asset, mode)

    cached = read_latest_signal(asset=asset, mode=mode)
    if not cached:
        return {
            "enabled": True,
            "found": False,
            "bucket": bucket,
            "key": key,
            "ttl_seconds": ttl,
        }

    cached_at = cached.get("cached_at", "")
    age = _age_seconds(cached_at) if cached_at else None
    fresh = _is_fresh(cached_at, ttl) if cached_at else False

    return {
        "enabled": True,
        "found": True,
        "bucket": bucket,
        "key": key,
        "ttl_seconds": ttl,
        "cached_at": cached_at,
        "age_seconds": age,
        "fresh": fresh,
        "payload": cached,
    }

@app.post("/signal/explain", response_model=ExplainResponse)
def explain_signal(req: ExplainRequest):
    price_sig = _load_price_pipeline(req.asset)
    latest_ts = price_sig.index[-1]
    price_signal = int(price_sig["signal"].iloc[-1])

    explanation_parts = [
        f"Asset: {req.asset}",
        f"Timestamp (latest bar): {latest_ts.isoformat()}",
        f"Price-model signal: {_signal_to_text(price_signal)} ({price_signal})",
    ]

    if req.mode == "combined":
        sent_aligned = _load_aligned_sentiment(req.asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        combined_signal = int(combined["signal_combined"].iloc[-1])
        sentiment = float(combined["sentiment_score"].iloc[-1])

        explanation_parts += [
            f"Sentiment score (live, 0..1): {sentiment:.2f}",
            f"Combined signal: {_signal_to_text(combined_signal)} ({combined_signal})",
            "Logic: combined signal adjusts the price-model signal using current market sentiment.",
        ]
    else:
        explanation_parts += [
            "Mode: price_only",
            "Logic: signal is derived strictly from price features (no sentiment adjustment).",
        ]

    return ExplainResponse(explanation="\n".join(explanation_parts))

handler = Mangum(app)
