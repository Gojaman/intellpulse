from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_combined_signal, generate_rule_based_signal
from src.utils.data_loader import load_price_data
from src.utils.s3_store import read_latest_signal, write_latest_signal

app = FastAPI(title="Intellpulse API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev. Later lock to your Vercel/Lovable domain.
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return int(os.getenv("SIGNAL_CACHE_TTL_SECONDS", "900"))  # 15 min default


def _is_fresh(cached_at_iso: str, ttl_seconds: int) -> bool:
    try:
        dt = datetime.fromisoformat(cached_at_iso.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age <= ttl_seconds
    except Exception:
        return False


# --- LIVE SENTIMENT (Fear & Greed Index) ---
FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"


def _get_fear_greed_score() -> float:
    """
    Returns 0..1 sentiment score derived from Fear & Greed Index (0..100).
    Fallback to 0.5 if upstream is unavailable (so API never 500s on sentiment fetch).
    """
    try:
        r = httpx.get(FNG_URL, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        value_0_100 = float(data["data"][0]["value"])
        return max(0.0, min(1.0, value_0_100 / 100.0))
    except Exception:
        return 0.5


def _load_aligned_sentiment(asset: str, price_df):
    """
    Create a minimal sentiment series aligned to the same timestamps as price_df.
    MVP: use ONE live market sentiment value and broadcast it across the index.
    """
    score = _get_fear_greed_score()
    sent_aligned = pd.DataFrame(index=price_df.index, data={"sentiment_score": score})
    return sent_aligned


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
    cached_at_utc: Optional[str] = None  # added for transparency


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
    """Simple endpoint so UI can show live sentiment directly."""
    score = _get_fear_greed_score()
    return {"source": "alternative.me_fng", "score": score}


@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
):
    # 1) Read cache first
    try:
        cached = read_latest_signal(asset, mode)
        if cached and _is_fresh(cached.get("cached_at_utc", ""), _cache_ttl_seconds()):
            return SignalResponse(**cached)
    except Exception:
        # Cache should never break the endpoint
        pass

    # 2) Compute
    price_sig = _load_price_pipeline(asset)

    latest_ts = price_sig.index[-1]
    latest_signal = int(price_sig["signal"].iloc[-1])
    latest_sentiment: Optional[float] = None

    if mode == "combined":
        sent_aligned = _load_aligned_sentiment(asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        latest_signal = int(combined["signal_combined"].iloc[-1])
        latest_sentiment = float(combined["sentiment_score"].iloc[-1])

    resp = SignalResponse(
        asset=asset,
        mode=mode,
        latest_timestamp=latest_ts.isoformat(),
        latest_signal=latest_signal,
        latest_signal_text=_signal_to_text(latest_signal),
        latest_sentiment=latest_sentiment,
        cached_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    # 3) Write cache (best-effort)
    try:
        write_latest_signal(resp.model_dump())
    except Exception:
        pass

    return resp


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
