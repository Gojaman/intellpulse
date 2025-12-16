from __future__ import annotations

import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any, List

import boto3
import httpx
import numpy as np
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

app = FastAPI(title="Intellpulse API", version="0.2.8")

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
# API Key protection + Usage Tracking
# -------------------------
PUBLIC_PATHS = {"/health"}

def _key_hash(x_api_key: Optional[str]) -> str:
    """
    Privacy-safe identifier for usage tracking.
    Never emit/store raw API keys.
    """
    if not x_api_key:
        return "unknown"
    return hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()[:12]

def _endpoint_name(path: str) -> str:
    return path or "unknown"

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    endpoint = _endpoint_name(request.url.path)

    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    provided = request.headers.get("x-api-key")
    request.state.key_hash = _key_hash(provided)

    api_key = os.getenv("API_KEY")
    if not api_key:
        # no auth configured → still track endpoint usage (no key)
        _emit_metric("EndpointRequest", 1, endpoint=endpoint)
        return await call_next(request)

    if provided != api_key:
        _emit_metric("ApiKeyUnauthorized", 1, endpoint=endpoint, key_hash="unknown")
        _emit_metric("EndpointRequest", 1, endpoint=endpoint)
        return JSONResponse({"detail": "Invalid API key"}, status_code=401)

    # Authorized
    _emit_metric("ApiKeyRequest", 1, endpoint=endpoint, key_hash=request.state.key_hash)
    _emit_metric("EndpointRequest", 1, endpoint=endpoint)

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

def _dump(model_obj) -> dict:
    """
    Pydantic v2: model_dump()
    Pydantic v1: dict()
    """
    if model_obj is None:
        return {}
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    if hasattr(model_obj, "dict"):
        return model_obj.dict()
    return dict(model_obj)

def _extract_latest_features(df: pd.DataFrame) -> dict:
    preferred = ["close", "ma_20", "ma_50", "rsi_14", "return", "vol_20"]
    out = {}
    if df is None or len(df) == 0:
        return out
    row = df.iloc[-1]
    for c in preferred:
        if c in df.columns:
            v = row.get(c)
            try:
                if pd.notna(v):
                    out[c] = float(v)
            except Exception:
                pass
    return out

def _build_explain(
    asset: str,
    mode: str,
    price_df: pd.DataFrame,
    latest_signal: int,
    latest_sentiment: Optional[float],
) -> "ExplainBlock":
    features = _extract_latest_features(price_df)

    parts = [
        f"Asset: {asset}",
        f"Mode: {mode}",
        f"Decision: {_signal_to_text(latest_signal)} ({latest_signal})",
    ]

    shown = []
    for k in ["close", "ma_20", "rsi_14"]:
        if k in features:
            shown.append(f"{k}={features[k]:.4f}")
    if shown:
        parts.append("Key indicators: " + ", ".join(shown))

    if mode == "combined":
        if latest_sentiment is None:
            parts.append("Sentiment: unavailable")
        else:
            parts.append(f"Sentiment (Fear&Greed normalized 0–1): {latest_sentiment:.4f}")

    parts.append(
        "Note: decision is produced by the deployed rule-based engine; this block exposes the inputs used for transparency."
    )

    facts = {
        "engine": {
            "price": "generate_rule_based_signal",
            "combined": "generate_combined_signal",
        },
        "latest_signal": latest_signal,
        "latest_signal_text": _signal_to_text(latest_signal),
        "latest_sentiment": latest_sentiment,
        "price_features_used": features,
    }

    return ExplainBlock(summary=" | ".join(parts), facts=facts)

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
        return 0.5  # stable fallback

def _load_aligned_sentiment(asset: str, price_df):
    return pd.DataFrame(index=price_df.index, data={"sentiment_score": _get_fear_greed_score()})

# -------------------------
# Backtest helpers
# -------------------------
def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def _infer_bar_interval(index: pd.DatetimeIndex) -> str:
    if index is None or len(index) < 3:
        return "unknown"
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return "unknown"
    med = deltas.median()
    sec = float(med.total_seconds())
    if 3600 * 0.9 <= sec <= 3600 * 1.1:
        return "1h"
    if 86400 * 0.9 <= sec <= 86400 * 1.1:
        return "1d"
    return "unknown"

def _default_annualization(bar_interval: str) -> int:
    if bar_interval == "1h":
        return 365 * 24
    if bar_interval == "1d":
        return 365
    return 252

def _trade_stats(position: pd.Series, strat_ret: pd.Series) -> Dict[str, Any]:
    pos = position.fillna(0).astype(int)
    r = strat_ret.fillna(0.0)

    in_trade = False
    trade_pnls: List[float] = []
    cur = 1.0

    for i in range(len(pos)):
        p = int(pos.iloc[i])
        ri = float(r.iloc[i])

        if not in_trade and p != 0:
            in_trade = True
            cur = 1.0

        if in_trade:
            cur *= (1.0 + ri)

        if in_trade and p == 0:
            in_trade = False
            trade_pnls.append(cur - 1.0)

    if in_trade:
        trade_pnls.append(cur - 1.0)

    if not trade_pnls:
        return {"trades": 0, "win_rate": 0.0}

    wins = sum(1 for x in trade_pnls if x > 0)
    return {"trades": int(len(trade_pnls)), "win_rate": float(wins / len(trade_pnls))}

def _run_backtest(
    df: pd.DataFrame,
    signal_col: str,
    fee_bps: float,
    slippage_bps: float,
    annualization: int,
) -> Dict[str, Any]:
    if "return" not in df.columns:
        raise ValueError("DataFrame must contain 'return' (log returns) column")
    if signal_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{signal_col}' column")

    work = df.copy()
    work["ret_simple"] = np.expm1(work["return"].astype(float))

    sig = work[signal_col].fillna(0).astype(int)
    pos = sig.shift(1).fillna(0).astype(int)
    work["position"] = pos

    total_cost_bps = float(fee_bps) + float(slippage_bps)
    cost_rate = total_cost_bps / 10000.0

    dpos = pos.diff().fillna(pos).abs()
    work["cost"] = dpos.astype(float) * cost_rate

    work["strategy_ret"] = (pos.astype(float) * work["ret_simple"]) - work["cost"]
    work["equity"] = (1.0 + work["strategy_ret"]).cumprod()

    work["buy_hold_equity"] = (1.0 + work["ret_simple"]).cumprod()

    strat = work["strategy_ret"].dropna()
    eq = work["equity"].dropna()
    bh_eq = work["buy_hold_equity"].dropna()

    if len(strat) < 5 or len(eq) < 5:
        return {
            "detail": "Not enough data to backtest (need more rows after feature dropna).",
            "rows": int(len(work)),
        }

    mean_r = float(strat.mean())
    std_r = float(strat.std(ddof=0))
    vol = float(std_r * np.sqrt(annualization)) if std_r > 0 else 0.0
    sharpe = float((mean_r / std_r) * np.sqrt(annualization)) if std_r > 0 else 0.0

    total_return = float(eq.iloc[-1] - 1.0)
    mdd = _max_drawdown(eq)
    trade_stats = _trade_stats(work["position"], work["strategy_ret"])

    buy_hold_total_return = float(bh_eq.iloc[-1] - 1.0) if len(bh_eq) else 0.0
    buy_hold_max_drawdown = _max_drawdown(bh_eq) if len(bh_eq) else 0.0
    buy_hold_equity_end = float(bh_eq.iloc[-1]) if len(bh_eq) else 1.0

    tail = work[["equity"]].tail(50)
    equity_points = [
        {"t": idx.isoformat(), "equity": float(val)}
        for idx, val in zip(tail.index, tail["equity"].values)
    ]

    return {
        "rows": int(len(work)),
        "signal_col": signal_col,
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "annualization": int(annualization),
        "total_return": total_return,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": trade_stats["trades"],
        "win_rate": trade_stats["win_rate"],
        "equity_end": float(eq.iloc[-1]),
        "equity_points": equity_points,
        "buy_hold_total_return": buy_hold_total_return,
        "buy_hold_max_drawdown": buy_hold_max_drawdown,
        "buy_hold_equity_end": buy_hold_equity_end,
        "period_start": work.index.min().isoformat() if len(work) else None,
        "period_end": work.index.max().isoformat() if len(work) else None,
    }

# -------------------------
# Schemas
# -------------------------
class ExplainBlock(BaseModel):
    summary: str
    facts: dict = Field(default_factory=dict)

class SignalResponse(BaseModel):
    asset: str
    mode: str
    latest_timestamp: str
    latest_signal: int
    latest_signal_text: Literal["BUY", "HOLD", "SELL"]
    latest_sentiment: Optional[float] = None
    cached_at_utc: Optional[str] = None
    explain: Optional[ExplainBlock] = None

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
    explain: int = 0,
):
    t0 = time.time()
    _emit_metric("SignalRequest", 1, asset=asset, mode=mode)

    ttl = _cache_ttl_seconds()

    try:
        bucket = cache_bucket()
        cached = read_latest_signal(asset, mode)
    except Exception:
        bucket = None
        cached = None

    cached_at_val = cached.get("cached_at", "") if cached else ""
    age = _age_seconds(cached_at_val) if cached else None
    fresh = _is_fresh(cached_at_val, ttl) if cached else False

    if cached and fresh and explain != 1:
        _emit_metric("CacheHit", 1, asset=asset, mode=mode)
        if age is not None:
            _emit_metric("CacheAgeSeconds", age, unit="Seconds", asset=asset, mode=mode)
        _emit_metric("CacheFresh", 1, unit="Count", asset=asset, mode=mode)

        _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000, unit="Milliseconds", asset=asset, mode=mode)

        return SignalResponse(
            asset=asset,
            mode=mode,
            latest_timestamp=cached["latest_timestamp"],
            latest_signal=int(cached["latest_signal"]),
            latest_signal_text=_signal_to_text(int(cached["latest_signal"])),
            latest_sentiment=cached.get("latest_sentiment"),
            cached_at_utc=cached.get("cached_at"),
            explain=cached.get("explain") if explain == 1 else None,
        )

    _emit_metric("CacheMiss", 1, asset=asset, mode=mode)
    _emit_metric("CacheFresh", 0, unit="Count", asset=asset, mode=mode)
    if cached and age is not None:
        _emit_metric("CacheAgeSeconds", age, unit="Seconds", asset=asset, mode=mode)

    try:
        price_sig = _load_price_pipeline(asset)
        if isinstance(price_sig, JSONResponse):
            return price_sig

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
            explain=None,
        )

        if explain == 1:
            resp.explain = _build_explain(asset, mode, price_sig, latest_signal, latest_sentiment)

        if bucket:
            payload = {
                "asset": resp.asset,
                "mode": resp.mode,
                "latest_timestamp": resp.latest_timestamp,
                "latest_signal": resp.latest_signal,
                "latest_sentiment": resp.latest_sentiment,
                "cached_at": cached_at,
                "explain": _dump(resp.explain) if resp.explain else None,
            }
            try:
                write_latest_signal(asset, mode, payload)
                _emit_metric("CacheWrite", 1, asset=asset, mode=mode)
            except Exception:
                _emit_metric("CacheWriteError", 1, asset=asset, mode=mode)

        _emit_metric("SignalLatencyMs", (time.time() - t0) * 1000, unit="Milliseconds", asset=asset, mode=mode)
        return resp

    except Exception as e:
        reason = type(e).__name__
        _emit_metric("SignalError", 1, asset=asset, mode=mode, reason=reason)
        print(f"SIGNAL_ERROR — asset={asset} mode={mode} reason={reason} err={e}")
        return JSONResponse(status_code=503, content={"detail": f"Data unavailable for asset {asset}"})

@app.get("/signal/explain", response_model=SignalResponse)
def get_signal_explain(asset: str = "BTC-USD", mode: Literal["price_only", "combined"] = "combined"):
    return get_signal(asset=asset, mode=mode, explain=1)

@app.get("/backtest")
def backtest(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    period: Optional[Literal["1h", "1d"]] = None,
    annualization: Optional[int] = None,
):
    t0 = time.time()
    _emit_metric("BacktestRequest", 1, asset=asset, mode=mode)

    try:
        price_sig = _load_price_pipeline(asset)
        if isinstance(price_sig, JSONResponse):
            return price_sig

        df = price_sig
        signal_col = "signal"

        if mode == "combined":
            sent = _load_aligned_sentiment(asset, df)
            df = generate_combined_signal(df, sent)
            signal_col = "signal_combined"

        inferred = _infer_bar_interval(df.index)
        bar_interval = period or inferred
        ann = int(annualization) if annualization is not None else _default_annualization(bar_interval)

        result = _run_backtest(
            df=df,
            signal_col=signal_col,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            annualization=ann,
        )

        _emit_metric("BacktestLatencyMs", (time.time() - t0) * 1000, unit="Milliseconds", asset=asset, mode=mode)

        return {
            "asset": asset,
            "mode": mode,
            "bar_interval": bar_interval,
            **result,
        }

    except Exception as e:
        reason = type(e).__name__
        _emit_metric("BacktestError", 1, asset=asset, mode=mode, reason=reason)
        print(f"BACKTEST_ERROR — asset={asset} mode={mode} reason={reason} err={e}")
        return JSONResponse(status_code=503, content={"detail": f"Backtest unavailable for asset {asset}"})

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
        return {"enabled": True, "found": False, "bucket": bucket, "key": key, "ttl_seconds": ttl}

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
