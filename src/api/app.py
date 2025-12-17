# src/api/app.py
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone, timedelta
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

app = FastAPI(title="Intellpulse API", version="0.2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CloudWatch Metrics + DynamoDB
# -------------------------
_cw = boto3.client("cloudwatch")
_ddb = boto3.client("dynamodb")

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
        print(f"METRICS_WARN — {e}")


# -------------------------
# Global (DynamoDB) rate limiter (fixed-window counter)
# Robust MVP implementation: always creates items you can scan + verify.
# -------------------------
RATE_TABLE = os.getenv("RATE_LIMIT_TABLE", "intellpulse-rate-limit")
GLOBAL_RATE_ENABLED = os.getenv("GLOBAL_RATE_LIMIT_ENABLED", "0") == "1"
GLOBAL_RPS = float(os.getenv("GLOBAL_RATE_LIMIT_RPS", "3"))  # sustained
GLOBAL_BURST = float(os.getenv("GLOBAL_RATE_LIMIT_BURST", "10"))  # extra headroom
GLOBAL_TTL_SECONDS = int(os.getenv("GLOBAL_RATE_LIMIT_TTL_SECONDS", "3600"))

# Window length in seconds (60 = per-minute quotas; easiest to validate)
GLOBAL_WINDOW_SECONDS = int(os.getenv("GLOBAL_RATE_LIMIT_WINDOW_SECONDS", "60"))


def _ddb_pk(key_hash: str, endpoint: str, window_id: int) -> str:
    # pk must be the partition key name in your table
    return f"{key_hash}#{endpoint}#{window_id}"


def _epoch_s() -> int:
    return int(time.time())


def _global_allow_request(key_hash: str, endpoint: str, cost: int = 1) -> bool:
    """
    Global limiter using a DynamoDB counter per fixed time window.

    Limit per window = GLOBAL_RPS*window + GLOBAL_BURST
    Example: rps=3, window=60 => 180 + 10 = 190 requests/min allowed.
    """
    if not GLOBAL_RATE_ENABLED:
        return True

    now = _epoch_s()
    window = max(1, int(GLOBAL_WINDOW_SECONDS))
    window_id = now // window
    pk = _ddb_pk(key_hash, endpoint, window_id)

    limit = int(max(1.0, GLOBAL_RPS) * window + max(0.0, GLOBAL_BURST))
    expires_at = now + max(60, int(GLOBAL_TTL_SECONDS))

    try:
        _ddb.update_item(
            TableName=RATE_TABLE,
            Key={"pk": {"S": pk}},
            UpdateExpression="SET expires_at = :exp ADD n :inc",
            ConditionExpression="attribute_not_exists(n) OR n < :limit",
            ExpressionAttributeValues={
                ":inc": {"N": str(int(max(1, cost)))},
                ":limit": {"N": str(limit)},
                ":exp": {"N": str(expires_at)},
            },
        )
        return True

    except _ddb.exceptions.ConditionalCheckFailedException:
        # Over limit
        return False

    except Exception as e:
        # MVP: fail-open, but visible
        print(f"RATE_LIMIT_DDB_WARN — {e}")
        _emit_metric("RateLimitDdbError", 1, endpoint=endpoint, key_hash=key_hash)
        return True


# -------------------------
# Usage quotas (DynamoDB) — daily counter per API key
# -------------------------
QUOTA_ENABLED = os.getenv("QUOTA_ENABLED", "0") == "1"
QUOTA_TABLE = os.getenv("QUOTA_TABLE", RATE_TABLE)  # default reuse rate table
QUOTA_DAILY_LIMIT = int(os.getenv("QUOTA_DAILY_LIMIT", "2000"))

# Endpoints that count against daily quota
BILLABLE_PATHS = {"/signal", "/signal/explain", "/backtest"}


def _utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _next_midnight_utc_epoch(extra_minutes: int = 10) -> int:
    now = datetime.now(timezone.utc)
    tomorrow_midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(
        days=1
    )
    return int((tomorrow_midnight + timedelta(minutes=extra_minutes)).timestamp())


def _quota_pk(key_hash: str) -> str:
    return f"quota#{key_hash}#{_utc_yyyymmdd()}"


def _today_quota_pk(key_hash: str) -> str:
    return f"quota#{key_hash}#{_utc_yyyymmdd()}"


def _quota_get_used_today(key_hash: str) -> int:
    """
    Read-only: returns how many requests have been used today.
    Never increments.
    """
    if not QUOTA_ENABLED:
        return 0

    pk = _today_quota_pk(key_hash)
    try:
        resp = _ddb.get_item(
            TableName=QUOTA_TABLE,
            Key={"pk": {"S": pk}},
            ConsistentRead=False,
        )
        item = resp.get("Item") or {}
        n = item.get("n", {}).get("N")
        return int(float(n)) if n is not None else 0
    except Exception as e:
        print(f"QUOTA_DDB_WARN — {e}")
        _emit_metric("QuotaReadError", 1, key_hash=key_hash)
        return 0


def _quota_allow_request(key_hash: str, endpoint: str, cost: int = 1) -> bool:
    """
    Atomically increments today's counter if under limit.
    Fail-open on DDB errors (MVP-safe).
    """
    if not QUOTA_ENABLED:
        return True

    pk = _quota_pk(key_hash)
    expires_at = _next_midnight_utc_epoch(extra_minutes=10)

    try:
        # Only increment if current n < limit
        _ddb.update_item(
            TableName=QUOTA_TABLE,
            Key={"pk": {"S": pk}},
            UpdateExpression="SET #n = if_not_exists(#n, :z) + :c, expires_at = :exp",
            ConditionExpression="attribute_not_exists(#n) OR #n < :limit",
            ExpressionAttributeNames={"#n": "n"},
            ExpressionAttributeValues={
                ":z": {"N": "0"},
                ":c": {"N": str(int(cost))},
                ":limit": {"N": str(int(QUOTA_DAILY_LIMIT))},
                ":exp": {"N": str(int(expires_at))},
            },
        )
        return True

    except _ddb.exceptions.ConditionalCheckFailedException:
        # limit hit
        return False

    except Exception as e:
        print(f"QUOTA_DDB_WARN — {e}")
        _emit_metric("QuotaError", 1, endpoint=endpoint, key_hash=key_hash)
        return True


# -------------------------
# Plans mapping (DynamoDB) — key_hash -> plan -> daily_limit
# -------------------------
PLAN_ENABLED = os.getenv("PLAN_ENABLED", "1") == "1"
PLAN_CACHE_SECONDS = int(os.getenv("PLAN_CACHE_SECONDS", "60"))  # allow 0 to disable cache
PLAN_DEBUG = os.getenv("PLAN_DEBUG", "0") == "1"

# Default limits if no plan item exists
PLAN_DEFAULTS = {
    "free": 200,
    "pro": 2000,
    "enterprise": 10000,
}

_plan_cache: dict[str, dict] = {}  # key_hash -> {"limit": int, "plan": str, "exp": epoch}


def _plan_pk(key_hash: str) -> str:
    return f"plan#{key_hash}"


def _plan_get_limit(key_hash: str) -> tuple[str, int]:
    """
    Returns (plan_name, daily_limit).

    Cache can be disabled by setting PLAN_CACHE_SECONDS=0.
    Fail-open to QUOTA_DAILY_LIMIT on DDB errors (MVP-safe).
    """
    if not PLAN_ENABLED:
        return ("pro", int(QUOTA_DAILY_LIMIT))

    now = int(time.time())

    # Cache hit (only if enabled)
    if PLAN_CACHE_SECONDS > 0:
        hit = _plan_cache.get(key_hash)
        if hit and hit.get("exp", 0) > now:
            return (hit["plan"], int(hit["limit"]))

    # Defaults if missing
    plan = "pro"
    limit = int(QUOTA_DAILY_LIMIT)

    try:
        resp = _ddb.get_item(
            TableName=QUOTA_TABLE,
            Key={"pk": {"S": _plan_pk(key_hash)}},
            ConsistentRead=False,
        )
        item = resp.get("Item")
        if item:
            if "plan" in item and "S" in item["plan"]:
                plan = item["plan"]["S"]
            if "daily_limit" in item and "N" in item["daily_limit"]:
                limit = int(float(item["daily_limit"]["N"]))
            else:
                limit = int(PLAN_DEFAULTS.get(plan, limit))

        # Store cache (only if enabled)
        if PLAN_CACHE_SECONDS > 0:
            _plan_cache[key_hash] = {
                "plan": plan,
                "limit": limit,
                "exp": now + PLAN_CACHE_SECONDS,
            }

        if PLAN_DEBUG:
            print(f"PLAN_DEBUG key_hash={key_hash} plan={plan} limit={limit}")

        return (plan, limit)

    except Exception as e:
        print(f"PLAN_DDB_WARN — {e}")
        _emit_metric("PlanLookupError", 1, key_hash=key_hash)
        return ("pro", int(QUOTA_DAILY_LIMIT))


def _quota_allow_request_with_limit(
    key_hash: str, endpoint: str, daily_limit: int, cost: int = 1
) -> bool:
    """
    Same as _quota_allow_request, but uses a dynamic daily_limit.
    """
    if not QUOTA_ENABLED:
        return True

    pk = _quota_pk(key_hash)
    expires_at = _next_midnight_utc_epoch(extra_minutes=10)

    try:
        _ddb.update_item(
            TableName=QUOTA_TABLE,
            Key={"pk": {"S": pk}},
            UpdateExpression="SET #n = if_not_exists(#n, :z) + :c, expires_at = :exp",
            ConditionExpression="attribute_not_exists(#n) OR #n < :limit",
            ExpressionAttributeNames={"#n": "n"},
            ExpressionAttributeValues={
                ":z": {"N": "0"},
                ":c": {"N": str(int(cost))},
                ":limit": {"N": str(int(daily_limit))},
                ":exp": {"N": str(int(expires_at))},
            },
        )
        return True
    except _ddb.exceptions.ConditionalCheckFailedException:
        return False
    except Exception as e:
        print(f"QUOTA_DDB_WARN — {e}")
        _emit_metric("QuotaError", 1, endpoint=endpoint, key_hash=key_hash)
        return True


# -------------------------
# API Key protection
# -------------------------
PUBLIC_PATHS = {"/health"}


def _sha256_12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Local (in-memory) rate limiter (fallback / extra)
# -------------------------
def _rate_limit_rps() -> float:
    return float(os.getenv("RATE_LIMIT_RPS", "3"))


def _rate_limit_burst() -> float:
    return float(os.getenv("RATE_LIMIT_BURST", "10"))


_local_buckets: dict[str, dict[str, float]] = {}  # {bucket_id: {"tokens":x,"last":ts}}


def _local_allow_request(key_hash: str, endpoint: str, cost: float = 1.0) -> bool:
    rps = max(0.01, _rate_limit_rps())
    cap = max(1.0, _rate_limit_burst())
    now = time.time()
    bid = f"{key_hash}#{endpoint}"

    st = _local_buckets.get(bid)
    if not st:
        st = {"tokens": cap, "last": now}
        _local_buckets[bid] = st

    elapsed = max(0.0, now - float(st["last"]))
    st["tokens"] = min(cap, float(st["tokens"]) + elapsed * rps)
    st["last"] = now

    if st["tokens"] >= cost:
        st["tokens"] -= cost
        return True
    return False


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    endpoint = request.url.path

    if endpoint in PUBLIC_PATHS:
        return await call_next(request)

    api_key = os.getenv("API_KEY")
    provided = request.headers.get("x-api-key")

    # If API_KEY not set, allow (dev mode)
    if not api_key:
        return await call_next(request)

    if provided != api_key:
        _emit_metric("ApiKeyUnauthorized", 1, endpoint=endpoint, key_hash="unknown")
        _emit_metric("EndpointRequest", 1, endpoint=endpoint)
        return JSONResponse({"detail": "Invalid API key"}, status_code=401)

    # Authenticated
    request.state.key_hash = _sha256_12(provided)
    _emit_metric("ApiKeyRequest", 1, endpoint=endpoint, key_hash=request.state.key_hash)
    _emit_metric("EndpointRequest", 1, endpoint=endpoint)

    # Usage quota (daily) — plan-aware
    if endpoint in BILLABLE_PATHS:
        plan, limit = _plan_get_limit(request.state.key_hash)
        ok = _quota_allow_request_with_limit(
            request.state.key_hash, endpoint, daily_limit=limit, cost=1
        )
        if not ok:
            _emit_metric(
                "ApiKeyQuotaExceeded",
                1,
                endpoint=endpoint,
                key_hash=request.state.key_hash,
                plan=plan,
            )
            return JSONResponse(
                {"detail": "Quota exceeded", "plan": plan, "daily_limit": limit}, status_code=429
            )

    # Global limiter (DDB)
    if not _global_allow_request(request.state.key_hash, endpoint, cost=1.0):
        _emit_metric("RateLimitedGlobal", 1, endpoint=endpoint, key_hash=request.state.key_hash)
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
    _emit_metric("RateLimitAllowedGlobal", 1, endpoint=endpoint, key_hash=request.state.key_hash)

    # Local limiter (in-memory)
    if not _local_allow_request(request.state.key_hash, endpoint, cost=1.0):
        _emit_metric("RateLimited", 1, endpoint=endpoint, key_hash=request.state.key_hash)
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

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

    if features:
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
        "engine": {"price": "generate_rule_based_signal", "combined": "generate_combined_signal"},
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
            content={
                "detail": f"Price data not found for asset {asset} ({symbol_filter}). {str(e)}"
            },
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


def _load_aligned_sentiment(asset: str, price_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(index=price_df.index, data={"sentiment_score": _get_fear_greed_score()})


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


class BacktestPoint(BaseModel):
    t: str
    equity: float


class BacktestResponse(BaseModel):
    asset: str
    mode: str
    bar_interval: str = "1h"
    rows: int
    signal_col: str
    fee_bps: float
    slippage_bps: float
    annualization: int
    total_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    equity_end: float
    equity_points: list[BacktestPoint]
    buy_hold_total_return: float
    buy_hold_max_drawdown: float
    buy_hold_equity_end: float


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/usage")
def usage(request: Request):
    key_hash = getattr(request.state, "key_hash", None)
    if not key_hash:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    plan, limit = _plan_get_limit(key_hash)
    used = _quota_get_used_today(key_hash)
    remaining = max(0, int(limit) - int(used))

    resets_at_epoch = _next_midnight_utc_epoch(extra_minutes=10)
    resets_at_iso = (
        datetime.fromtimestamp(resets_at_epoch, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        "plan": plan,
        "daily_limit": int(limit),
        "used_today": int(used),
        "remaining": int(remaining),
        "resets_at": resets_at_iso,
        "resets_at_epoch": int(resets_at_epoch),
        "quota_pk": _today_quota_pk(key_hash),
        "date_utc": _utc_yyyymmdd(),
    }


@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
    explain: int = 0,
):
    t0 = time.time()
    _emit_metric("SignalRequest", 1, asset=asset, mode=mode)

    try:
        ttl = _cache_ttl_seconds()

        # Cache read (best effort)
        try:
            bucket = cache_bucket()
            cached = read_latest_signal(asset, mode)
        except Exception:
            bucket = None
            cached = None

        if cached and _is_fresh(cached.get("cached_at", ""), ttl) and explain != 1:
            _emit_metric("CacheHit", 1, asset=asset, mode=mode)

            age = _age_seconds(cached.get("cached_at", ""))
            if age is not None:
                _emit_metric("CacheAgeSeconds", age, unit="Seconds", asset=asset, mode=mode)
            _emit_metric("CacheFresh", 1, unit="Count", asset=asset, mode=mode)

            _emit_metric(
                "SignalLatencyMs",
                (time.time() - t0) * 1000,
                unit="Milliseconds",
                asset=asset,
                mode=mode,
            )
            return SignalResponse(
                asset=asset,
                mode=mode,
                latest_timestamp=cached["latest_timestamp"],
                latest_signal=int(cached["latest_signal"]),
                latest_signal_text=_signal_to_text(int(cached["latest_signal"])),
                latest_sentiment=cached.get("latest_sentiment"),
                cached_at_utc=cached.get("cached_at"),
                explain=None,
            )

        _emit_metric("CacheMiss", 1, asset=asset, mode=mode)
        _emit_metric("CacheFresh", 0, unit="Count", asset=asset, mode=mode)

        # Compute
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

        # Cache write (best effort)
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

        _emit_metric(
            "SignalLatencyMs",
            (time.time() - t0) * 1000,
            unit="Milliseconds",
            asset=asset,
            mode=mode,
        )
        return resp

    except Exception as e:
        reason = e.__class__.__name__
        _emit_metric("SignalError", 1, asset=asset, mode=mode, reason=reason)
        return JSONResponse(
            status_code=503, content={"detail": "Data unavailable for asset " + asset}
        )


@app.get("/signal/explain", response_model=SignalResponse)
def get_signal_explain(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
):
    return get_signal(asset=asset, mode=mode, explain=1)


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


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


@app.get("/backtest", response_model=BacktestResponse)
def backtest(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    annualization: int = 8760,  # 1h bars default
):
    t0 = time.time()
    _emit_metric("BacktestRequest", 1, asset=asset, mode=mode)

    try:
        price_sig = _load_price_pipeline(asset)
        if isinstance(price_sig, JSONResponse):
            return price_sig

        df = price_sig.copy()
        signal_col = "signal"

        if mode == "combined":
            sent = _load_aligned_sentiment(asset, df)
            df = generate_combined_signal(df, sent)
            signal_col = "signal_combined"

        if "close" not in df.columns or signal_col not in df.columns:
            raise ValueError("missing required columns")

        # returns
        ret = df["close"].pct_change().fillna(0.0)

        # position: enter on next bar to avoid lookahead
        sig = df[signal_col].fillna(0).astype(float).clip(-1, 1)
        pos = sig.shift(1).fillna(0.0)

        # trading cost per position change
        cost_rate = (float(fee_bps) + float(slippage_bps)) / 10000.0
        pos_prev = pos.shift(1).fillna(0.0)
        trade = (pos != pos_prev).astype(float)
        costs = trade * cost_rate

        strat_ret = (pos * ret) - costs
        equity = (1.0 + strat_ret).cumprod()

        total_return = float(equity.iloc[-1] - 1.0)
        vol = float(strat_ret.std() * (annualization**0.5))
        sharpe = float(
            (strat_ret.mean() * annualization)
            / (strat_ret.std() * (annualization**0.5) + 1e-12)
        )
        mdd = _max_drawdown(equity)

        # trades + win rate
        trade_idx = df.index[trade == 1.0]
        trades = int(len(trade_idx))

        # crude win rate: per-bar PnL positive while in position
        wins = int((strat_ret > 0).sum())
        win_rate = float(wins / max(1, int((pos != 0).sum())))

        # buy & hold
        bh_ret = ret
        bh_equity = (1.0 + bh_ret).cumprod()
        bh_total = float(bh_equity.iloc[-1] - 1.0)
        bh_mdd = _max_drawdown(bh_equity)
        bh_end = float(bh_equity.iloc[-1])

        pts = []
        tail = equity.tail(200)  # keep response light
        for ts, v in tail.items():
            pts.append(BacktestPoint(t=ts.isoformat(), equity=float(v)))

        resp = BacktestResponse(
            asset=asset,
            mode=mode,
            bar_interval="1h",
            rows=int(len(df)),
            signal_col=signal_col,
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            annualization=int(annualization),
            total_return=total_return,
            volatility=vol,
            sharpe=sharpe,
            max_drawdown=float(mdd),
            trades=trades,
            win_rate=float(win_rate),
            equity_end=float(equity.iloc[-1]),
            equity_points=pts,
            buy_hold_total_return=bh_total,
            buy_hold_max_drawdown=float(bh_mdd),
            buy_hold_equity_end=bh_end,
        )

        _emit_metric(
            "BacktestLatencyMs",
            (time.time() - t0) * 1000,
            unit="Milliseconds",
            asset=asset,
            mode=mode,
        )
        return resp

    except Exception as e:
        reason = e.__class__.__name__
        _emit_metric("BacktestError", 1, asset=asset, mode=mode, reason=reason)
        return JSONResponse(
            status_code=503, content={"detail": "Backtest unavailable for asset " + asset}
        )


handler = Mangum(app)
