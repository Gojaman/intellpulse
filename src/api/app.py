# src/api/app.py
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Optional

import boto3
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mangum import Mangum

from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_combined_signal, generate_rule_based_signal
from src.utils.data_loader import load_price_data
from src.utils.s3_store import (
    cache_bucket,
    cache_key,
    read_latest_signal,
    write_latest_signal,
)

app = FastAPI(title="Intellpulse API", version="0.2.10")

# IMPORTANT:
# Do NOT add FastAPI CORSMiddleware. Lambda Function URL CORS + middleware below handles it.
# (FastAPI CORSMiddleware can cause duplicate Access-Control-* headers -> browser "Failed to fetch".)


# -------------------------
# CORS helper (single source of truth)
# -------------------------
def _add_cors_headers(resp: Response, request: Request) -> Response:
    # Remove any existing CORS headers so we never send duplicates.
    for h in [
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Methods",
        "Access-Control-Expose-Headers",
        "Access-Control-Max-Age",
        "Vary",
    ]:
        if h in resp.headers:
            del resp.headers[h]

    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "x-api-key,content-type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Vary"] = "Origin"
    return resp


# -------------------------
# Basic health/debug
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/__routes")
def __routes():
    return sorted([r.path for r in app.router.routes])


# -------------------------
# AWS clients
# -------------------------
_cw = boto3.client("cloudwatch")
_ddb = boto3.client("dynamodb")
_sm = boto3.client("secretsmanager")

METRICS_NS = os.getenv("METRICS_NAMESPACE", "Intellpulse/MVP1")
SERVICE_NAME = os.getenv("SERVICE_NAME", os.getenv("AWS_LAMBDA_FUNCTION_NAME", "intellpulse-api"))


def _emit_metric(name: str, value: float = 1.0, unit: str = "Count", **dims) -> None:
    try:
        dimensions = [{"Name": "Service", "Value": SERVICE_NAME}]
        for k, v in dims.items():
            if v is not None:
                dimensions.append({"Name": str(k), "Value": str(v)})

        _cw.put_metric_data(
            Namespace=METRICS_NS,
            MetricData=[
                {"MetricName": name, "Value": float(value), "Unit": unit, "Dimensions": dimensions}
            ],
        )
    except Exception as e:
        print(f"METRICS_WARN — {e}")


# -------------------------
# Secrets (API key + Admin key)
# -------------------------
API_KEY_SECRET_ARN = os.getenv("API_KEY_SECRET_ARN", "").strip()
ADMIN_KEY_SECRET_ARN = os.getenv("ADMIN_KEY_SECRET_ARN", "").strip()


@lru_cache(maxsize=32)
def _get_secret_string(secret_id: str) -> str:
    if not secret_id:
        return ""
    try:
        resp = _sm.get_secret_value(SecretId=secret_id)
        s = resp.get("SecretString") or ""

        # If secret is JSON, try common fields
        if s and s.strip().startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return str(obj.get("value") or obj.get("secret") or obj.get("key") or obj.get("token") or s)
            except Exception:
                pass

        return s
    except Exception as e:
        print(f"SECRETS_WARN — {e}")
        _emit_metric("SecretsError", 1, secret_id=secret_id[:32])
        return ""


def _get_api_key() -> str:
    return _get_secret_string(API_KEY_SECRET_ARN) or os.getenv("API_KEY", "")


def _get_admin_key() -> str:
    return _get_secret_string(ADMIN_KEY_SECRET_ARN) or os.getenv("ADMIN_KEY", "")


# -------------------------
# Dynamic config getters
# -------------------------
def _rate_table() -> str:
    return os.getenv("RATE_LIMIT_TABLE", "intellpulse-rate-limit")


def _quota_enabled() -> bool:
    return os.getenv("QUOTA_ENABLED", "0") == "1"


def _quota_table() -> str:
    return os.getenv("QUOTA_TABLE", _rate_table())


def _quota_daily_limit_default() -> int:
    return int(os.getenv("QUOTA_DAILY_LIMIT", "2000"))


def _global_rate_enabled() -> bool:
    return os.getenv("GLOBAL_RATE_LIMIT_ENABLED", "0") == "1"


def _global_rps() -> float:
    return float(os.getenv("GLOBAL_RATE_LIMIT_RPS", "3"))


def _global_burst() -> float:
    return float(os.getenv("GLOBAL_RATE_LIMIT_BURST", "10"))


def _global_ttl_seconds() -> int:
    return int(os.getenv("GLOBAL_RATE_LIMIT_TTL_SECONDS", "3600"))


def _global_window_seconds() -> int:
    return int(os.getenv("GLOBAL_RATE_LIMIT_WINDOW_SECONDS", "60"))


# -------------------------
# Global rate limiter (DynamoDB)
# -------------------------
def _ddb_pk(key_hash: str, endpoint: str, window_id: int) -> str:
    return f"{key_hash}#{endpoint}#{window_id}"


def _epoch_s() -> int:
    return int(time.time())


def _global_allow_request(key_hash: str, endpoint: str, cost: int = 1) -> bool:
    if not _global_rate_enabled():
        return True

    now = _epoch_s()
    window = max(1, int(_global_window_seconds()))
    window_id = now // window
    pk = _ddb_pk(key_hash, endpoint, window_id)

    rps = max(0.0, float(_global_rps()))
    burst = max(0.0, float(_global_burst()))
    limit = int(max(1.0, rps) * window + burst)

    expires_at = now + max(60, int(_global_ttl_seconds()))
    table = _rate_table()

    try:
        _ddb.update_item(
            TableName=table,
            Key={"pk": {"S": pk}},
            UpdateExpression="SET expires_at = :exp ADD #n :inc",
            ConditionExpression="attribute_not_exists(#n) OR #n < :limit",
            ExpressionAttributeNames={"#n": "n"},
            ExpressionAttributeValues={
                ":inc": {"N": str(int(max(1, cost)))},
                ":limit": {"N": str(limit)},
                ":exp": {"N": str(expires_at)},
            },
        )
        return True
    except _ddb.exceptions.ConditionalCheckFailedException:
        return False


# -------------------------
# Usage quotas (DynamoDB) — daily counter per API key
# -------------------------
BILLABLE_PATHS = {"/signal", "/signal/explain", "/backtest"}


def _utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _next_midnight_utc_epoch(extra_minutes: int = 10) -> int:
    now = datetime.now(timezone.utc)
    tomorrow_midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
    return int((tomorrow_midnight + timedelta(minutes=extra_minutes)).timestamp())


def _quota_pk(key_hash: str) -> str:
    return f"quota#{key_hash}#{_utc_yyyymmdd()}"


# -------------------------
# Plans mapping (DynamoDB)
# -------------------------
PLAN_ENABLED = os.getenv("PLAN_ENABLED", "1") == "1"
PLAN_CACHE_SECONDS = int(os.getenv("PLAN_CACHE_SECONDS", "60"))
PLAN_DEFAULTS = {"free": 200, "pro": 2000, "enterprise": 10000}
_plan_cache: dict[str, dict[str, Any]] = {}


def _plan_pk(key_hash: str) -> str:
    return f"plan#{key_hash}"


def _plan_get_limit(key_hash: str) -> tuple[str, int]:
    if not PLAN_ENABLED:
        return ("pro", int(_quota_daily_limit_default()))

    now = int(time.time())
    if PLAN_CACHE_SECONDS > 0:
        hit = _plan_cache.get(key_hash)
        if hit and hit.get("exp", 0) > now:
            return (hit["plan"], int(hit["limit"]))

    plan = "pro"
    limit = int(_quota_daily_limit_default())
    table = _quota_table()

    try:
        resp = _ddb.get_item(TableName=table, Key={"pk": {"S": _plan_pk(key_hash)}}, ConsistentRead=False)
        item = resp.get("Item")
        if item:
            if "plan" in item and "S" in item["plan"]:
                plan = item["plan"]["S"]
            if "daily_limit" in item and "N" in item["daily_limit"]:
                limit = int(float(item["daily_limit"]["N"]))
            else:
                limit = int(PLAN_DEFAULTS.get(plan, limit))
        else:
            limit = int(PLAN_DEFAULTS.get(plan, limit))

        if PLAN_CACHE_SECONDS > 0:
            _plan_cache[key_hash] = {"plan": plan, "limit": limit, "exp": now + PLAN_CACHE_SECONDS}

        return (plan, limit)
    except Exception as e:
        print(f"PLAN_DDB_WARN — {e}")
        return ("pro", int(_quota_daily_limit_default()))


def _quota_allow_request_with_limit(key_hash: str, endpoint: str, daily_limit: int, cost: int = 1) -> bool:
    if not _quota_enabled():
        return True

    pk = _quota_pk(key_hash)
    expires_at = _next_midnight_utc_epoch(extra_minutes=10)
    table = _quota_table()

    try:
        _ddb.update_item(
            TableName=table,
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
        return True  # fail-open


def _quota_status(key_hash: str) -> tuple[int, int]:
    pk = _quota_pk(key_hash)
    used = 0
    reset_epoch = _next_midnight_utc_epoch(extra_minutes=10)

    try:
        resp = _ddb.get_item(TableName=_quota_table(), Key={"pk": {"S": pk}}, ConsistentRead=False)
        item = resp.get("Item") or {}
        if "n" in item and "N" in item["n"]:
            used = int(float(item["n"]["N"]))
        if "expires_at" in item and "N" in item["expires_at"]:
            reset_epoch = int(float(item["expires_at"]["N"]))
    except Exception:
        pass

    return used, reset_epoch


def _rl_json(
    status_code: int,
    body: dict,
    *,
    limit: Optional[int] = None,
    remaining: Optional[int] = None,
    reset_epoch: Optional[int] = None,
) -> JSONResponse:
    headers: dict[str, str] = {}
    if reset_epoch is not None:
        headers["X-RateLimit-Reset"] = str(int(reset_epoch))
        retry_after = max(1, int(reset_epoch) - _epoch_s())
        headers["Retry-After"] = str(retry_after)
    if limit is not None:
        headers["X-RateLimit-Limit"] = str(int(limit))
    if remaining is not None:
        headers["X-RateLimit-Remaining"] = str(int(remaining))
    return JSONResponse(content=body, status_code=status_code, headers=headers)


# -------------------------
# API Key + Admin Key
# -------------------------
PUBLIC_PATHS = {"/health", "/__routes"}
ADMIN_PATHS = {"/admin/whoami"}

ADMIN_IP_ALLOWLIST = os.getenv("ADMIN_IP_ALLOWLIST", "").strip()


def _sha256_12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _get_client_ip(request: Request) -> Optional[str]:
    try:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            ip = xff.split(",")[0].strip()
            if ip:
                return ip
        cf = request.headers.get("cf-connecting-ip", "").strip()
        if cf:
            return cf
        xri = request.headers.get("x-real-ip", "").strip()
        if xri:
            return xri
        return getattr(getattr(request, "client", None), "host", None)
    except Exception:
        return None


def _ip_allowed(ip: Optional[str]) -> bool:
    if not ADMIN_IP_ALLOWLIST:
        return True
    if not ip:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
        for part in ADMIN_IP_ALLOWLIST.split(","):
            part = part.strip()
            if not part:
                continue
            net = ipaddress.ip_network(part, strict=False)
            if ip_obj in net:
                return True
        return False
    except Exception:
        return False


def _require_admin(request: Request) -> Optional[JSONResponse]:
    admin_key = _get_admin_key()
    if not admin_key:
        return JSONResponse({"detail": "ADMIN_KEY not configured"}, status_code=503)

    client_ip = _get_client_ip(request)
    if not _ip_allowed(client_ip):
        _emit_metric("AdminIpDenied", 1, endpoint=request.url.path, ip=client_ip or "none")
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    provided = request.headers.get("x-admin-key", "")
    if provided != admin_key:
        _emit_metric("AdminUnauthorized", 1, endpoint=request.url.path)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return None


@app.get("/admin/whoami")
def admin_whoami(request: Request):
    return {
        "client_ip": _get_client_ip(request),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "cf_connecting_ip": request.headers.get("cf-connecting-ip"),
    }


# -------------------------
# Local (in-memory) limiter (fallback)
# -------------------------
def _rate_limit_rps() -> float:
    return float(os.getenv("RATE_LIMIT_RPS", "3"))


def _rate_limit_burst() -> float:
    return float(os.getenv("RATE_LIMIT_BURST", "10"))


_local_buckets: dict[str, dict[str, float]] = {}


def _local_allow_request(key_hash: str, endpoint: str, cost: float = 1.0) -> bool:
    rps = max(0.01, _rate_limit_rps())
    cap = max(1.0, _rate_limit_burst())
    now = time.time()
    bid = f"{key_hash}#{endpoint}"
    st = _local_buckets.get(bid) or {"tokens": cap, "last": now}
    elapsed = max(0.0, now - float(st["last"]))
    st["tokens"] = min(cap, float(st["tokens"]) + elapsed * rps)
    st["last"] = now
    _local_buckets[bid] = st
    if st["tokens"] >= cost:
        st["tokens"] -= cost
        return True
    return False


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    endpoint = request.url.path

    if request.method == "OPTIONS":
        resp = Response(status_code=200)
        return _add_cors_headers(resp, request)

    if endpoint in PUBLIC_PATHS:
        resp = await call_next(request)
        return _add_cors_headers(resp, request)

    if endpoint in ADMIN_PATHS:
        rej = _require_admin(request)
        if rej:
            return _add_cors_headers(rej, request)
        resp = await call_next(request)
        return _add_cors_headers(resp, request)

    api_key = _get_api_key()
    demo_key = os.getenv("DEMO_API_KEY", "").strip()
    provided = (request.headers.get("x-api-key") or "").strip()

    if not api_key:
        resp = JSONResponse({"detail": "API key auth misconfigured (API_KEY missing)"}, status_code=500)
        return _add_cors_headers(resp, request)

    ok = (provided == api_key) or (demo_key and provided == demo_key)
    if not ok:
        resp = JSONResponse({"detail": "Invalid API key"}, status_code=401)
        return _add_cors_headers(resp, request)

    request.state.key_hash = _sha256_12(provided)

    if endpoint in BILLABLE_PATHS:
        plan, limit = _plan_get_limit(request.state.key_hash)
        okq = _quota_allow_request_with_limit(request.state.key_hash, endpoint, daily_limit=limit, cost=1)
        if not okq:
            used, reset_epoch = _quota_status(request.state.key_hash)
            remaining = max(0, int(limit) - int(used))
            resp = _rl_json(
                429,
                {
                    "error": "quota_exceeded",
                    "message": "Quota exceeded",
                    "plan": plan,
                    "daily_limit": int(limit),
                    "used_today": int(used),
                    "remaining": int(remaining),
                    "reset_epoch": int(reset_epoch),
                },
                limit=int(limit),
                remaining=int(remaining),
                reset_epoch=int(reset_epoch),
            )
            return _add_cors_headers(resp, request)

    if not _global_allow_request(request.state.key_hash, endpoint, cost=1):
        window = max(1, int(_global_window_seconds()))
        reset_epoch = ((_epoch_s() // window) + 1) * window
        resp = _rl_json(429, {"error": "rate_limited_global", "message": "Rate limit exceeded"}, reset_epoch=reset_epoch)
        return _add_cors_headers(resp, request)

    if not _local_allow_request(request.state.key_hash, endpoint, cost=1.0):
        resp = _rl_json(429, {"error": "rate_limited_local", "message": "Rate limit exceeded"}, reset_epoch=_epoch_s() + 1)
        return _add_cors_headers(resp, request)

    resp = await call_next(request)
    return _add_cors_headers(resp, request)


# -------------------------------------------------------------------
# Signal / Explain / Backtest / Usage endpoints (RESTORED)
# -------------------------------------------------------------------
def _map_signal(num: Any) -> str:
    try:
        n = int(num)
    except Exception:
        n = 0
    if n == 1:
        return "BUY"
    if n == -1:
        return "SELL"
    return "HOLD"


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _coerce_signal_payload(out: Any) -> dict:
    """
    Accept various shapes from your internal engine and normalize to the API contract.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # If engine returned dict already
    if isinstance(out, dict):
        latest_signal = _safe_get(out, "latest_signal", _safe_get(out, "signal", 0))
        latest_sentiment = _safe_get(out, "latest_sentiment", _safe_get(out, "sentiment", 0.0))
        latest_ts = _safe_get(out, "latest_timestamp", _safe_get(out, "timestamp", None))

        payload = {
            "latest_timestamp": latest_ts,
            "latest_signal": int(latest_signal) if str(latest_signal).lstrip("-").isdigit() else latest_signal,
            "latest_signal_text": _map_signal(latest_signal),
            "latest_sentiment": float(latest_sentiment) if latest_sentiment is not None else 0.0,
            "cached_at_utc": now,
            "explain": _safe_get(out, "explain", None),
        }
        return payload

    # If engine returned tuple/list (signal, sentiment, ts, explain?)
    if isinstance(out, (list, tuple)):
        parts = list(out) + [None, None, None, None]
        latest_signal = parts[0]
        latest_sentiment = parts[1]
        latest_ts = parts[2]
        explain = parts[3]
        return {
            "latest_timestamp": latest_ts,
            "latest_signal": latest_signal,
            "latest_signal_text": _map_signal(latest_signal),
            "latest_sentiment": float(latest_sentiment) if latest_sentiment is not None else 0.0,
            "cached_at_utc": now,
            "explain": explain,
        }

    # Fallback
    return {
        "latest_timestamp": None,
        "latest_signal": 0,
        "latest_signal_text": "HOLD",
        "latest_sentiment": 0.0,
        "cached_at_utc": now,
        "explain": None,
    }


def _compute_signal(asset: str, mode: str) -> dict:
    # Try S3 cache first (optional)
    try:
        b = cache_bucket()
        k = cache_key(asset=asset, mode=mode)
        cached = read_latest_signal(bucket=b, key=k)
        if cached:
            if isinstance(cached, dict):
                return cached
    except Exception as e:
        print(f"CACHE_READ_WARN — {e}")

    # Compute
    df = load_price_data(asset)
    feats = build_price_feature_set(df)

    if mode == "rule":
        out = generate_rule_based_signal(feats)
    else:
        out = generate_combined_signal(feats)

    payload = _coerce_signal_payload(out)
    payload["asset"] = asset
    payload["mode"] = mode

    # Write cache best-effort
    try:
        b = cache_bucket()
        k = cache_key(asset=asset, mode=mode)
        write_latest_signal(bucket=b, key=k, obj=payload)
    except Exception as e:
        print(f"CACHE_WRITE_WARN — {e}")

    return payload


@app.get("/signal")
def signal(asset: str = "BTC-USD", mode: str = "combined"):
    mode = mode.lower().strip()
    if mode not in {"combined", "rule"}:
        mode = "combined"

    payload = _compute_signal(asset=asset, mode=mode)
    return payload


@app.get("/signal/explain")
def signal_explain(asset: str = "BTC-USD", mode: str = "combined"):
    mode = mode.lower().strip()
    if mode not in {"combined", "rule"}:
        mode = "combined"

    payload = _compute_signal(asset=asset, mode=mode)
    explain = payload.get("explain")

    # Normalize explain to your frontend shape
    summary = None
    if isinstance(explain, dict):
        summary = explain.get("summary") or explain.get("explanation") or explain.get("text")
    elif isinstance(explain, str):
        summary = explain

    return {"explain": {"summary": summary or "No explanation available."}}


@app.get("/backtest")
def backtest(asset: str = "BTC-USD", mode: str = "combined", window_days: int = 90):
    """
    Lightweight backtest. If your engine already provides richer backtest,
    you can replace this later — but this returns stable numbers for the UI now.
    """
    mode = mode.lower().strip()
    if mode not in {"combined", "rule"}:
        mode = "combined"

    df = load_price_data(asset)
    if df is None or len(df) < 10:
        return {
            "total_return": 0.0,
            "buy_hold_total_return": 0.0,
            "trades": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    # Try common column names
    price_col = None
    for c in ["close", "Close", "price", "Price"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        # fallback: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        price_col = num_cols[0] if num_cols else df.columns[-1]

    s = df[price_col].astype(float).dropna()
    if len(s) < 10:
        return {
            "total_return": 0.0,
            "buy_hold_total_return": 0.0,
            "trades": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    # Use last N days-ish
    s = s.tail(max(20, int(window_days)))

    # Basic returns
    rets = s.pct_change().fillna(0.0)

    # Strategy: use latest signal as a constant position for now (keeps UI stable)
    sig = _compute_signal(asset=asset, mode=mode).get("latest_signal", 0)
    try:
        sig = int(sig)
    except Exception:
        sig = 0
    position = 1.0 if sig == 1 else (-1.0 if sig == -1 else 0.0)

    strat_rets = rets * position
    total_return = float((1.0 + strat_rets).prod() - 1.0)
    buy_hold_total_return = float((1.0 + rets).prod() - 1.0)

    # Sharpe (simple daily-ish; safe)
    mean = float(strat_rets.mean())
    std = float(strat_rets.std()) if float(strat_rets.std()) > 0 else 0.0
    sharpe = float((mean / std) * (252 ** 0.5)) if std > 0 else 0.0

    # Max drawdown
    equity = (1.0 + strat_rets).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_drawdown = float(dd.min()) if len(dd) else 0.0

    # Trades / win_rate (placeholder but consistent)
    trades = 1 if position != 0 else 0
    win_rate = 1.0 if total_return > 0 else 0.0 if trades else 0.0

    return {
        "total_return": total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "trades": trades,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


@app.get("/usage")
def usage(request: Request):
    key_hash = getattr(request.state, "key_hash", None)
    if not key_hash:
        return {"detail": "missing key_hash"}

    plan, limit = _plan_get_limit(key_hash)
    used, reset_epoch = _quota_status(key_hash)
    remaining = max(0, int(limit) - int(used))

    return {
        "plan": plan,
        "daily_limit": int(limit),
        "used_today": int(used),
        "remaining": int(remaining),
        "reset_epoch": int(reset_epoch),
    }


handler = Mangum(app)
