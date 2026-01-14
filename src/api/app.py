# src/api/app.py
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

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
# Remove FastAPI CORSMiddleware to avoid duplicate Access-Control-* headers.
# Lambda Function URL CORS + our middleware below will handle browser CORS reliably.

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
# CORS helper (single source of truth)
# -------------------------
def _add_cors_headers(resp: Response, request: Request) -> Response:
    """
    Ensure exactly ONE set of CORS headers on every response,
    including 401/404/429/etc, so browsers do not fail with "Failed to fetch".
    """
    # Defensive: remove any existing CORS headers so we never send duplicates.
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

    # You can keep this as "*" since your Lambda Function URL CORS is also "*"
    # and you are NOT using credentials.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "x-api-key,content-type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Vary"] = "Origin"
    return resp

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

        if s and s.strip().startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return str(
                        obj.get("value")
                        or obj.get("secret")
                        or obj.get("key")
                        or obj.get("token")
                        or s
                    )
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
# Usage quotas (DynamoDB)
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
PLAN_DEBUG = os.getenv("PLAN_DEBUG", "0") == "1"

PLAN_DEFAULTS = {"free": 200, "pro": 2000, "enterprise": 10000}
_plan_cache: dict[str, dict] = {}

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
        resp = _ddb.get_item(
            TableName=table,
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
        else:
            limit = int(PLAN_DEFAULTS.get(plan, limit))

        if PLAN_CACHE_SECONDS > 0:
            _plan_cache[key_hash] = {"plan": plan, "limit": limit, "exp": now + PLAN_CACHE_SECONDS}

        return (plan, limit)

    except Exception as e:
        print(f"PLAN_DDB_WARN — {e}")
        _emit_metric("PlanLookupError", 1, key_hash=key_hash)
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
        _emit_metric("QuotaError", 1, endpoint=endpoint, key_hash=key_hash)
        return True

def _quota_status(key_hash: str) -> tuple[int, int]:
    pk = _quota_pk(key_hash)
    used = 0
    reset_epoch = _next_midnight_utc_epoch(extra_minutes=10)

    try:
        resp = _ddb.get_item(
            TableName=_quota_table(),
            Key={"pk": {"S": pk}},
            ConsistentRead=False,
        )
        item = resp.get("Item") or {}
        if "n" in item and "N" in item["n"]:
            used = int(float(item["n"]["N"]))
        if "expires_at" in item and "N" in item["expires_at"]:
            reset_epoch = int(float(item["expires_at"]["N"]))
    except Exception as e:
        print(f"QUOTA_STATUS_WARN — {e}")

    return used, reset_epoch

def _rl_json(status_code: int, body: dict, *, limit: Optional[int] = None, remaining: Optional[int] = None, reset_epoch: Optional[int] = None) -> JSONResponse:
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
PUBLIC_PATHS = {"/health", "/__routes"}  # include __routes for debugging
ADMIN_PATHS = {"/admin/plan", "/admin/whoami"}

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
        "request_client_host": getattr(getattr(request, "client", None), "host", None),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "cf_connecting_ip": request.headers.get("cf-connecting-ip"),
    }

# -------------------------
# Local (in-memory) rate limiter
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

    # ✅ Always allow browser preflight.
    if request.method == "OPTIONS":
        resp = Response(status_code=200)
        return _add_cors_headers(resp, request)

    # Public endpoints: allow through, but STILL add CORS headers.
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
        _emit_metric("ApiKeyMisconfigured", 1, endpoint=endpoint)
        resp = JSONResponse({"detail": "API key auth misconfigured (API_KEY missing)"}, status_code=500)
        return _add_cors_headers(resp, request)

    ok = (provided == api_key) or (demo_key and provided == demo_key)

    if not ok:
        _emit_metric("ApiKeyUnauthorized", 1, endpoint=endpoint, key_hash="unknown")
        _emit_metric("EndpointRequest", 1, endpoint=endpoint)
        resp = JSONResponse({"detail": "Invalid API key"}, status_code=401)
        return _add_cors_headers(resp, request)

    request.state.key_hash = _sha256_12(provided)
    _emit_metric("ApiKeyRequest", 1, endpoint=endpoint, key_hash=request.state.key_hash)
    _emit_metric("EndpointRequest", 1, endpoint=endpoint)

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
# ... rest of your file unchanged (routes, models, etc.)
# -------------------------------------------------------------------

handler = Mangum(app)
