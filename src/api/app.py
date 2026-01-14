# src/api/app.py
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Literal, Optional, Tuple

import boto3
import httpx
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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

app = FastAPI(title="Intellpulse API", version="0.2.10")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Secrets (API key + Admin key)
# -------------------------
API_KEY_SECRET_ARN = os.getenv("API_KEY_SECRET_ARN", "").strip()
ADMIN_KEY_SECRET_ARN = os.getenv("ADMIN_KEY_SECRET_ARN", "").strip()


@lru_cache(maxsize=32)
def _get_secret_string(secret_id: str) -> str:
    """
    Fetches a secret from AWS Secrets Manager.
    Supports:
      - raw SecretString (e.g. "abc123")
      - JSON SecretString like {"value":"abc123"} / {"key":"abc123"} / {"secret":"abc123"}
    Cached in-process via lru_cache to avoid calling Secrets Manager on every request.
    """
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
# Dynamic config getters (IMPORTANT for warm Lambda envs)
# -------------------------
def _rate_table() -> str:
    return os.getenv("RATE_LIMIT_TABLE", "intellpulse-rate-limit")


def _quota_enabled() -> bool:
    return os.getenv("QUOTA_ENABLED", "0") == "1"


def _quota_table() -> str:
    # default reuse rate table
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
# CORS helper for early returns (401/429/etc.)
# -------------------------
def _corsify(request: Request, resp: Response) -> Response:
    """
    Ensure browser clients (Lovable) can read 401/429 responses.
    Without these headers, fetch() often shows "Failed to fetch"
    because the browser blocks the response on CORS.
    """
    origin = request.headers.get("origin")
    resp.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Headers"] = "x-api-key,content-type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


# -------------------------
# Global (DynamoDB) rate limiter (fixed-window counter)
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

    print(
        "RATE_LIMIT_DEBUG "
        + json.dumps(
            {
                "enabled": 1,
                "table": table,
                "endpoint": endpoint,
                "key_hash": key_hash,
                "window": window,
                "window_id": window_id,
                "pk": pk,
                "limit": limit,
                "cost": int(cost),
                "expires_at": expires_at,
            }
        )
    )

    try:
        resp = _ddb.update_item(
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
            ReturnValues="UPDATED_NEW",
        )
        print(
            "RATE_LIMIT_DEBUG_OK "
            + json.dumps(
                {
                    "http": resp.get("ResponseMetadata", {}).get("HTTPStatusCode"),
                    "attrs": resp.get("Attributes"),
                }
            )
        )

        return True

    except _ddb.exceptions.ConditionalCheckFailedException:
        print("RATE_LIMIT_DEBUG_THROTTLED " + json.dumps({"pk": pk, "limit": limit}))
        return False

    except Exception as e:
        # TEMP: fail-closed so we surface the real root cause
        print(
            "RATE_LIMIT_DEBUG_ERR "
            + json.dumps({"err": repr(e), "pk": pk, "table": table})
        )
        raise


# -------------------------
# Usage quotas (DynamoDB) — daily counter per API key
# -------------------------
BILLABLE_PATHS = {"/signal", "/signal/explain", "/backtest"}


def _utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _next_midnight_utc_epoch(extra_minutes: int = 10) -> int:
    now = datetime.now(timezone.utc)
    tomorrow_midnight = datetime(
        now.year, now.month, now.day, tzinfo=timezone.utc
    ) + timedelta(days=1)
    return int((tomorrow_midnight + timedelta(minutes=extra_minutes)).timestamp())


def _quota_pk(key_hash: str) -> str:
    return f"quota#{key_hash}#{_utc_yyyymmdd()}"


# -------------------------
# Plans mapping (DynamoDB) — key_hash -> plan -> daily_limit
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
        return ("pro", int(_quota_daily_limit_default()))


def _quota_allow_request_with_limit(
    key_hash: str, endpoint: str, daily_limit: int, cost: int = 1
) -> bool:
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


# -------------------------
# Quota status + unified 429 response helper (MVP-grade)
# -------------------------
def _quota_status(key_hash: str) -> tuple[int, int]:
    """
    Returns (used_today, reset_epoch).
    Only used on the quota-exceeded path.
    """
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
# API Key + Admin Key (+ Admin IP allowlist)
# -------------------------
PUBLIC_PATHS = {"/health"}
ADMIN_PATHS = {"/admin/plan", "/admin/whoami"}

ADMIN_IP_ALLOWLIST = os.getenv("ADMIN_IP_ALLOWLIST", "").strip()


def _sha256_12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _get_client_ip(request: Request) -> Optional[str]:
    """
    Extract the real client IP behind Lambda Function URL / API Gateway.
    Prefer X-Forwarded-For (first IP), then CF / X-Real-IP.
    """
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
# Local (in-memory) rate limiter (fallback / extra)
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

    # ✅ Allow CORS preflight requests (browser OPTIONS) — must return CORS headers
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "*")
        req_headers = request.headers.get(
            "access-control-request-headers", "x-api-key,content-type"
        )
        req_method = request.headers.get("access-control-request-method", "GET")

        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin if origin else "*",
                "Access-Control-Allow-Methods": req_method if req_method else "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": req_headers,
                "Access-Control-Max-Age": "86400",
                "Vary": "Origin",
            },
        )

    if endpoint in PUBLIC_PATHS:
        return await call_next(request)

    if endpoint in ADMIN_PATHS:
        rej = _require_admin(request)
        if rej:
            return _corsify(request, rej)
        return await call_next(request)

    api_key = _get_api_key()
    demo_key = os.getenv("DEMO_API_KEY", "").strip()
    provided = (request.headers.get("x-api-key") or "").strip()

    if not api_key:
        _emit_metric("ApiKeyMisconfigured", 1, endpoint=endpoint)
        return _corsify(
            request,
            JSONResponse(
                {"detail": "API key auth misconfigured (API_KEY missing)"},
                status_code=500,
            ),
        )

    ok = (provided == api_key) or (demo_key and provided == demo_key)

    if not ok:
        _emit_metric("ApiKeyUnauthorized", 1, endpoint=endpoint, key_hash="unknown")
        _emit_metric("EndpointRequest", 1, endpoint=endpoint)
        return _corsify(request, JSONResponse({"detail": "Invalid API key"}, status_code=401))

    request.state.key_hash = _sha256_12(provided)
    _emit_metric("ApiKeyRequest", 1, endpoint=endpoint, key_hash=request.state.key_hash)
    _emit_metric("EndpointRequest", 1, endpoint=endpoint)

    if endpoint in BILLABLE_PATHS:
        plan, limit = _plan_get_limit(request.state.key_hash)
        ok2 = _quota_allow_request_with_limit(
            request.state.key_hash, endpoint, daily_limit=limit, cost=1
        )
        if not ok2:
            used, reset_epoch = _quota_status(request.state.key_hash)
            remaining = max(0, int(limit) - int(used))
            return _corsify(
                request,
                _rl_json(
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
                ),
            )

    if not _global_allow_request(request.state.key_hash, endpoint, cost=1):
        window = max(1, int(_global_window_seconds()))
        reset_epoch = ((_epoch_s() // window) + 1) * window
        return _corsify(
            request,
            _rl_json(
                429,
                {"error": "rate_limited_global", "message": "Rate limit exceeded"},
                reset_epoch=reset_epoch,
            ),
        )

    if not _local_allow_request(request.state.key_hash, endpoint, cost=1.0):
        return _corsify(
            request,
            _rl_json(
                429,
                {"error": "rate_limited_local", "message": "Rate limit exceeded"},
                reset_epoch=_epoch_s() + 1,
            ),
        )

    return await call_next(request)


# -------------------------
# Admin schemas + routes
# -------------------------
class AdminPlanUpsertRequest(BaseModel):
    api_key: Optional[str] = None
    key_hash: Optional[str] = None
    plan: str = "free"
    daily_limit: int = 200
    note: Optional[str] = None


class AdminPlanResponse(BaseModel):
    key_hash: str
    pk: str
    plan: str
    daily_limit: int
    updated_at: str


def _resolve_key_hash(api_key: Optional[str], key_hash: Optional[str]) -> Optional[str]:
    if api_key:
        return _sha256_12(api_key)
    if key_hash:
        return key_hash.strip()
    return None


@app.post("/admin/plan", response_model=AdminPlanResponse)
def admin_plan_upsert(payload: AdminPlanUpsertRequest, request: Request):
    kh = _resolve_key_hash(payload.api_key, payload.key_hash)
    if not kh:
        return JSONResponse({"detail": "Provide api_key or key_hash"}, status_code=400)

    plan = (payload.plan or "free").strip().lower()
    daily_limit = int(payload.daily_limit)
    if daily_limit < 1:
        return JSONResponse({"detail": "daily_limit must be >= 1"}, status_code=400)

    pk = _plan_pk(kh)
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    item = {
        "pk": {"S": pk},
        "plan": {"S": plan},
        "daily_limit": {"N": str(daily_limit)},
        "updated_at": {"S": updated_at},
    }
    if payload.note:
        item["note"] = {"S": str(payload.note)[:500]}

    _ddb.put_item(TableName=_quota_table(), Item=item)
    _plan_cache.pop(kh, None)
    _emit_metric("AdminPlanUpsert", 1, key_hash=kh, plan=plan)

    return AdminPlanResponse(
        key_hash=kh,
        pk=pk,
        plan=plan,
        daily_limit=daily_limit,
        updated_at=updated_at,
    )


@app.get("/admin/plan", response_model=AdminPlanResponse)
def admin_plan_get(
    request: Request, api_key: Optional[str] = None, key_hash: Optional[str] = None
):
    kh = _resolve_key_hash(api_key, key_hash)
    if not kh:
        return JSONResponse({"detail": "Provide api_key or key_hash"}, status_code=400)

    pk = _plan_pk(kh)
    resp = _ddb.get_item(TableName=_quota_table(), Key={"pk": {"S": pk}}, ConsistentRead=False)
    item = resp.get("Item")
    if not item:
        plan, limit = _plan_get_limit(kh)
        updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return AdminPlanResponse(
            key_hash=kh, pk=pk, plan=plan, daily_limit=int(limit), updated_at=updated_at
        )

    plan = item.get("plan", {}).get("S", "pro")
    daily_limit = int(
        float(item.get("daily_limit", {}).get("N", str(_quota_daily_limit_default())))
    )
    updated_at = item.get("updated_at", {}).get("S", "")

    return AdminPlanResponse(
        key_hash=kh, pk=pk, plan=plan, daily_limit=daily_limit, updated_at=updated_at
    )


# -------------------------
# Usage endpoint (non-billable)
# -------------------------
@app.get("/usage")
def usage(request: Request):
    key_hash = getattr(request.state, "key_hash", None)
    if not key_hash:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    plan, limit = _plan_get_limit(key_hash)
    pk = _quota_pk(key_hash)

    used = 0
    try:
        resp = _ddb.get_item(TableName=_quota_table(), Key={"pk": {"S": pk}}, ConsistentRead=False)
        item = resp.get("Item")
        if item and "n" in item and "N" in item["n"]:
            used = int(float(item["n"]["N"]))
    except Exception as e:
        print(f"USAGE_DDB_WARN — {e}")

    resets_epoch = _next_midnight_utc_epoch(extra_minutes=10)
    resets_at = datetime.fromtimestamp(resets_epoch, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    remaining = max(0, int(limit) - int(used))

    return {
        "plan": plan,
        "daily_limit": int(limit),
        "used_today": int(used),
        "remaining": int(remaining),
        "resets_at": resets_at,
        "resets_at_epoch": int(resets_epoch),
        "quota_pk": pk,
        "date_utc": _utc_yyyymmdd(),
    }


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
        "Note: decision is produced by the deployed rule-based engine; this block exposes inputs for transparency."
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
    symbol_filter = asset.replace("-", "_")
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
        return JSONResponse(status_code=503, content={"detail": "Data unavailable for asset " + asset})


@app.get("/signal/explain", response_model=SignalResponse)
def get_signal_explain(asset: str = "BTC-USD", mode: Literal["price_only", "combined"] = "combined"):
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
    annualization: int = 8760,
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

        ret = df["close"].pct_change().fillna(0.0)

        sig = df[signal_col].fillna(0).astype(float).clip(-1, 1)
        pos = sig.shift(1).fillna(0.0)

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

        trade_idx = df.index[trade == 1.0]
        trades = int(len(trade_idx))

        wins = int((strat_ret > 0).sum())
        win_rate = float(wins / max(1, int((pos != 0).sum())))

        bh_equity = (1.0 + ret).cumprod()
        bh_total = float(bh_equity.iloc[-1] - 1.0)
        bh_mdd = _max_drawdown(bh_equity)
        bh_end = float(bh_equity.iloc[-1])

        pts = []
        tail = equity.tail(200)
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
        return JSONResponse(status_code=503, content={"detail": "Backtest unavailable for asset " + asset})


handler = Mangum(app)
