import json
import os
from datetime import datetime, timezone
from typing import Any, Optional, Dict

import boto3

_s3 = boto3.client("s3")


def _bucket() -> str:
    b = os.getenv("SIGNALS_BUCKET")
    if not b:
        raise RuntimeError("SIGNALS_BUCKET env var is not set")
    return b


def _key(asset: str, mode: str) -> str:
    asset_norm = asset.replace("/", "_")
    return f"signals/latest/{asset_norm}/{mode}.json"


def read_latest_signal(asset: str, mode: str) -> Optional[Dict[str, Any]]:
    try:
        obj = _s3.get_object(Bucket=_bucket(), Key=_key(asset, mode))
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except _s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


def write_latest_signal(asset: str, mode: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("cached_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    _s3.put_object(
        Bucket=_bucket(),
        Key=_key(asset, mode),
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
