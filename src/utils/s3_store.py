import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3

_s3 = boto3.client("s3")


def _bucket() -> str:
    b = os.getenv("SIGNALS_BUCKET")
    if not b:
        raise RuntimeError("SIGNALS_BUCKET env var is not set")
    return b


def _key(asset: str, mode: str) -> str:
    asset_norm = asset.replace("/", "_").replace("-", "_")
    return f"signals/latest/{asset_norm}/{mode}.json"



def read_latest_signal(asset: str, mode: str) -> Optional[Dict[str, Any]]:
    try:
        obj = _s3.get_object(Bucket=_bucket(), Key=_key(asset, mode))
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except _s3.exceptions.NoSuchKey:
        print("DEBUG — cache object does not exist")
        return None
    except Exception as e:
        print(f"ERROR — S3 read failed: {e}")
        return None


def write_latest_signal(asset: str, mode: str, payload: Dict[str, Any]) -> None:
    try:
        _s3.put_object(
            Bucket=_bucket(),
            Key=_key(asset, mode),
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        print(f"DEBUG — S3 write OK: {_key(asset, mode)}")
    except Exception as e:
        print(f"ERROR — S3 write failed: {e}")
