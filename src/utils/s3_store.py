import json
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")


def cache_bucket() -> str:
    b = os.getenv("SIGNALS_BUCKET")
    if not b:
        raise RuntimeError("SIGNALS_BUCKET env var is not set")
    return b


def cache_key(asset: str, mode: str) -> str:
    # normalize to be safe for S3 key paths
    asset_norm = asset.replace("/", "_").replace("-", "_")
    return f"signals/latest/{asset_norm}/{mode}.json"


def read_latest_signal(asset: str, mode: str) -> Optional[Dict[str, Any]]:
    try:
        obj = _s3.get_object(Bucket=cache_bucket(), Key=cache_key(asset, mode))
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # Common "not found" codes for S3
        if code in ("NoSuchKey", "NoSuchBucket", "404", "NotFound"):
            print(f"DEBUG — cache object missing ({code})")
            return None
        print(f"ERROR — S3 read failed ({code}): {e}")
        return None
    except Exception as e:
        print(f"ERROR — S3 read failed: {e}")
        return None


def write_latest_signal(asset: str, mode: str, payload: Dict[str, Any]) -> None:
    try:
        _s3.put_object(
            Bucket=cache_bucket(),
            Key=cache_key(asset, mode),
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        print(f"DEBUG — S3 write OK: {cache_key(asset, mode)}")
    except Exception as e:
        print(f"ERROR — S3 write failed: {e}")
