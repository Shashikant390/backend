# utils.py
import math
import socket
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
from PIL import Image
import os
import logging
import uuid
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

def dns_lookup(host: str = "services.sentinel-hub.com") -> tuple[bool, list[str] | str]:
    try:
        addrs = socket.getaddrinfo(host, 443)
        return True, sorted({a[4][0] for a in addrs})
    except Exception as e:
        return False, str(e)


def request_with_retries(method: str, url: str, headers=None, json_payload=None, data=None,
                         timeout=30, max_retries=4, backoff_factor=1.0) -> requests.Response:
    attempt = 0
    while True:
        try:
            if method.upper() == "POST":
                r = requests.post(url, headers=headers, json=json_payload, data=data, timeout=timeout)
            elif method.upper() == "GET":
                r = requests.get(url, headers=headers, params=json_payload, timeout=timeout)
            else:
                raise RuntimeError("Unsupported method")

            if 500 <= r.status_code < 600:
                attempt += 1
                if attempt > max_retries:
                    return r
                wait = backoff_factor * (2 ** (attempt - 1))
                # at top logger already exists
                # inside request_with_retries replace prints:
                logger.warning(f"[retry] {url} -> HTTP {r.status_code}. sleeping {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            return r
        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"[retry-exc] {url} -> {repr(e)}. sleeping {wait}s (attempt {attempt})")

            time.sleep(wait)
            continue

def polygon_to_bbox(geojson: dict) -> list[float]:
    """
    Support Polygon and MultiPolygon. Returns [minx, miny, maxx, maxy].
    """
    try:
        t = geojson.get("type", "").lower()
        coords = geojson.get("coordinates", [])
        if t == "polygon":
            rings = coords
        elif t == "multipolygon":
            # MultiPolygon -> list of polygons -> take first polygon's exterior ring
            if not coords:
                raise ValueErroStart("Invalid MultiPolygon")
            # flatten first polygon's first ring
            rings = coords[0][0] if isinstance(coords[0], list) and coords[0] else coords[0]
        else:
            # fallback: try to find any numeric coordinates
            # flatten lists until we find a coordinate pair
            def _find_ring(x):
                if not isinstance(x, list):
                    return None
                # dive until we find [ [lon,lat], ... ]
                if x and isinstance(x[0], (list, tuple)) and isinstance(x[0][0], (int, float)):
                    return x
                for i in x:
                    r = _find_ring(i)
                    if r:
                        return r
                return None
            rings = _find_ring(coords) or []
        if not rings:
            raise ValueError("No coordinates found")
        # rings might be [ [lon,lat], ... ] or nested further; pick first ring if needed
        if isinstance(rings[0][0], (list, tuple)):
            ring = rings[0]
        else:
            ring = rings
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        raise ValueError("Invalid GeoJSON polygon/multipolygon")


def parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def array_to_png_bytes_uint8(arr: np.ndarray, vmin=None, vmax=None) -> bytes:
    if vmin is None:
        vmin = np.nanpercentile(arr, 2) if np.isfinite(arr).any() else 0.0
    if vmax is None:
        vmax = np.nanpercentile(arr, 98) if np.isfinite(arr).any() else 1.0
    if vmax - vmin == 0:
        vmax = vmin + 1e-6
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[np.isnan(arr)] = 0.0
    rgb = (np.dstack([scaled, scaled, scaled]) * 255.0).astype("uint8")
    im = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def ndvi_to_png_bytes(arr: np.ndarray) -> bytes:
    scaled = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    rgb = (np.dstack([scaled, scaled, scaled]) * 255.0).astype("uint8")
    im = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def png_bytes_to_ndvi_arr(png_bytes: bytes) -> np.ndarray:
    im = Image.open(BytesIO(png_bytes)).convert("L")
    arr = np.array(im).astype(np.float32)
    arr = (arr / 255.0) * 2.0 - 1.0
    return arr


def compute_stats(arr: np.ndarray) -> dict:
    res: dict = {}
    if arr is None or not np.isfinite(arr).any():
        res["note"] = "no_valid_pixels"
        return res
    res["mean"] = float(np.nanmean(arr))
    res["median"] = float(np.nanmedian(arr))
    res["std"] = float(np.nanstd(arr))
    res["min"] = float(np.nanmin(arr))
    res["max"] = float(np.nanmax(arr))
    res["p10"] = float(np.nanpercentile(arr, 10))
    res["p90"] = float(np.nanpercentile(arr, 90))
    return res

# --- S3 helpers (append to utils.py) ---


try:
    import boto3
except Exception:
    boto3 = None

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "").strip("/")

_s3_client = None

def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        if boto3 is None:
            raise RuntimeError("boto3 not installed. run `pip install boto3`")
        cfg = Config(region_name=AWS_REGION or None, retries={"max_attempts": 3})
        _s3_client = boto3.client("s3", region_name=AWS_REGION or None, config=cfg)
    return _s3_client

def make_s3_key(prefix: str, filename_hint: str = None) -> str:
    """
    Build a mostly-unique S3 key. prefix example: "ndvi_png" or "process-results/ndvi".
    Returns key WITHOUT bucket (use returned key with upload_bytes_to_s3/generate_presigned_url).
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:12]
    fname = (filename_hint or uid).replace(" ", "_")
    # ensure prefix is clean
    p = prefix.strip("/ ")
    key = f"{p}/{ts}_{uid}_{fname}"
    if S3_PREFIX:
        key = f"{S3_PREFIX.rstrip('/')}/{key}"
    return key.lstrip("/")

def upload_bytes_to_s3(data_bytes: bytes, key: str, content_type: str = "application/octet-stream", acl: str | None = None) -> str:
    """
    Upload raw bytes to S3 and return the object key used.
    key may be full or relative; function will prepend configured S3_PREFIX if set.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not configured in environment")

    client = _get_s3_client()
    put_kwargs = {"Bucket": S3_BUCKET, "Key": key, "Body": data_bytes, "ContentType": content_type}
    if acl:
        put_kwargs["ACL"] = acl
    try:
        client.put_object(**put_kwargs)
        logger.info("upload_bytes_to_s3 ok key=%s", key)
    except ClientError as e:
        logger.exception("upload_bytes_to_s3 failed for key=%s", key)
        raise
    return key

def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """
    Return a presigned GET URL for the given object key. Expires in seconds.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not configured in environment")

    client = _get_s3_client()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=int(expires_in)
        )
        return url
    except ClientError:
        logger.exception("generate_presigned_url failed for key=%s", key)
        raise

# End of S3 helpers
