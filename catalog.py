# catalog.py
from datetime import datetime, timezone
import math
import numpy as np
import logging


import config
from config import CATALOG_URL, PROCESS_URL, COL_S2, AVAILABLE_COLLECTIONS
from auth import get_sh_token
from utils import request_with_retries, parse_iso, png_bytes_to_ndvi_arr, polygon_to_bbox
import cache

LOG = logging.getLogger("sh_app.catalog")


def refresh_available_collections() -> list:
    """
    Populate config.AVAILABLE_COLLECTIONS from the STAC /collections list.
    Safe to call at startup; failures are swallowed (best-effort).
    Returns the list of collection ids found (may be empty).
    """
    try:
        j = list_collections()
        cols = [c.get("id") for c in j.get("collections", []) if c.get("id")]
        if cols:
            config.AVAILABLE_COLLECTIONS = cols
            LOG.info("Populated config.AVAILABLE_COLLECTIONS: %s", cols)
        return cols
    except Exception as e:
        LOG.exception("refresh_available_collections failed: %s", e)
        return []


def list_collections() -> dict:
    """
    Return the raw /collections JSON from the Sentinel-Hub catalog.
    Raises on HTTP errors.
    """
    token = get_sh_token()
    url = CATALOG_URL.replace("/search", "/collections")
    headers = {"Authorization": f"Bearer {token}" if token else "", "Accept": "application/json"}
    r = request_with_retries("GET", url, headers=headers, timeout=30)
    try:
        r.raise_for_status()
    except Exception:
        LOG.error("LIST COLLECTIONS FAILED: %s %s", r.status_code, getattr(r, "text", None))
        raise
    j = r.json()
    # small debug log
    LOG.debug("list_collections: found %d collections", len(j.get("collections", [])))
    return j


def catalog_search(collection: str, bbox: list, start: str, end: str, limit: int = 10, max_cloud: int | None = None,
                   cache_ttl: int | None = None) -> dict:
    """
    Query catalog /search for the given collection and bbox/time window.
    Uses Redis cache (cache.make_key). Returns parsed JSON from SH catalog.
    Only supports collection provided (we expect sentinel-2-l2a).
    """
    if collection is None:
        raise RuntimeError("catalog_search: collection is None (not available in account)")

    # If AVAILABLE_COLLECTIONS is populated, validate the requested collection
    if AVAILABLE_COLLECTIONS and collection not in AVAILABLE_COLLECTIONS:
        raise RuntimeError(f"catalog_search: illegal collection '{collection}' for this account. Available: {AVAILABLE_COLLECTIONS}")

    # Build cache key
    key_payload = {"bbox": bbox, "start": start, "end": end, "limit": limit, "max_cloud": max_cloud}
    key = cache.make_key("catalog", collection, key_payload)

    # Return cached if present
    cached = cache.get_json(key)
    if cached is not None:
        LOG.debug("catalog_search: cache hit key=%s", key)
        return cached

    token = get_sh_token()
    headers = {"Authorization": f"Bearer {token}" if token else "", "Content-Type": "application/json", "Accept": "application/geo+json"}

    payload = {
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "collections": [collection],
        "limit": int(limit)
    }
    # apply cloud filter for S2 only if requested
    if max_cloud is not None:
        payload["filter"] = f"eo:cloud_cover<{int(max_cloud)}"

    r = request_with_retries("POST", CATALOG_URL, headers=headers, json_payload=payload, timeout=60)
    if r.status_code != 200:
        LOG.error("CATALOG SEARCH FAILED: %s %s", r.status_code, getattr(r, "text", None))
        r.raise_for_status()

    j = r.json()

    # cache the raw catalog results (small JSON)
    try:
        ttl = cache_ttl if cache_ttl is not None else 3600 * 6
        cache.set_json(key, j, ttl=ttl)
    except Exception:
        LOG.exception("catalog_search: failed to set cache (ignored)")

    LOG.debug("catalog_search: returned %d features for collection=%s", len(j.get("features", [])), collection)
    return j


def quick_ndvi_mean(bbox: list, start: str, end: str, cache_ttl: int | None = None) -> float | None:
    """
    Quick NDVI mean used for scoring scenes. Produces a low-res PNG from SH using EVAL_QUICK_NDVI
    and returns the numeric mean NDVI value (float) or None on failure.
    Results are cached per bbox/start/end.
    """
    # lazy import to avoid import-time issues / circular imports
    try:
        from evalscripts import EVAL_QUICK_NDVI
    except Exception:
        LOG.exception("quick_ndvi_mean: failed to import EVAL_QUICK_NDVI from evalscripts")
        return None

    if not COL_S2:
        LOG.warning("quick_ndvi_mean: COL_S2 not configured")
        return None
    if AVAILABLE_COLLECTIONS and COL_S2 not in AVAILABLE_COLLECTIONS:
        LOG.warning("quick_ndvi_mean: COL_S2 not available in account AVAILABLE_COLLECTIONS")
        return None

    key_payload = {"bbox": bbox, "start": start, "end": end}
    key = cache.make_key("quick_ndvi", COL_S2, key_payload)

    cached = cache.get_json(key)
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            # fall through to re-compute
            LOG.debug("quick_ndvi_mean: cache value not float, recomputing")

    token = get_sh_token()
    headers = {"Authorization": f"Bearer {token}" if token else "", "Content-Type": "application/json", "Accept": "image/png"}

    payload = {
        "input": {
            "bounds": {"bbox": bbox},
            "data": [{"type": COL_S2, "dataFilter": {"timeRange": {"from": start, "to": end}}, "processing": {"mosaicking": "SIMPLE"}}]
        },
        "evalscript": EVAL_QUICK_NDVI,
        "output": {"width": 128, "height": 128, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]}
    }

    r = request_with_retries("POST", PROCESS_URL, headers=headers, json_payload=payload, timeout=60)
    if r is None or r.status_code != 200:
        LOG.error("QUICK NDVI PROCESS FAILED: %s %s", getattr(r, "status_code", None), getattr(r, "text", None))
        return None

    try:
        arr = png_bytes_to_ndvi_arr(r.content)
    except Exception:
        LOG.exception("quick_ndvi_mean: failed to decode png -> ndvi arr")
        return None

    if arr is None or not np.isfinite(arr).any():
        return None

    mean_val = float(np.nanmean(arr))

    # cache numeric mean
    try:
        ttl = cache_ttl if cache_ttl is not None else 3600 * 6
        cache.set_json(key, mean_val, ttl=ttl)
    except Exception:
        LOG.exception("quick_ndvi_mean: failed to set cache (ignored)")

    return mean_val


def score_scene(item: dict, now_dt: datetime, use_quick_ndvi: bool = False,
                bbox: list | None = None, start: str | None = None, end: str | None = None) -> tuple[float, dict]:
    """
    Score a single STAC item (scene) for preference ordering.
    Returns (score, meta) where meta contains age_days, cloud_cover, quick_ndvi_mean (if requested).
    Scoring components (tunable):
      - 50% clear_score (based on eo:cloud_cover)
      - 30% recency (age_days)
      - 20% quick_ndvi (if requested)
    """
    dt = item.get("properties", {}).get("datetime") or item.get("properties", {}).get("start_datetime")
    age_days = 9999.0
    if dt:
        try:
            age_days = (now_dt - parse_iso(dt)).total_seconds() / 86400.0
        except Exception:
            pass
    age_score = math.exp(-max(0.0, age_days) / 14.0)

    cloud = item.get("properties", {}).get("eo:cloud_cover")
    clear_score = 1.0 if cloud is None else max(0.0, min(1.0, 1.0 - float(cloud) / 100.0))

    ndvi_score = 0.5
    quick_mean_val = None
    if use_quick_ndvi and bbox and start and end and (cloud is None or cloud <= 80):
        try:
            m = quick_ndvi_mean(bbox, start, end)
            if m is not None:
                lo, hi = -0.2, 0.9
                ndvi_score = (max(lo, min(hi, m)) - lo) / (hi - lo)
                quick_mean_val = float(m)
        except Exception:
            LOG.exception("score_scene: quick_ndvi_mean failed (ignored)")

    score = 0.5 * clear_score + 0.3 * age_score + 0.2 * ndvi_score
    meta = {"age_days": age_days, "cloud_cover": cloud, "quick_ndvi_mean": quick_mean_val if use_quick_ndvi else None}
    return float(score), meta
