import base64
from datetime import datetime, timedelta, timezone
from rasterio.io import MemoryFile
import numpy as np
from typing import Tuple, Optional
from flask import request
from utils import compute_stats, ndvi_to_png_bytes
from auth import get_sh_token
from utils import request_with_retries
import config
from evalscripts import EVAL_S2_NDVI, EVAL_S2_NDVI_SIMPLE, EVAL_QUICK_NDVI

from flask import current_app

def inspect_tiff_bytes(tiff_bytes: bytes) -> Tuple[np.ndarray, int, dict]:
    """
    Inspect TIFF bytes returned from SH. Returns (arr, valid_pixels, profile).
    arr is the raw array read as float32 with shape (bands, H, W).
    valid_pixels is number of finite values in band 0 (NDVI)
    profile is rasterio profile dict (may be empty on error).
    """
    try:
        with MemoryFile(tiff_bytes) as mem:
            with mem.open() as src:
                arr = src.read().astype("float32")   # shape: (bands, H, W)
                profile = src.profile
        if arr.size == 0:
            print("inspect_tiff_bytes: empty arr")
            return arr, 0, profile
        ndvi = arr[0]
        finite_mask = np.isfinite(ndvi)
        total = ndvi.size
        valid = int(finite_mask.sum())
        # debugging print (to server logs)
        print("\n===== TIFF DEBUG REPORT =====")
        print("NDVI shape:", ndvi.shape)
        print("Valid pixels:", valid, "/", total)
        print("Fraction valid:", valid/total if total>0 else 0.0)
        if valid > 0:
            print("min/max/mean:", float(np.nanmin(ndvi)), float(np.nanmax(ndvi)), float(np.nanmean(ndvi)))
        else:
            print("NO VALID PIXELS — ALL NAN")
        print("Profile keys:", list(profile.keys()) if isinstance(profile, dict) else profile)
        print("==============================\n")
        return arr, valid, profile
    except Exception as e:
        # Log and re-raise to make failures obvious in server logs
        current_app.logger.exception("inspect_tiff_bytes failed: %s", str(e))
        raise

def _call_process_with_payload(payload: dict, dev_header: Optional[str] = None, timeout: int = 120):
    """
    Helper to call Sentinel-Hub Process API using your existing token & retry helper.
    Returns requests.Response (or raises).
    """
    try:
        token = get_sh_token()
    except Exception as e:
        current_app.logger.exception("Failed to get SH token")
        raise

    if not token:
        raise RuntimeError("Missing Sentinel-Hub token")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if dev_header:
        headers["X-DEV-UID"] = dev_header

    resp = request_with_retries("POST", config.PROCESS_URL, headers=headers, json_payload=payload, timeout=timeout)
    return resp

def _make_tiff_payload(geojson, collection, t_from, t_to, spatial_scene_id: Optional[str]=None, width: int = 800, height: int = 800, use_processing_mosaic: bool = True):
    """
    Build a basic PROCESS payload requesting image/tiff float result for NDVI evalscript.
    Caller will attach evalscript as needed.
    """
    data_item = {"type": collection, "dataFilter": {"timeRange": {"from": t_from, "to": t_to}}}
    if use_processing_mosaic:
        data_item["processing"] = {"mosaicking": "SIMPLE"}
    if spatial_scene_id:
        # spatialFilter is optional; include when we mean to lock to a scene
        data_item["dataFilter"]["spatialFilter"] = {"type": "scene", "value": spatial_scene_id}
    payload = {
        "input": {
            "bounds": {"geometry": geojson},
            "data": [data_item]
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]}
    }
    return payload

def process_s2(geojson, start, end):
    """
    Robust Sentinel-2 processing that:
      - tries SCL-masked NDVI first (EVAL_S2_NDVI)
      - falls back to simple NDVI (EVAL_S2_NDVI_SIMPLE) when renderer complains
      - if TIFF has no valid pixels: retries without spatialFilter, with wider windows (±1d, ±3d)
      - if still empty: tries collection fallback to 'sentinel-2-l1c' (if available)
      - final fallback: request quick-NDVI PNG (EVAL_QUICK_NDVI) and return preview
    Returns a dict similar to your existing output with format_used, indices_stats, health_score, previews_base64.
    """
    if not config.COL_S2 or (config.AVAILABLE_COLLECTIONS and config.COL_S2 not in config.AVAILABLE_COLLECTIONS):
        raise RuntimeError("process_s2: sentinel-2 collection not available in account")

    dev_header = request.headers.get("X-DEV-UID")
    # use the input start/end initially (these may be tight scene datetimes in some callers)
    t_from = start
    t_to = end

    current_collection = config.COL_S2
    tried_collections = []
    attempts = []

    # set of windows to try if we find completely-empty TIFFs.
    # Each entry is a tuple (use_spatial_lock:bool, delta_from_days, delta_to_days)
    #  - first attempt will usually be (True, 0, 0) i.e. exact timeRange passed by caller
    fallback_windows = [
        ("tight_spatial", True, timedelta(minutes=1), timedelta(minutes=1)),   # tight window (caller likely uses exact scene window)
        ("wide_nospat_1d", False, timedelta(days=1), timedelta(days=0)),        # -1..+0 day (1 day back)
        ("wide_nospat_3d", False, timedelta(days=3), timedelta(days=0)),        # -3..+0 days
        ("wide_nospat_7d", False, timedelta(days=7), timedelta(days=0)),        # -7..+0 days
    ]

    def _build_time_window(center_dt: Optional[datetime], from_delta: timedelta, to_delta: timedelta):
        if center_dt:
            f = (center_dt - from_delta).isoformat().replace("+00:00", "Z")
            t = (center_dt + to_delta).isoformat().replace("+00:00", "Z")
        else:
            # fallback: treat start/end as absolute if center unknown
            # here we simply use provided t_from/t_to expanded by deltas
            try:
                base_from = datetime.fromisoformat(t_from.replace("Z", "+00:00"))
                base_to = datetime.fromisoformat(t_to.replace("Z", "+00:00"))
                f = (base_from - from_delta).isoformat().replace("+00:00", "Z")
                t = (base_to + to_delta).isoformat().replace("+00:00", "Z")
            except Exception:
                # as absolute fallback, just use start/end unchanged
                f, t = t_from, t_to
        return f, t

    # Try series: primary evalscript (with SCL) then fallback evalscript (simple NDVI).
    evalscript_primary = EVAL_S2_NDVI
    evalscript_fallback = EVAL_S2_NDVI_SIMPLE
    evalscript_quick_png = EVAL_QUICK_NDVI

    # If 'start' and 'end' appear to be the exact scene time range, compute a center dt for windowing
    center_dt = None
    try:
        # try parse start as ISO and use its midpoint with end if both parse
        s_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        center_dt = s_dt + (e_dt - s_dt) / 2
    except Exception:
        center_dt = None

    # Main loop: try collection (current_collection) with different window strategies.
    collection_fallbacks = [current_collection]
    if "sentinel-2-l1c" in config.AVAILABLE_COLLECTIONS and "sentinel-2-l1c" not in collection_fallbacks:
        collection_fallbacks.append("sentinel-2-l1c")

    last_valid_result = None
    used_metadata = None

    for coll in collection_fallbacks:
        tried_collections.append(coll)
        # loop window attempts
        for window_name, use_spatial, from_delta, to_delta in fallback_windows:
            fwin, twin = _build_time_window(center_dt, from_delta, to_delta)
            current_app.logger.info("[process_s2] trying collection=%s window=%s from=%s to=%s spatial=%s", coll, window_name, fwin, twin, use_spatial)

            # Build payload (use spatialFilter only when use_spatial True AND caller provided spatialFilter info)
            # For spatial locking by scene, the caller of process_s2 should pass a payload that includes the scene id.
            # In your original code you may have the scene id; here we rely on caller start/end and geojson.
            # We'll not add 'spatialFilter' here so caller can opt in earlier in flow; this function focuses on window retries.
            payload = _make_tiff_payload(geojson, coll, fwin, twin, spatial_scene_id=None, width=800, height=800, use_processing_mosaic=True)
            # try primary evalscript first (SCL-aware)
            payload_primary = dict(payload)
            payload_primary["evalscript"] = evalscript_primary

            try:
                resp = _call_process_with_payload(payload_primary, dev_header=dev_header, timeout=120)
            except Exception as e:
                current_app.logger.exception("[process_s2] process call failed (primary): %s", str(e))
                # try fallback evalscript immediately if error mentions dataset / renderer (let outer logic decide)
                resp = None

            # If primary returned an error that indicates fallback, try simple evalscript
            if resp is None or (hasattr(resp, "status_code") and resp.status_code != 200):
                # check the response text if available
                text = ""
                try:
                    if resp is not None:
                        text = getattr(resp, "text", "") or ""
                except Exception:
                    text = ""
                # If renderer complains about SCL/units or dataset id -> try fallback evalscript (no SCL)
                if resp is not None and resp.status_code == 400 and ("Dataset with id" in text or "RENDERER_EXCEPTION" in text or "unsupported units" in text):
                    current_app.logger.info("[process_s2] primary render failed with renderer/dataset error, trying simple NDVI (no SCL)")
                    payload_fb = dict(payload)
                    payload_fb["evalscript"] = evalscript_fallback
                    try:
                        resp2 = _call_process_with_payload(payload_fb, dev_header=dev_header, timeout=120)
                        resp = resp2
                    except Exception:
                        current_app.logger.exception("[process_s2] fallback process call failed")
                        resp = None
                else:
                    # other error: continue to next window/collection
                    current_app.logger.warning("[process_s2] process call returned non-200 and not renderer-dataset error: %s", text[:500])
                    resp = resp  # could be None or error response

            # If we have a successful TIFF response, inspect it
            if resp is not None and getattr(resp, "status_code", None) == 200:
                try:
                    arr, valid, profile = inspect_tiff_bytes(resp.content)
                except Exception:
                    # cannot inspect -> treat as failure and move on
                    arr, valid, profile = None, 0, {}
                if valid > 0:
                    # Good result: compute stats and build return
                    ndvi = arr[0]
                    stats = {"ndvi": compute_stats(ndvi)}
                    def norm(x, lo, hi):
                        if x is None or not np.isfinite(x):
                            return None
                        return (np.clip(x, lo, hi) - lo) / (hi - lo)
                    s_ndvi = norm(stats["ndvi"].get("mean"), -0.2, 0.9)
                    health = float(np.clip(s_ndvi * 100.0, 0, 100)) if s_ndvi is not None else None
                    ndvi_b64 = base64.b64encode(ndvi_to_png_bytes(ndvi)).decode("utf-8")
                    current_app.logger.info("[process_s2] success: collection=%s window=%s valid_pixels=%d", coll, window_name, valid)
                    return {"format_used": "tiff_float32_ndvi", "indices_stats": stats, "health_score": health, "previews_base64": {"ndvi": ndvi_b64}, "provenance": {"collection": coll, "used_start": fwin, "used_end": twin}}
                else:
                    # zero valid pixels - move to next fallback (wider window / collection)
                    current_app.logger.info("[process_s2] TIFF contained 0 valid pixels for collection=%s window=%s; trying next fallback", coll, window_name)
                    attempts.append({"collection": coll, "window": window_name, "valid": valid, "profile": profile})
                    # continue loop to next window
            else:
                # resp is non-200: check if it contains useful info, otherwise continue
                if resp is not None:
                    detail = None
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = getattr(resp, "text", None)
                    current_app.logger.warning("[process_s2] Process API non-200 resp for coll=%s window=%s status=%s detail=%s", coll, window_name, getattr(resp, "status_code", None), str(detail)[:500])
                # keep trying next window

        # end windows loop for this collection -- try next collection if any

    # End of collection/window attempts: nothing produced valid pixels.
    # Final fallback: request quick NDVI uint8 PNG via EVAL_QUICK_NDVI (no TIFF)
    current_app.logger.info("[process_s2] All TIFF attempts returned zero valid pixels. Trying quick-NDVI PNG fallback.")
    try:
        payload_png = {
            "input": {
                "bounds": {"geometry": geojson},
                "data": [{"type": config.COL_S2, "dataFilter": {"timeRange": {"from": start, "to": end}}}]
            },
            "evalscript": evalscript_quick_png,
            "output": {"width": 512, "height": 512, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]}
        }
        resp_png = _call_process_with_payload(payload_png, dev_header=dev_header, timeout=120)
        if resp_png is not None and getattr(resp_png, "status_code", None) == 200:
            # return PNG preview only; cannot compute float stats from this
            png_b64 = base64.b64encode(resp_png.content).decode("utf-8")
            current_app.logger.info("[process_s2] quick-NDVI PNG fallback succeeded")
            return {"format_used": "png_quick_ndvi", "indices_stats": {"ndvi": {"note": "no_valid_pixels_in_tiff; returned_quick_png_preview"}}, "health_score": None, "previews_base64": {"ndvi": png_b64}, "provenance": {"collection": config.COL_S2, "used_start": start, "used_end": end}}
    except Exception:
        current_app.logger.exception("[process_s2] quick-NDVI PNG fallback failed")

    # Nothing worked — raise a clear error (caller can wrap)
    current_app.logger.error("[process_s2] All attempts failed; returning no_valid_pixels")
    return {"format_used": "tiff_float32_ndvi", "indices_stats": {"ndvi": {"note": "no_valid_pixels"}}, "health_score": None, "previews_base64": {"ndvi": base64.b64encode(np.zeros((10,10), dtype=np.uint8)).decode("utf-8")}}
