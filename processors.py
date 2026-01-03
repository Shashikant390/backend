import base64
import io
from datetime import datetime, timedelta, timezone
from rasterio.io import MemoryFile
import numpy as np
from PIL import Image  # Required for colorization
from typing import Tuple, Optional
from flask import request, current_app

from utils import compute_stats, ndvi_to_png_bytes  # We will override the usage of ndvi_to_png_bytes
from auth import get_sh_token
from utils import request_with_retries
import config
from evalscripts import EVAL_S2_NDVI, EVAL_S2_NDVI_SIMPLE, EVAL_QUICK_NDVI


# ==========================================
# 🎨 NEW: Local Colorization Helper
# ==========================================
def ndvi_to_colored_png(ndvi_arr: np.ndarray) -> bytes:
    """
    Converts a raw float32 NDVI array (-1.0 to 1.0) into a COLORED PNG byte stream.
    Matches the colors in your JS Evalscript.
    """
    # 1. Create an empty RGB image (Height, Width, 3)
    h, w = ndvi_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. Define Masks based on your thresholds
    # < 0.0 : Grey [128, 128, 128]
    mask_grey = (ndvi_arr < 0.0)
    rgb[mask_grey] = [128, 128, 128]

    # 0.0 - 0.2 : Red [255, 0, 0]
    mask_red = (ndvi_arr >= 0.0) & (ndvi_arr < 0.2)
    rgb[mask_red] = [255, 0, 0]

    # 0.2 - 0.4 : Orange [255, 166, 0]
    mask_orange = (ndvi_arr >= 0.2) & (ndvi_arr < 0.4)
    rgb[mask_orange] = [255, 166, 0]

    # 0.4 - 0.6 : Yellow [255, 255, 0]
    mask_yellow = (ndvi_arr >= 0.4) & (ndvi_arr < 0.6)
    rgb[mask_yellow] = [255, 255, 0]

    # 0.6 - 0.8 : Light Green [51, 204, 51]
    mask_lgreen = (ndvi_arr >= 0.6) & (ndvi_arr < 0.8)
    rgb[mask_lgreen] = [51, 204, 51]

    # >= 0.8 : Dark Green [0, 255, 0]
    mask_dgreen = (ndvi_arr >= 0.8)
    rgb[mask_dgreen] = [0, 255, 0]
    
    # Handle NaNs (Transparent or Black) - let's make them transparent
    # We need RGBA for transparency
    rgba = np.dstack((rgb, np.full((h, w), 255, dtype=np.uint8))) # Add Alpha channel
    mask_nan = np.isnan(ndvi_arr)
    rgba[mask_nan] = [0, 0, 0, 0] # Fully transparent

    # 3. Convert to PNG bytes
    img = Image.fromarray(rgba, 'RGBA')
    with io.BytesIO() as bio:
        img.save(bio, format='PNG')
        return bio.getvalue()


def inspect_tiff_bytes(tiff_bytes: bytes) -> Tuple[np.ndarray, int, dict]:
    """
    Inspect TIFF bytes returned from SH. Returns (arr, valid_pixels, profile).
    arr is the raw array read as float32 with shape (bands, H, W).
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
        
        # Log summary
        # print(f"TIFF Inspect: Valid {valid}/{total}") 
        return arr, valid, profile
    except Exception as e:
        current_app.logger.exception("inspect_tiff_bytes failed: %s", str(e))
        raise

def _call_process_with_payload(payload: dict, dev_header: Optional[str] = None, timeout: int = 120):
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
    data_item = {"type": collection, "dataFilter": {"timeRange": {"from": t_from, "to": t_to}}}
    if use_processing_mosaic:
        data_item["processing"] = {"mosaicking": "SIMPLE"}
    if spatial_scene_id:
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
    if not config.COL_S2 or (config.AVAILABLE_COLLECTIONS and config.COL_S2 not in config.AVAILABLE_COLLECTIONS):
        raise RuntimeError("process_s2: sentinel-2 collection not available in account")

    dev_header = request.headers.get("X-DEV-UID")
    t_from = start
    t_to = end

    current_collection = config.COL_S2
    tried_collections = []
    attempts = []

    fallback_windows = [
        ("tight_spatial", True, timedelta(minutes=1), timedelta(minutes=1)),
        ("wide_nospat_1d", False, timedelta(days=1), timedelta(days=0)),
        ("wide_nospat_3d", False, timedelta(days=3), timedelta(days=0)),
        ("wide_nospat_7d", False, timedelta(days=7), timedelta(days=0)),
    ]

    def _build_time_window(center_dt: Optional[datetime], from_delta: timedelta, to_delta: timedelta):
        if center_dt:
            f = (center_dt - from_delta).isoformat().replace("+00:00", "Z")
            t = (center_dt + to_delta).isoformat().replace("+00:00", "Z")
        else:
            try:
                base_from = datetime.fromisoformat(t_from.replace("Z", "+00:00"))
                base_to = datetime.fromisoformat(t_to.replace("Z", "+00:00"))
                f = (base_from - from_delta).isoformat().replace("+00:00", "Z")
                t = (base_to + to_delta).isoformat().replace("+00:00", "Z")
            except Exception:
                f, t = t_from, t_to
        return f, t

    evalscript_primary = EVAL_S2_NDVI
    evalscript_fallback = EVAL_S2_NDVI_SIMPLE
    evalscript_quick_png = EVAL_QUICK_NDVI

    center_dt = None
    try:
        s_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        center_dt = s_dt + (e_dt - s_dt) / 2
    except Exception:
        center_dt = None

    collection_fallbacks = [current_collection]
    if "sentinel-2-l1c" in config.AVAILABLE_COLLECTIONS and "sentinel-2-l1c" not in collection_fallbacks:
        collection_fallbacks.append("sentinel-2-l1c")

    for coll in collection_fallbacks:
        tried_collections.append(coll)
        for window_name, use_spatial, from_delta, to_delta in fallback_windows:
            fwin, twin = _build_time_window(center_dt, from_delta, to_delta)
            current_app.logger.info("[process_s2] trying collection=%s window=%s from=%s to=%s", coll, window_name, fwin, twin)

            payload = _make_tiff_payload(geojson, coll, fwin, twin, spatial_scene_id=None, width=800, height=800, use_processing_mosaic=True)
            payload_primary = dict(payload)
            payload_primary["evalscript"] = evalscript_primary

            try:
                resp = _call_process_with_payload(payload_primary, dev_header=dev_header, timeout=120)
            except Exception as e:
                current_app.logger.exception("[process_s2] process call failed (primary): %s", str(e))
                resp = None

            if resp is None or (hasattr(resp, "status_code") and resp.status_code != 200):
                # Fallback logic for simple evalscript
                if resp is not None and resp.status_code == 400: # Simplified check
                    current_app.logger.info("[process_s2] trying simple NDVI (no SCL)")
                    payload_fb = dict(payload)
                    payload_fb["evalscript"] = evalscript_fallback
                    try:
                        resp = _call_process_with_payload(payload_fb, dev_header=dev_header, timeout=120)
                    except Exception:
                        resp = None

            if resp is not None and getattr(resp, "status_code", None) == 200:
                try:
                    arr, valid, profile = inspect_tiff_bytes(resp.content)
                except Exception:
                    arr, valid, profile = None, 0, {}

                if valid > 0:
                    ndvi = arr[0]
                    stats = {"ndvi": compute_stats(ndvi)}
                    
                    # Compute Health Score
                    def norm(x, lo, hi):
                        if x is None or not np.isfinite(x): return None
                        return (np.clip(x, lo, hi) - lo) / (hi - lo)
                    s_ndvi = norm(stats["ndvi"].get("mean"), -0.2, 0.9)
                    health = float(np.clip(s_ndvi * 100.0, 0, 100)) if s_ndvi is not None else None

                    # ---------------------------------------------------------
                    # 🔥 THE FIX: Use our new colorizer instead of generic PNG
                    # ---------------------------------------------------------
                    colored_png_bytes = ndvi_to_colored_png(ndvi)
                    ndvi_b64 = base64.b64encode(colored_png_bytes).decode("utf-8")
                    
                    current_app.logger.info("[process_s2] success: valid_pixels=%d", valid)
                    
                    return {
                        "format_used": "tiff_float32_ndvi", 
                        "indices_stats": stats, 
                        "health_score": health, 
                        "previews_base64": {"ndvi": ndvi_b64}, 
                        "provenance": {"collection": coll, "used_start": fwin, "used_end": twin}
                    }
                else:
                    current_app.logger.info("[process_s2] TIFF 0 valid pixels, trying next...")
                    attempts.append({"collection": coll, "window": window_name})

    # Final Fallback: Quick PNG
    current_app.logger.info("[process_s2] All TIFF attempts failed. Trying quick-NDVI PNG fallback.")
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
            png_b64 = base64.b64encode(resp_png.content).decode("utf-8")
            return {
                "format_used": "png_quick_ndvi", 
                "indices_stats": {"ndvi": {"note": "preview_only"}}, 
                "health_score": None, 
                "previews_base64": {"ndvi": png_b64}, 
                "provenance": {"collection": config.COL_S2, "used_start": start, "used_end": end}
            }
    except Exception:
        pass

    current_app.logger.error("[process_s2] All attempts failed.")
    return {
        "format_used": "failed", 
        "indices_stats": {}, 
        "health_score": None, 
        "previews_base64": {"ndvi": ""}
    }