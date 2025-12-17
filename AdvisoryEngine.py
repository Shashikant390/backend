# sh_app/advisory_engine.py
"""
Advisory & Insights Engine
Contains:
 - snapshot/trend/advice building helpers (pure functions)
 - get_or_refresh_latest() which ensures latest process_result (calls local /process-or-refresh if stale)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import os
import requests
import logging

from dateutil import parser as date_parser

import config
import repos  # your DB wrapper module


LOG = logging.getLogger("advisory_engine")

# Configuration defaults (tweak in config.py)
# Max age (days) before we try to refresh via process-or-refresh
PROCESS_STALE_DAYS = getattr(config, "PROCESS_STALE_DAYS", 5)
# Local HTTP call timeout when calling process-or-refresh
PROCESS_OR_REFRESH_TIMEOUT = getattr(config, "PROCESS_OR_REFRESH_TIMEOUT", 120)
# Local service host/port to call our own endpoints (fallback: environment)
LOCAL_HOST = os.environ.get("LOCAL_HOST", "127.0.0.1")
LOCAL_PORT = os.environ.get("PORT", os.environ.get("FLASK_RUN_PORT", "5000"))
PROCESS_OR_REFRESH_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}/process-or-refresh"


# ---------- Small helpers (copied/compatible with your earlier helpers) ----------
def _extract_scene_dt_from_pr(pr: Dict) -> Optional[datetime]:
    if not isinstance(pr, dict):
        return None
    ts_str = pr.get("end_ts") or pr.get("start_ts")
    prov = pr.get("provenance") or {}
    if not ts_str:
        ts_str = prov.get("used_end") or prov.get("used_start")
    if not ts_str:
        return None
    try:
        return date_parser.isoparse(ts_str)
    except Exception:
        try:
            # final fallback
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            return None


def _extract_ndvi_stats_from_pr(pr: Dict) -> Dict[str, Optional[float]]:
    if not isinstance(pr, dict):
        pr = {}
    indices_stats = pr.get("indices_stats") or {}
    if not indices_stats and isinstance(pr.get("result"), dict):
        indices_stats = pr["result"].get("indices_stats") or {}
    ndvi = indices_stats.get("ndvi") or {}
    def _f(k):
        v = ndvi.get(k)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    return {
        "min": _f("min"),
        "max": _f("max"),
        "mean": _f("mean"),
        "median": _f("median"),
        "p10": _f("p10"),
        "p90": _f("p90"),
        "std": _f("std"),
    }


def _human_age_from_dt(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    days = diff.days
    if days < 0:
        return "in future"
    if days == 0:
        hrs = diff.seconds // 3600
        if hrs == 0:
            mins = diff.seconds // 60
            return f"{mins} min ago" if mins > 0 else "just now"
        return f"{hrs} h ago"
    if days == 1:
        return "1 day ago"
    if days < 7:
        return f"{days} days ago"
    weeks = days // 7
    return f"{weeks} weeks ago"


def _normalize_crop(meta: Optional[dict]) -> Dict[str, Any]:
    raw_name = None
    if isinstance(meta, dict):
        raw_name = meta.get("crop_name") or meta.get("crop") or meta.get("cropType") or meta.get("crop_name_local")
    if not raw_name:
        return {"name": None, "type": "OTHER", "label": "Unknown crop"}
    name_lower = str(raw_name).lower()
    if "wheat" in name_lower or "गहू" in name_lower:
        return {"name": raw_name, "type": "WHEAT", "label": "Wheat"}
    if "rice" in name_lower or "धान" in name_lower or "paddy" in name_lower:
        return {"name": raw_name, "type": "RICE", "label": "Rice"}
    return {"name": raw_name, "type": "OTHER", "label": raw_name}


def _classify_health_bucket(h: Optional[float]) -> Dict[str, str]:
    if h is None:
        return {"bucket": "unknown", "label": "No data", "message": "No health score available."}
    if h < 30:
        return {"bucket": "critical", "label": "Critical", "message": "Crop health looks poor. Inspect urgently."}
    if h < 60:
        return {"bucket": "warning", "label": "Moderate", "message": "Some stress visible. Check irrigation/nutrients."}
    return {"bucket": "good", "label": "Good", "message": "Crop health looks generally good. Keep monitoring."}


def _ndvi_level_from_mean(m: Optional[float]) -> Dict[str, str]:
    if m is None:
        return {"level": "unknown", "label": "No NDVI", "message": "NDVI not available."}
    if m < 0.2:
        return {"level": "very_low", "label": "Very low", "message": "Very low NDVI (bare/very sparse)."}
    if m < 0.4:
        return {"level": "low", "label": "Low", "message": "Low NDVI (weak canopy)."}
    if m < 0.6:
        return {"level": "medium", "label": "Medium", "message": "Moderate NDVI."}
    if m < 0.8:
        return {"level": "high", "label": "High", "message": "High NDVI (dense canopy)."}
    return {"level": "very_high", "label": "Very high", "message": "Very high NDVI."}


# ---------- Public snapshot/trend/advice builders ----------
def build_current_snapshot(farm: Dict, latest_pr: Dict) -> Dict[str, Any]:
    meta = farm.get("meta") or {}
    crop = _normalize_crop(meta)
    scene_dt = _extract_scene_dt_from_pr(latest_pr)
    scene_dt_iso = scene_dt.isoformat() if scene_dt else None
    scene_age = _human_age_from_dt(scene_dt)
    # health_score top-level preferred
    hs = latest_pr.get("health_score")
    if hs is None:
        hs = (latest_pr.get("result") or {}).get("health_score")
    try:
        hs = float(hs) if hs is not None else None
    except Exception:
        hs = None
    ndvi_stats = _extract_ndvi_stats_from_pr(latest_pr)
    ndvi_mean = ndvi_stats.get("mean")
    health_bucket = _classify_health_bucket(hs)
    ndvi_level = _ndvi_level_from_mean(ndvi_mean)
    return {
        "farm_id": farm.get("id"),
        "source": latest_pr.get("source"),
        "scene_id": latest_pr.get("scene_id"),
        "scene_datetime": scene_dt_iso,
        "scene_age_human": scene_age,
        "health_score": hs,
        "health_bucket": health_bucket,
        "ndvi_stats": ndvi_stats,
        "ndvi_level": ndvi_level,
        "crop": crop,
    }


def analyze_trend(history_prs: List[Dict]) -> Dict[str, Any]:
    # history_prs: newest-first (repo returns that). We'll build chronological series oldest->newest
    if not history_prs:
        return {"has_history": False, "points": [], "overall": "no_data", "message": "No history."}
    series = []
    for pr in reversed(history_prs):
        ts = _extract_scene_dt_from_pr(pr)
        if not ts:
            continue
        res = pr.get("result") or {}
        hs = pr.get("health_score", res.get("health_score"))
        ndvi_mean = _extract_ndvi_stats_from_pr(pr).get("mean")
        series.append({"ts": ts, "health_score": hs, "ndvi_mean": ndvi_mean})
    if not series:
        return {"has_history": False, "points": [], "overall": "no_data", "message": "No usable timestamps in history."}
    if len(series) == 1:
        p = series[0]
        return {
            "has_history": True,
            "points": [{"timestamp": p["ts"].isoformat(), "health_score": p["health_score"], "ndvi_mean": p["ndvi_mean"]}],
            "overall": "not_enough_data",
            "message": "Only one observation; need more for trend analysis."
        }
    # slope calculation (first -> last / (n-1))
    def slope(vals):
        vals_f = [float(v) for v in vals]
        return (vals_f[-1] - vals_f[0]) / (len(vals_f) - 1)
    hs_vals = [p["health_score"] for p in series if p["health_score"] is not None]
    ndvi_vals = [p["ndvi_mean"] for p in series if p["ndvi_mean"] is not None]
    hs_s = slope(hs_vals) if len(hs_vals) > 1 else None
    nd_s = slope(ndvi_vals) if len(ndvi_vals) > 1 else None
    def classify(s, label):
        if s is None:
            return {"direction": "unknown", "label": "Unknown", "detail": f"Not enough {label} data."}
        if s > 5:
            return {"direction": "improving", "label": "Improving", "detail": f"{label} improving."}
        if s < -5:
            return {"direction": "declining", "label": "Declining", "detail": f"{label} declining."}
        if abs(s) < 2:
            return {"direction": "stable", "label": "Stable", "detail": f"{label} roughly stable."}
        return {"direction": "variable", "label": "Variable", "detail": f"{label} fluctuating."}
    hs_tr = classify(hs_s, "Health")
    nd_tr = classify(nd_s, "NDVI")
    overall = hs_tr["direction"] if hs_tr["direction"] != "unknown" else nd_tr["direction"]
    points = [{"timestamp": p["ts"].isoformat(), "health_score": p["health_score"], "ndvi_mean": p["ndvi_mean"]} for p in series]
    overall_msg = "Trend unknown."
    if overall == "improving":
        overall_msg = "Crop vegetation looks to be improving."
    elif overall == "declining":
        overall_msg = "Recent data shows a decline; inspect the field."
    elif overall == "stable":
        overall_msg = "Overall trend is stable."
    return {
        "has_history": True,
        "points": points,
        "health_trend": {"slope_per_obs": hs_s, **hs_tr},
        "ndvi_trend": {"slope_per_obs": nd_s, **nd_tr},
        "overall": overall,
        "message": overall_msg
    }


def build_crop_advice(crop_info: Dict[str, Any],
                      health_bucket: Dict[str, Any],
                      ndvi_level: Dict[str, Any],
                      trend_block: Dict[str, Any]) -> Dict[str, Any]:
    severity = health_bucket.get("bucket")
    ndvi_level_key = ndvi_level.get("level")
    trend_dir = trend_block.get("overall")
    messages: List[str] = []
    actions: List[str] = []
    # severity messages
    if severity == "critical":
        messages.append("Overall crop condition looks critical in this satellite pass.")
        actions.append("Inspect the field urgently for irrigation/pest/disease issues.")
    elif severity == "warning":
        messages.append("Crop condition is moderate; some areas may be underperforming.")
        actions.append("Check irrigation and nutrients in low NDVI zones.")
    else:
        messages.append("Crop condition appears generally good in this pass.")
        actions.append("Continue current management and monitor for sudden changes.")
    # NDVI nuance
    if ndvi_level_key in ("very_low", "low"):
        messages.append("NDVI is low — check seeding, irrigation and nutrient management.")
        actions.append("Scout low NDVI patches; consider targeted interventions.")
    elif ndvi_level_key in ("high", "very_high"):
        messages.append("NDVI is high — canopy appears dense and healthy.")
        actions.append("Maintain irrigation and nutrient regime.")
    # trend
    if trend_dir == "improving":
        messages.append("Trend shows improvement over previous passes.")
        actions.append("Current practice effective; continue and monitor.")
    elif trend_dir == "declining":
        messages.append("Trend indicates decline vs earlier passes.")
        actions.append("Prioritize scouting the recently declined zones.")
    # crop specific
    if crop_info.get("type") == "WHEAT":
        actions.append("For wheat: ensure timely nitrogen top dressing where required.")
    elif crop_info.get("type") == "RICE":
        actions.append("For rice: check standing water levels and fungal disease symptoms.")
    else:
        actions.append("Use the farm map to inspect low NDVI patches first.")
    # dedupe maintain order
    seen = set()
    def dedupe(seq):
        out = []
        for s in seq:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return {"severity": severity, "summary_messages": dedupe(messages), "recommended_actions": dedupe(actions)}


# ---------- Orchestration: ensure latest data ----------

def _pr_age_days(latest_pr: Dict) -> Optional[float]:
    """Return age in days since scene end (float) or None."""
    dt = _extract_scene_dt_from_pr(latest_pr)
    if not dt:
        # fallback to top-level end_ts string
        end_ts = latest_pr.get("end_ts") or (latest_pr.get("provenance") or {}).get("used_end")
        if end_ts:
            try:
                dt = date_parser.isoparse(end_ts)
            except Exception:
                pass
    if not dt:
        return None
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() / 86400.0


def get_or_refresh_latest(farm_id: int, force_refresh: bool = False) -> Dict:
    """
    Returns the latest process_result dict for farm_id.
    Behavior:
      - If DB has a recent result (age <= PROCESS_STALE_DAYS) and not force_refresh -> return it.
      - Otherwise call local /process-or-refresh (HTTP POST) to obtain fresh result (or cached).
      - If refresh fails, return latest DB result (if any) with a 'fallback' flag.
    """
    # 1) get latest from DB
    latest = None
    try:
        latest = repos.get_latest_process_result(farm_id)
    except Exception as e:
        LOG.exception("get_or_refresh_latest: DB fetch failed for farm %s", farm_id)

    if not force_refresh and latest:
        try:
            age_days = _pr_age_days(latest)
            if age_days is None:
                LOG.debug("get_or_refresh_latest: could not determine age; proceeding to use DB result unless forced")
            else:
                LOG.debug("get_or_refresh_latest: latest age_days=%.2f", age_days)
                if age_days <= PROCESS_STALE_DAYS:
                    return latest
        except Exception:
            LOG.exception("get_or_refresh_latest: age check failed (ignored)")

    # 2) Attempt to call local process-or-refresh endpoint
    try:
        resp = requests.post(PROCESS_OR_REFRESH_URL, json={"farm_id": int(farm_id)}, timeout=PROCESS_OR_REFRESH_TIMEOUT)
        if resp.status_code == 200:
            j = resp.json()
            # ensure we return dict in standardized shape
            if isinstance(j, dict):
                return j
        else:
            LOG.warning("get_or_refresh_latest: process-or-refresh returned %s: %s", resp.status_code, getattr(resp, "text", None))
    except Exception:
        LOG.exception("get_or_refresh_latest: calling process-or-refresh failed")

    # 3) fallback: return DB result if present
    if latest:
        # attach a small note
        latest.setdefault("meta", {})
        latest["meta"]["fallback_from"] = "db_cached"
        return latest

    # 4) nothing to return
    raise RuntimeError("No process_result available and refresh failed.")
