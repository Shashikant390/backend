from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, date
import logging
import math

from dateutil import parser as date_parser
import requests

import config
import repos

LOG = logging.getLogger("advisory_engine")

PROCESS_STALE_DAYS = getattr(config, "PROCESS_STALE_DAYS", 5)
PROCESS_OR_REFRESH_TIMEOUT = getattr(config, "PROCESS_OR_REFRESH_TIMEOUT", 120)
LOCAL_HOST = getattr(config, "LOCAL_HOST", None) or "127.0.0.1"
LOCAL_PORT = getattr(config, "LOCAL_PORT", None) or ("5000")
PROCESS_OR_REFRESH_URL = getattr(config, "PROCESS_OR_REFRESH_URL", None) or f"http://{LOCAL_HOST}:{LOCAL_PORT}/process-or-refresh"

if getattr(config, "PROCESS_OR_REFRESH_URL", None):
    PROCESS_OR_REFRESH_URL = config.PROCESS_OR_REFRESH_URL
else:
    # local-only fallback
    PROCESS_OR_REFRESH_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}/process-or-refresh"


_DEFAULT_ADVISORY = {
    "weights": {
        "ndvi_drop": 0.35,
        "ndvi_variance": 0.20,
        "health_drop": 0.25,
        "weather": 0.20,
    },
    "thresholds": {"info": 0.35, "warning": 0.65},
   
    "ndvi_drop_ref": 0.15,       
    "variance_ref": 0.25,        
    "health_drop_ref": 20.0,     
    "cloud_cover_max_pct": 40,
    "min_history_points": 2,
    "patch_ndvi_abs_thresh": 0.35,  
    "patch_fraction_for_action": 0.15,
}

ADVISORY_CONFIG = getattr(config, "ADVISORY_CONFIG", _DEFAULT_ADVISORY)

CROP_STAGE_TABLE = {
    "WHEAT": [("emergence", 0, 20), ("vegetative", 21, 60), ("reproductive", 61, 110), ("maturity", 111, 160)],
    "RICE": [("nursery", 0, 20), ("vegetative", 21, 50), ("reproductive", 51, 90), ("maturity", 91, 140)],
    "MAIZE": [("emergence", 0, 18), ("vegetative", 19, 60), ("reproductive", 61, 100), ("maturity", 101, 150)],
    "SORGHUM": [("emergence",0,20),("vegetative",21,60),("reproductive",61,110),("maturity",111,150)],
    "Bajra": [("emergence",0,18),("vegetative",19,55),("reproductive",56,95),("maturity",96,140)],
    "BARLEY": [("emergence",0,20),("vegetative",21,55),("reproductive",56,95),("maturity",96,140)],
    "SOYBEAN": [("emergence",0,15),("vegetative",16,50),("reproductive",51,95),("maturity",96,140)],
    "GROUNDNUT": [("emergence",0,18),("vegetative",19,50),("reproductive",51,90),("maturity",91,140)],
    "COTTON": [("emergence",0,20),("vegetative",21,60),("flowering",61,120),("maturity",121,180)],
    "SUGARCANE": [("establishment",0,60),("tillering",61,180),("grand_growth",181,900)],
    "MUSTARD": [("emergence",0,20),("vegetative",21,60),("reproductive",61,100),("maturity",101,150)],
    "POTATO": [("emergence",0,30),("tuberization",31,70),("maturity",71,120)],
    "ONION": [("establishment",0,40),("bulbing",41,100),("maturity",101,150)],
    "TOMATO": [("establishment",0,30),("vegetative",31,70),("fruiting",71,120),("maturity",121,160)],
    "CHICKPEA": [("emergence",0,20),("vegetative",21,60),("reproductive",61,110),("maturity",111,150)],
    "PIGEONPEA": [("establishment",0,30),("vegetative",31,80),("reproductive",81,150)],
    "LENTIL": [("emergence",0,20),("vegetative",21,50),("reproductive",51,90),("maturity",91,130)],
    "SUNFLOWER": [("emergence",0,20),("vegetative",21,60),("flowering",61,100),("maturity",101,140)],
    "SUGARBET": [("emergence",0,30),("vegetative",31,90),("maturity",91,160)],
    "MILLET": [("emergence",0,20),("vegetative",21,60),("reproductive",61,100),("maturity",101,140)],
    "RAPESEED": [("emergence",0,20),("vegetative",21,60),("reproductive",61,100),("maturity",101,140)],
    "MAIZE_SWEET": [("emergence",0,18),("vegetative",19,50),("reproductive",51,90),("maturity",91,130)],
    "CABBAGE": [("establishment",0,30),("vegetative",31,80),("maturity",81,130)],
    "CAULIFLOWER": [("establishment",0,30),("vegetative",31,80),("maturity",81,130)],
    "GARLIC": [("establishment",0,30),("bulbing",31,90),("maturity",91,180)],
    "PEANUT": [("emergence",0,18),("vegetative",19,50),("reproductive",51,90),("maturity",91,140)],
    "BERSEEM": [("establishment",0,20),("vegetative",21,50),("maturity",51,80)],
}


STAGE_SENSITIVITY = {
    "emergence": 0.4,
    "establishment": 0.4,
    "nursery": 0.4,
    "vegetative": 1.0,
    "tillering": 1.0,
    "reproductive": 1.2,
    "flowering": 1.2,
    "fruiting": 1.2,
    "tuberization": 1.0,
    "maturity": 0.3,
    "grand_growth": 0.9,
}

# ------------------- small helpers -------------------

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


# small numeric helpers

def clamp(x: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if x is None:
        return None
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b
    except Exception:
        return default


# ---------------- spatial metrics ----------------

def compute_spatial_metrics(ndvi_array: Optional[List[float]] = None, ndvi_stats: Optional[Dict] = None) -> Dict[str, Any]:
    """Return spatial proxies. If ndvi_array (1D flatten) is given - compute directly,
    otherwise use ndvi_stats (p10/p90/std/mean).
    Output keys: p10,p90,mean,std,spread,variance_norm,prop_low_estimate
    """
    out = {
        "p10": None, "p90": None, "mean": None, "std": None,
        "spread": None, "variance_norm": None, "prop_low_estimate": None,
        "largest_patch_frac": None
    }
    cfg = ADVISORY_CONFIG
    if ndvi_array:
        try:
            import numpy as _np
            arr = _np.asarray(ndvi_array).ravel()
            arr = arr[~_np.isnan(arr)]
            if arr.size:
                out["mean"] = float(_np.mean(arr))
                out["p10"], out["p90"] = [float(v) for v in _np.percentile(arr, [10, 90])]
                out["std"] = float(_np.std(arr))
                out["spread"] = out["p90"] - out["p10"]
                out["variance_norm"] = clamp(out["spread"] / cfg["variance_ref"]) if out["spread"] is not None else None
                # proportion below absolute threshold
                prop_low = float((arr < cfg.get("patch_ndvi_abs_thresh", 0.35)).sum()) / arr.size
                out["prop_low_estimate"] = prop_low
                # note: largest_patch_frac requires connected component analysis on 2D mask; not done here
        except Exception:
            LOG.exception("compute_spatial_metrics: failed to compute from array")
    else:
        if ndvi_stats:
            out["mean"] = ndvi_stats.get("mean")
            out["p10"] = ndvi_stats.get("p10")
            out["p90"] = ndvi_stats.get("p90")
            out["std"] = ndvi_stats.get("std")
            if out["p10"] is not None and out["p90"] is not None:
                out["spread"] = out["p90"] - out["p10"]
                out["variance_norm"] = clamp(out["spread"] / cfg["variance_ref"]) if out["spread"] is not None else None
            # coarse prop low estimate using distribution approximation
            try:
                if out["mean"] is not None and out["std"] is not None:
                    # using Chebyshev-ish approximate: proportion below mean - 0.5*std
                    thresh = cfg.get("patch_ndvi_abs_thresh", 0.35)
                    # estimate z = (thresh - mean) / std -> approximate using normal CDF
                    import math as _math
                    if out["std"] > 0:
                        z = (thresh - out["mean"]) / out["std"]
                        # normal cdf approx (error function)
                        prop_low = 0.5 * (1 + _math.erf(z / _math.sqrt(2)))
                        out["prop_low_estimate"] = float(max(0.0, min(1.0, prop_low)))
            except Exception:
                LOG.debug("compute_spatial_metrics: prop_low_estimate failed")
    return out


# ---------------- temporal metrics ----------------

def compute_temporal_metrics(history_prs: List[Dict]) -> Dict[str, Any]:
    """Compute simple temporal scores: slope (per-day), recency score, history_count.
    Slope normalized as decline severity (positive = decline severity normalized 0..1).
    """
    out = {"slope_per_day": None, "slope_norm": None, "recency_days": None, "recency_score": None, "history_count": 0}
    if not history_prs:
        return out
    # build chronological list
    series = []
    for pr in reversed(history_prs):
        ts = _extract_scene_dt_from_pr(pr)
        if not ts:
            continue
        nd = _extract_ndvi_stats_from_pr(pr).get("mean")
        if nd is None:
            continue
        series.append((ts, float(nd)))
    out["history_count"] = len(series)
    if not series:
        return out
    if len(series) == 1:
        last_dt = series[-1][0]
        out["recency_days"] = (datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)).days
        out["recency_score"] = clamp(1.0 - (out["recency_days"] / max(1.0, PROCESS_STALE_DAYS)))
        return out
    # compute linear slope (ndvi units per day) using first and last
    first_dt, first_v = series[0]
    last_dt, last_v = series[-1]
    delta_days = max(1.0, (last_dt - first_dt).total_seconds() / 86400.0)
    slope = (last_v - first_v) / delta_days
    out["slope_per_day"] = slope
    # normalize slope: decline negative -> positive severity
    trend_ref = ADVISORY_CONFIG.get("trend_slope_ref", 0.005)  # NDVI units per day
    out["slope_norm"] = clamp(max(0.0, -slope) / trend_ref)
    out["recency_days"] = (datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)).days
    out["recency_score"] = clamp(1.0 - (out["recency_days"] / max(1.0, PROCESS_STALE_DAYS)))
    return out


# ---------------- weather/context processing ----------------

def compute_weather_norms(weather_block: Optional[Dict], crop_info: Optional[Dict]) -> Dict[str, Optional[float]]:
    """Return normalized weather signals: rain_deficit_norm, temp_norm, combined weather_norm (0..1)
    weather_block expected keys: rain_7d, rain_30d, avg_temp_5d, max_temp_3d
    crop_info may include expected_rain_7d and opt_temp
    """
    if not weather_block:
        return {"rain_deficit_norm": None, "temp_norm": None, "weather_norm": None}
    expected_7d = (crop_info or {}).get("expected_rain_7d") or 10.0
    opt_temp = (crop_info or {}).get("opt_temp") or 25.0
    rain_7d = weather_block.get("rain_7d")
    avg_temp_5d = weather_block.get("avg_temp_5d")
    rain_deficit_norm = None
    temp_norm = None
    if rain_7d is not None:
        # deficit normalized 0..1
        if expected_7d <= 0:
            rain_deficit_norm = 0.0
        else:
            rain_deficit_norm = clamp(max(0.0, (expected_7d - rain_7d) / expected_7d))
    if avg_temp_5d is not None:
        temp_norm = clamp(max(0.0, (avg_temp_5d - opt_temp) / 7.0))
    # combine with weights
    comps = []
    w = []
    if rain_deficit_norm is not None:
        comps.append(rain_deficit_norm); w.append(0.6)
    if temp_norm is not None:
        comps.append(temp_norm); w.append(0.4)
    weather_norm = None
    if comps and sum(w) > 0:
        weather_norm = clamp(sum(c * ww for c, ww in zip(comps, w)) / sum(w))
    return {"rain_deficit_norm": rain_deficit_norm, "temp_norm": temp_norm, "weather_norm": weather_norm}


# ---------------- signal extraction ----------------

def compute_signals(latest_pr: Dict, history_prs: List[Dict], weather_block: Optional[Dict], crop_info: Optional[Dict]) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """Return (signals, meta)
    signals: normalized contributions in 0..1 or None
    meta: diagnostic values for explanation
    """
    meta: Dict[str, Any] = {}
    signals: Dict[str, Optional[float]] = {"ndvi_drop": None, "ndvi_variance": None, "health_drop": None, "weather": None, "temporal_slope": None, "recency": None, "prop_low": None}

    # spatial
    ndvi_stats = _extract_ndvi_stats_from_pr(latest_pr)
    ndvi_array = None
    # optional: if process_result includes flattened NDVI array (rare) path or list
    res = latest_pr.get("result") or {}
    if isinstance(res.get("ndvi_array"), (list, tuple)):
        ndvi_array = res.get("ndvi_array")
    spatial = compute_spatial_metrics(ndvi_array=ndvi_array, ndvi_stats=ndvi_stats)
    meta.update({"spatial": spatial})

    # baseline: median of recent history (exclude current)
    baseline_vals = []
    for pr in (history_prs or [])[:8]:
        v = _extract_ndvi_stats_from_pr(pr).get("mean")
        if v is not None:
            baseline_vals.append(v)
    baseline_ndvi = None
    if baseline_vals:
        # median-like
        baseline_vals_sorted = sorted(baseline_vals)
        mid = len(baseline_vals_sorted) // 2
        baseline_ndvi = float(baseline_vals_sorted[mid]) if len(baseline_vals_sorted) % 2 == 1 else float((baseline_vals_sorted[mid - 1] + baseline_vals_sorted[mid]) / 2.0)
    cur_ndvi = ndvi_stats.get("mean")
    meta["baseline_ndvi"] = baseline_ndvi
    meta["cur_ndvi"] = cur_ndvi

    # ndvi drop
    if baseline_ndvi is not None and cur_ndvi is not None:
        ndvi_drop = max(0.0, baseline_ndvi - cur_ndvi)
        signals["ndvi_drop"] = clamp(ndvi_drop / ADVISORY_CONFIG["ndvi_drop_ref"])
        meta["ndvi_drop_abs"] = ndvi_drop

    # ndvi variance
    if spatial.get("spread") is not None:
        signals["ndvi_variance"] = clamp(spatial.get("variance_norm"))
    meta["spread"] = spatial.get("spread")

    # proportion low patches
    signals["prop_low"] = clamp(spatial.get("prop_low_estimate")) if spatial.get("prop_low_estimate") is not None else None

    # health drop (previous vs current)
    prev_health = None
    for pr in (history_prs or [])[:6]:
        h = pr.get("health_score") or (pr.get("result") or {}).get("health_score")
        if h is not None:
            prev_health = float(h); break
    cur_health = latest_pr.get("health_score") or (latest_pr.get("result") or {}).get("health_score")
    if prev_health is not None and cur_health is not None:
        try:
            hd = max(0.0, float(prev_health) - float(cur_health))
            signals["health_drop"] = clamp(hd / ADVISORY_CONFIG["health_drop_ref"])
            meta["health_drop_abs"] = hd
        except Exception:
            LOG.debug("health drop compute failed")

    # temporal
    temporal = compute_temporal_metrics(history_prs)
    signals["temporal_slope"] = temporal.get("slope_norm")
    signals["recency"] = temporal.get("recency_score")
    meta["temporal"] = temporal

    # weather
    weather_norms = compute_weather_norms(weather_block or {}, crop_info or {})
    signals["weather"] = weather_norms.get("weather_norm")
    meta["weather_block"] = weather_block
    meta["weather_norms"] = weather_norms

    return signals, meta


# ---------------- scoring, classification, confidence ----------------

def compute_stress_score(signals: Dict[str, Optional[float]]) -> float:
    """Weighted average of available signals. Missing signals reduce denominator.
    Returns 0..1 score.
    """
    weights = ADVISORY_CONFIG.get("weights", {})
    numerator = 0.0
    denom = 0.0
    for name, weight in weights.items():
        val = signals.get(name)
        if val is None:
            # if NDVI variance missing, we may use prop_low as partial substitute
            if name == "ndvi_variance" and signals.get("prop_low") is not None:
                v = signals.get("prop_low")
                numerator += weight * v
                denom += weight
            else:
                continue
        else:
            numerator += weight * float(val)
            denom += weight
    # fallback: consider temporal slope if no other signals
    if denom == 0:
        # try temporal or recency
        fallback = signals.get("temporal_slope") or signals.get("recency") or 0.0
        return float(clamp(fallback))
    raw = numerator / denom
    return float(clamp(raw))


def classify_severity(score: float) -> str:
    t = ADVISORY_CONFIG.get("thresholds", {"info": 0.35, "warning": 0.65})
    if score < t["info"]:
        return "INFO"
    if score < t["warning"]:
        return "WARNING"
    return "CRITICAL"


def estimate_confidence(signals: Dict[str, Optional[float]], latest_pr: Dict, meta: Dict) -> Tuple[str, float]:
    """Estimate confidence (LOW/MEDIUM/HIGH) and the numeric confidence score 0..1.
    Factors: coverage (fraction of signals present), data_quality (clouds), temporal recency, history_count.
    """
    expected_signal_names = set(ADVISORY_CONFIG.get("weights", {}).keys())
    present = 0
    for n in expected_signal_names:
        if signals.get(n) is not None:
            present += 1
        else:
            # allow prop_low to substitute for variance
            if n == "ndvi_variance" and signals.get("prop_low") is not None:
                present += 1
    coverage = present / max(1, len(expected_signal_names))

    # cloud cover check
    cloud_pct = None
    prov = latest_pr.get("provenance") or {}
    cloud_pct = prov.get("cloud_pct") or (latest_pr.get("result") or {}).get("cloud_pct")
    data_quality = 1.0
    if cloud_pct is not None:
        try:
            if float(cloud_pct) > ADVISORY_CONFIG.get("cloud_cover_max_pct", 40):
                data_quality = 0.5
        except Exception:
            pass

    # recency & history
    temporal = meta.get("temporal") or {}
    recency = temporal.get("recency_score") or 0.0
    history_count = temporal.get("history_count") or 0
    temporal_factor = clamp((recency * 0.6) + (min(history_count, 6) / 6.0 * 0.4))

    confidence_score = clamp(coverage * data_quality * temporal_factor)
    if confidence_score > 0.8:
        return "HIGH", confidence_score
    if confidence_score > 0.5:
        return "MEDIUM", confidence_score
    return "LOW", confidence_score


# ---------------- cause inference & explanation ----------------

def infer_causes(signals: Dict[str, Optional[float]], meta: Dict, crop_info: Optional[Dict]) -> List[str]:
    """Infers likely causes (short list)."""
    causes: List[str] = []
    ndvi_drop = signals.get("ndvi_drop")
    weather = signals.get("weather")
    var = signals.get("ndvi_variance") or signals.get("prop_low")
    temporal_slope = signals.get("temporal_slope")

    # strong water-driver
    if ndvi_drop and ndvi_drop > 0.25 and weather and weather > 0.3 and (meta.get("weather_norms", {}).get("rain_deficit_norm") or 0) > 0.2:
        causes.append("Likely water stress (NDVI drop with recent rainfall deficit).")

    # heat
    if weather and (meta.get("weather_norms", {}).get("temp_norm") or 0) > 0.4:
        causes.append("Possible heat stress (high recent temperatures).")

    # patchy disease/pests
    if ndvi_drop and ndvi_drop > 0.15 and var and var > 0.35:
        causes.append("Patchy decline — could be pest/disease or irrigation non-uniformity.")

    # uniform decline
    if ndvi_drop and ndvi_drop > 0.15 and (var or 0) < 0.15:
        causes.append("Uniform decline across the field — consider water/nutrient deficit.")

    # temporal confirmation
    if temporal_slope and temporal_slope > 0.5:
        causes.append("Decline confirmed by multi-pass trend.")

    if not causes:
        causes.append("General vegetation stress signal; requires scouting to determine cause.")
    return causes


def build_explanation(signals: Dict[str, Optional[float]], meta: Dict, score: float, severity: str, confidence: str, confidence_score: float) -> Dict[str, Any]:
    """Return explanation dict with triggers, top_contributors, causes, confidence_rationale."""
    triggers: List[str] = []
    top_contributors: List[Dict[str, Any]] = []

    # triggers
    if meta.get("ndvi_drop_abs") is not None:
        triggers.append(f"NDVI drop of {meta['ndvi_drop_abs']:.3f} vs recent baseline")
    s = signals
    if s.get("ndvi_variance") is not None and s.get("ndvi_variance") > 0.2:
        triggers.append("High NDVI spread (patchiness) detected")
    if s.get("health_drop") is not None and s.get("health_drop") > 0.2:
        triggers.append("Health score decline observed")
    if s.get("weather") is not None and s.get("weather") > 0.2:
        triggers.append("Recent weather anomaly likely contributing")

    # top contributors (by contribution = weight*value)
    weights = ADVISORY_CONFIG.get("weights", {})
    contribs = []
    for name, val in s.items():
        if val is None:
            continue
        w = weights.get(name, 0.0)
        contribs.append((name, float(val) * float(w)))
    contribs.sort(key=lambda x: x[1], reverse=True)
    for name, contrib in contribs[:4]:
        top_contributors.append({"name": name, "contribution": round(contrib, 4), "value": round(float(s.get(name) or 0.0), 4)})

    causes = infer_causes(s, meta, None)

    confidence_rationale = {
        "coverage": sum(1 for k in weights.keys() if s.get(k) is not None) / max(1, len(weights)),
        "data_quality": "low_clouds" if ((latest_pr := meta.get("latest_pr")) and ((latest_pr.get("provenance") or {}).get("cloud_pct") or 0) > ADVISORY_CONFIG.get("cloud_cover_max_pct", 40)) else "ok",
        "temporal_recency_days": meta.get("temporal", {}).get("recency_days") if meta.get("temporal") else None,
    }

    return {
        "triggers": triggers,
        "top_contributors": top_contributors,
        "causes": causes,
        "confidence_rationale": confidence_rationale,
    }


# ---------------- actions & recommendations ----------------
def recommend_actions(severity: str, causes: List[str], crop_type: Optional[str], crop_stage: Optional[str]) -> List[str]:
    """
    Enhanced recommend_actions:
     - Keeps same signature.
     - Infers likely drivers from `causes` (keywords like water, rain, heat, pest, disease, patchy, uniform).
     - Provides crop-specific actionable hints for ~30 crops.
     - Adjusts recommended items by severity and crop_stage.
     - Preserves original dedupe and order behavior.
    """
    actions: List[str] = []

    # normalize helpers
    def has_keyword(keywords: List[str]) -> bool:
        if not causes:
            return False
        ctext = " ".join(causes).lower()
        return any(k in ctext for k in keywords)

    c = (crop_type or "").upper() if crop_type else ""
    stg = (crop_stage or "").lower() if crop_stage else ""

    # 1) Base actions by severity
    if severity == "CRITICAL":
        actions.extend([
            "Inspect the field urgently (walk the field) to identify cause.",
            "Check irrigation system and water availability immediately.",
            "If pest/disease suspected, collect samples and consult extension before treatment."
        ])
    elif severity == "WARNING":
        actions.extend([
            "Scout suspicious/low-NDVI patches (note coordinates using the farm map).",
            "Check irrigation uniformity and recent fertilizer applications.",
            "Monitor closely and rescan with next satellite pass; prioritize scouting high-risk zones."
        ])
    else:
        # INFO severity does NOT always mean "do nothing"
        ndvi_related = any(
            tag in causes for tag in [
                "moderate_ndvi_drop",
                "severe_ndvi_drop",
                "large_low_vigor_area",
                "high_patchiness",
                "consistent_decline_trend",
                "rapid_health_decline",
                "weather_moderate_stress",
                "weather_extreme_stress"
            ]
        )
    
        if not ndvi_related:
            actions.append("No immediate action; continue routine monitoring.")

    # 2) Driver-based short-circuit suggestions (water / heat / pest / nutrient / patchy)
    water_driver = has_keyword(["water", "rain", "drought", "dry", "irrigat"])
    heat_driver = has_keyword(["heat", "hot", "temperature", "heat stress"])
    pest_driver = has_keyword(["pest", "insect", "locust", "bollworm", "borer"])
    disease_driver = has_keyword(["disease", "blast", "blight", "rust", "fung", "sheath"])
    patchy_driver = has_keyword(["patch", "patchy", "heterogen", "non-uniform", "patchiness"])
    nutrient_driver = has_keyword(["nitrogen", "nitro", "phosphor", "potash", "fertil"])
    uniform_decline = has_keyword(["uniform", "uniform decline", "uniformly"])

    # 3) Crop-specific advice templates (30+ crops)
    # For each crop, we add best-practice bullet(s). Keep conservative, avoid exact dosing unless stage-specific,
    # but provide general actionable guidance. Use stg to tune suggestions (e.g., vegetative -> N sidedress)
    crop_map = {
        "WHEAT": [
            "Wheat: check nitrogen status — if vegetative, consider split N top-dressing after scouting.",
            "Look for rust/leaf blotch; sample and treat per extension if confirmed."
        ],
        "RICE": [
            "Rice: verify standing water levels; maintain 2–5 cm water at tillering/vegetative stages.",
            "If humid and patchy decline, scout for sheath blight or blast; sample and consult extension."
        ],
        "MAIZE": [
            "Maize: check for nitrogen deficiency (striping) and consider sidedress N in vegetative stage.",
            "Check for stem/leaf borers if patchy damage observed."
        ],
        "MAIZE_SWEET": [
            "Sweet maize: sidedress N during early vegetative if low vigour; scout for pests."
        ],
        "SORGHUM": [
            "Sorghum: check moisture; ensure timely top-dress if vegetative and deficient."
        ],
        "BAJRA": [
            "Pearl millet (Bajra): verify moisture; foliar stress can indicate drought—irrigate if deficit."
        ],
        "BARLEY": [
            "Barley: check N status during tillering; scout for fungal disease in humid conditions."
        ],
        "SOYBEAN": [
            "Soybean: patchy stress may indicate pests (pod borers) or uneven emergence—scout and sample.",
            "Avoid unnecessary N — soy fixes nitrogen; focus on inoculant and S/P if pod set poor."
        ],
        "GROUNDNUT": ["Groundnut: check pod development and soil moisture; consider gypsum/calcium if pod filling issues."],
        "PEANUT": ["Groundnut/Peanut: check pod filling and irrigation; avoid late N heavy application."],
        "COTTON": [
            "Cotton: check for bollworm/armyworm in patchy areas; ensure balanced N and K during growth."
        ],
        "SUGARCANE": [
            "Sugarcane: check irrigation and fertilizer schedule; for establishment/tillering ensure water availability."
        ],
        "MUSTARD": [
            "Mustard/Rapeseed: check nutrient status (S and N) during vegetative stage; guard against aphids."
        ],
        "RAPESEED": [
            "Rapeseed: monitor for S deficiency and water stress during flowering and pod set."
        ],
        "POTATO": [
            "Potato: patchy NDVI can indicate tuber/seed issue or late blight; scout and sample for disease."
        ],
        "ONION": [
            "Onion: patchy emergence may be seeding/soil issue; check for thrips in bulb-forming stage."
        ],
        "TOMATO": [
            "Tomato: check irrigation uniformity; patchy stress often indicates soil moisture or pest pressure."
        ],
        "CHICKPEA": [
            "Chickpea: assess moisture during reproductive stage; drought at pod fill reduces yield."
        ],
        "PIGEONPEA": [
            "Pigeonpea: watch for water stress during reproductive stage; scout for pod borers."
        ],
        "LENTIL": ["Lentil: check for moisture stress during flowering/setting; sample for foliar disease if patchy."],
        "SUNFLOWER": ["Sunflower: check for moisture and insect damage in patchy areas; manage nutrition."],
        "SUGARBET": ["Sugarbeet: check for early-season emergence issues; ensure nutrients and moisture."],
        "MILLET": ["Millet: drought-tolerant but check moisture in reproductive stage; consider irrigation if severe."],
        "CABBAGE": ["Cabbage: patchy NDVI may indicate pest/disease; scout for caterpillars or nutrient issues."],
        "CAULIFLOWER": ["Cauliflower: uniform canopy needed for head formation; scout low NDVI patches early."],
        "GARLIC": ["Garlic: check bulb formation stage; ensure balanced nutrition and moisture."],
        "BERSEEM": ["Berseem/fodder: moisture & cutting schedule influence NDVI; irrigate if yield-critical."],
        "SUGARCANE": ["Sugarcane: manage irrigation & N splits; for ratoon check stubble health."],
        # add other synonyms mapping
        "OTHER": ["Use the farm map to inspect low NDVI patches first and sample for specific diagnosis."]
    }

    # Add crop-specific items (if available)
    crop_specific = crop_map.get(c, None) or crop_map.get(crop_type.split()[0] if crop_type else "", None)
    if crop_specific:
        for item in crop_specific:
            actions.append(item)

    # 4) Stage-aware tuning
    if stg:
        if stg in ("vegetative", "tillering", "emergence"):
            actions.append("Stage note: vegetative stage — nutrients (esp. N) and irrigation impact vigour strongly; consider targeted nutrient top-up after scouting.")
        if stg in ("reproductive", "flowering", "fruiting", "tuberization"):
            actions.append("Stage note: reproductive stage — avoid heavy N; prioritize irrigation and pest/disease scouting to protect yield.")
        if stg in ("maturity",):
            actions.append("Stage note: maturity — NDVI decline may be natural; avoid aggressive interventions unless clear damage found.")

    # 5) Driver-specific concrete suggestions
    # water-related
    if water_driver:
        actions.insert(0, "Driver detected: probable water stress — inspect irrigation schedule and soil moisture; irrigate small test blocks if needed.")
    # heat-related
    if heat_driver:
        actions.append("Heat driver: irrigate during cooler hours (evening/early morning), consider mulching or shade if feasible.")
    # pest/disease
    if pest_driver or disease_driver:
        actions.append("Pest/Disease driver: scout immediately and collect samples; apply targeted treatment only after confirmation from extension.")
    # patchy -> targeted actions
    if patchy_driver:
        actions.append("Patchy decline: perform zonal scouting and consider spot treatment (spot irrigation, spot pesticide/fertilizer) rather than whole-field spray.")
    # uniform decline -> whole-field actions
    if uniform_decline:
        actions.append("Uniform decline: consider field-wide interventions (irrigation, balanced nutrient application) after rapid scouting confirms cause.")
    # nutrient
    if nutrient_driver:
        actions.append("Nutrient hint: check leaf colour charts or quick soil/leaf tests; apply balanced fertilizer as per extension recommendations.")

    # 6) Severity-specific escalations (additive)
    if severity == "CRITICAL":
        actions.append("Consider calling local extension officer or agronomist for urgent field visit.")
    elif severity == "WARNING":
        actions.append("Prioritize scouting and short-term corrective measures; delay chemical treatment until confirmed.")

    # 7) Safety & confidence guardrails
    # If causes include 'requires scouting' phrase or no clear driver, recommend scouting first
    if not (water_driver or heat_driver or pest_driver or disease_driver or nutrient_driver):
        # No strong inferred driver — be conservative
        actions.append("No clear driver detected from data; prioritize visual scouting and sample collection before chemical measures.")
    # Advice about low-confidence
    # (we don't have confidence label here; rely on causes containing 'requires scouting' cue)
    if has_keyword(["requires scouting", "requires scouting to determine cause", "requires scouting to determine", "requires scouting to"]):
        actions.append("Confidence low: scout before acting and re-evaluate after next satellite pass.")

    # 8) Deduplicate while preserving order (existing behavior)
    seen = set()
    out = []
    for a in actions:
        if a and a not in seen:
            seen.add(a)
            out.append(a)

    return out

# ---------------- orchestration: generate advisory for a farm ----------------

def generate_advisory_for_farm(farm_id: int, force_refresh: bool = False, persist: bool = True) -> Dict[str, Any]:
    """Top-level orchestration. Returns advisory payload dict.
    Steps:
      - ensure latest process_result (uses existing get_or_refresh_latest)
      - fetch history, weather, crop info from repos
      - compute signals/meta
      - compute score, severity, confidence
      - build explanation & actions
      - persist advisory_history via repos.save_advisory_history
    """
    # 1) latest process_result
    try:
        latest = get_or_refresh_latest(farm_id, force_refresh=force_refresh)
    except Exception as e:
        LOG.exception("generate_advisory_for_farm: failed to fetch latest for %s", farm_id)
        raise

    # 2) fetch history (recent passes)
    try:
        history = repos.get_process_results(farm_id, limit=8) or []
    except Exception:
        LOG.exception("generate_advisory_for_farm: failed to fetch history for %s", farm_id)
        history = []

    # 3) fetch weather & crop info
    try:
        weather = None
        try:
            if hasattr(repos, "get_recent_weather_for_farm"):
                weather = repos.get_recent_weather_for_farm(farm_id)
        except Exception:
            LOG.debug("weather not available for farm %s", farm_id)

    except Exception:
        LOG.exception("get recent weather failed for %s", farm_id)
        weather = None
    try:
        try:
            farm = repos.get_farm(farm_id)
            crop_info = farm.get("meta", {}) if farm else {}
        except Exception:
            crop_info = {}

    except Exception:
        LOG.exception("get crop info failed for %s", farm_id)
        crop_info = {}

    # determine crop_stage from sowing_date if present
    sow = crop_info.get("sowing_date")
    sow_date = None
    if sow:
        try:
            # accept date or iso string
            if isinstance(sow, str):
                sow_date = date_parser.isoparse(sow).date()
            elif isinstance(sow, (datetime, date)):
                sow_date = sow if isinstance(sow, date) else sow.date()
        except Exception:
            LOG.debug("sowing date parse failed for %s", sow)
    crop_type_norm = (crop_info.get("type") or crop_info.get("crop_name") or "").upper()
    crop_stage = None
    if crop_type_norm and sow_date:
        try:
            # map to our stage table key variants e.g., 'WHEAT' present
            key = crop_type_norm
            if key not in CROP_STAGE_TABLE:
                # fallback: try simple normalized
                key = key.split()[0]
            if key in CROP_STAGE_TABLE:
                for stg, s_start, s_end in CROP_STAGE_TABLE[key]:
                    age = (date.today() - sow_date).days
                    if s_start <= age <= s_end:
                        crop_stage = stg; break
                if not crop_stage:
                    # last stage
                    crop_stage = CROP_STAGE_TABLE[key][-1][0]
        except Exception:
            LOG.debug("crop stage detection failed for %s %s", crop_type_norm, sow_date)

    # 4) compute signals & meta
    signals, meta = compute_signals(latest, history, weather, crop_info)
    meta["latest_pr"] = latest

    # 5) score fusion
    score = compute_stress_score(signals)
    severity = classify_severity(score)
    confidence_label, confidence_score = estimate_confidence(signals, latest, meta)

    # 6) explanation & actions
    explanation = build_explanation(
        signals, meta, score, severity, confidence_label, confidence_score
    )

    # ---- 🔥 ENRICH CAUSES USING SIGNALS (KEY FIX) ----
    causes = list(explanation.get("causes", []))  # copy

    ndvi_drop = signals.get("ndvi_drop", 0.0)
    variance = signals.get("ndvi_variance", 0.0)
    prop_low = signals.get("prop_low", 0.0)
    temporal_slope = signals.get("temporal_slope")
    health_drop = signals.get("health_drop", 0.0)
    weather_norm = signals.get("weather")

    # NDVI severity
    if ndvi_drop >= 0.15:
        causes.append("severe_ndvi_drop")
    elif ndvi_drop >= 0.05:
        causes.append("moderate_ndvi_drop")

    # Spatial stress
    if prop_low >= 0.6:
        causes.append("large_low_vigor_area")
    if variance >= 0.6:
        causes.append("high_patchiness")

    # Temporal trend
    if temporal_slope is not None and temporal_slope < -0.5:
        causes.append("consistent_decline_trend")

    # Health score
    if health_drop >= 15:
        causes.append("rapid_health_decline")

    # Weather stress
    if isinstance(weather_norm, (int, float)):
        if weather_norm >= 0.7:
            causes.append("weather_extreme_stress")
        elif weather_norm >= 0.4:
            causes.append("weather_moderate_stress")

    # ---- ACTIONS NOW BECOME SERIOUSNESS-AWARE ----
    actions = recommend_actions(
        severity=severity,
        causes=causes,
        crop_type=crop_type_norm,
        crop_stage=crop_stage
    )


    advisory_payload = {
        "farm_id": farm_id,
        "scene_id": latest.get("scene_id"),
        "scene_datetime": _extract_scene_dt_from_pr(latest).isoformat() if _extract_scene_dt_from_pr(latest) else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "stress_score": round(score, 4),
        "confidence": confidence_label,
        "confidence_score": round(confidence_score, 4),
        "signals": signals,
        "meta": meta,
        "explanation": explanation,
        "recommended_actions": actions,
        "crop_type": crop_type_norm,
        "crop_stage": crop_stage,
    }

    # 7) persist advisory
    if persist:
        try:
            # expected repos.save_advisory_history to accept this payload or similar
            if hasattr(repos, "save_advisory_history"):
                repos.save_advisory_history(advisory_payload)
            else:
                # fallback: generic insert
                # repos.insert_advisory_history(advisory_payload)
                pass
        except Exception:
            LOG.exception("Failed to persist advisory for farm %s", farm_id)

    return advisory_payload


# ----------------- end of advisory engine -----------------
# You can unit test helpers: compute_spatial_metrics, compute_temporal_metrics,
# compute_weather_norms, compute_signals, compute_stress_score, estimate_confidence,
# infer_causes, recommend_actions, generate_advisory_for_farm


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
    # 2) Refresh using internal logic (NO HTTP)
    try:
        from routes import process_or_refresh_farm  # or wherever you place it
        return process_or_refresh_farm(farm_id)
    except Exception:
        LOG.exception("get_or_refresh_latest: process_or_refresh_farm failed")
    
    if latest:
        latest.setdefault("meta", {})
        latest["meta"]["fallback_from"] = "db_cached"
        return latest

    raise RuntimeError("No process_result available and refresh failed.")
