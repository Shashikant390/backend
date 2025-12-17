# routes.py (S2-only simplified)
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, current_app
import requests
import config
from utils import dns_lookup, polygon_to_bbox  # your utils
from catalog import catalog_search, list_collections, score_scene
import repos
from auth import get_current_user_or_none  # see fallback below
import cache  # import at top
import json
import traceback
import os
import logging
from datetime import datetime, timedelta, timezone
from utils import polygon_to_bbox
import boto3
from processors import  process_s2


api = Blueprint("api", __name__)

# ensure AVAILABLE_COLLECTIONS is populated on first import (best-effort)
try:
    if not config.AVAILABLE_COLLECTIONS:
        from catalog import refresh_available_collections
        refresh_available_collections()
except Exception:
    current_logger = logging.getLogger("sh_app.routes")
    current_logger.warning("Could not refresh AVAILABLE_COLLECTIONS at startup; processors will use config.COL_* defaults")


# -----------------------
# Basic health / collections
# -----------------------
@api.route("/health", methods=["GET"])
def health():
    resolved, info = dns_lookup()
    return jsonify({
        "status": "ok",
        "sh_client_id_present": bool(config.SH_CLIENT_ID and config.SH_CLIENT_ID != "<PUT_YOUR_CLIENT_ID_HERE>"),
        "sh_access_token_present": bool(config.SH_ACCESS_TOKEN),
        "dns_resolves": resolved,
        "dns_info": info
    })


@api.route("/collections", methods=["GET"])
def http_list_collections():
    try:
        j = list_collections()
        return jsonify(j), 200
    except Exception as e:
        return jsonify({"error": "list_collections_failed", "detail": str(e)}), 500


# -----------------------
# Farms: create / read / list
# -----------------------
@api.route("/farms", methods=["POST"])
def create_farm():
    """
    Dev-safe farm creation:
      - Accepts geojson as object OR string (will try to parse).
      - Logs payload.
      - Returns detailed trace when FLASK_DEBUG=1 for debugging.
    """
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    body = request.get_json(force=True, silent=True) or {}
    current_app.logger.debug("Incoming /farms payload (raw): %s", body)

    geojson = body.get("geojson")
    name = body.get("name", "Unnamed")
    meta = body.get("meta", {})

    if isinstance(geojson, str):
        try:
            parsed = json.loads(geojson)
            geojson = parsed
            current_app.logger.debug("Parsed geojson string into object")
        except Exception as e:
            current_app.logger.exception("Failed parsing geojson string")
            return jsonify({"error": "invalid_geojson_string", "detail": str(e)}), 400

    if not isinstance(geojson, dict):
        current_app.logger.warning("geojson missing or wrong type: %s", type(geojson))
        return jsonify({"error": "geojson required and must be an object (not string)"}), 400

    geo_type = geojson.get("type")
    coords = geojson.get("coordinates")
    if not geo_type or not coords:
        current_app.logger.warning("geojson missing required keys: type/coordinates - payload=%s", geojson)
        return jsonify({"error": "invalid_geojson", "detail": "missing type or coordinates"}), 400

    try:
        farm = repos.create_farm(user_id=user.id, name=name, geojson=geojson, meta=meta)
        current_app.logger.info("Created farm id=%s user_id=%s name=%s", farm.id, user.id, name)
        return jsonify({"id": farm.id}), 201
    except Exception as e:
        tb = traceback.format_exc()
        current_app.logger.exception("create_farm exception: %s", e)
        debug_on = os.environ.get("FLASK_DEBUG", "1") == "1" or current_app.config.get("DEBUG", False)
        if debug_on:
            return jsonify({"error": "create_farm_failed", "detail": str(e), "trace": tb}), 500
        else:
            return jsonify({"error": "create_farm_failed"}), 500


@api.route("/farms/<int:farm_id>", methods=["GET"])
def get_farm(farm_id: int):
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    farm = repos.get_farm(farm_id)
    if not farm:
        return jsonify({"error": "not_found"}), 404
    if farm["user_id"] != user.id:
        return jsonify({"error": "forbidden"}), 403
    return jsonify(farm), 200

@api.route("/farms", methods=["GET"])
def list_farms():
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    results = repos.list_farms(user.id)
    return jsonify(results), 200


@api.route("/catalog-score", methods=["POST"])
def catalog_score_route():
    """
    Body: { geojson, start, end, limit, max_cloud, use_quick_ndvi, farm_id (optional) }
    Stores top-N S2 scenes into scene table (upsert) if farm_id provided.
    Returns top N list to client.
    """
    body = request.get_json(force=True, silent=True) or {}

    # Required inputs
    geojson = body.get("geojson")
    start = body.get("start")
    end = body.get("end")
    if not geojson or not start or not end:
        return jsonify({"error": "geojson,start,end required"}), 400

    # Safe coercions and defaults
    try:
        limit = int(body.get("limit", 10) or 10)
    except (ValueError, TypeError):
        limit = 10
    # cap limit to prevent expensive queries
    MAX_LIMIT = 50
    if limit < 1:
        limit = 1
    if limit > MAX_LIMIT:
        limit = MAX_LIMIT

    try:
        max_cloud = float(body.get("max_cloud", 60) or 60)
    except (ValueError, TypeError):
        max_cloud = 60.0
    # clamp
    if max_cloud < 0:
        max_cloud = 0.0
    if max_cloud > 100:
        max_cloud = 100.0

    use_quick_ndvi = bool(body.get("use_quick_ndvi", False))
    farm_id_raw = body.get("farm_id")
    farm_id = None
    if farm_id_raw is not None:
        try:
            farm_id = int(farm_id_raw)
        except (ValueError, TypeError):
            current_app.logger.warning("catalog-score: invalid farm_id provided (%r); ignoring", farm_id_raw)
            farm_id = None

    # compute bbox from geojson (your existing helper)
    try:
        bbox = polygon_to_bbox(geojson)
    except Exception as e:
        current_app.logger.exception("catalog-score: polygon_to_bbox failed")
        return jsonify({"error": "invalid_geojson", "detail": str(e)}), 400

    now_dt = datetime.now(timezone.utc)
    response = {"S2": []}

    def canonical_cache_key(*parts):
        # produce deterministic key string for caching
        try:
            return cache.make_key(*[json.dumps(p, sort_keys=True, default=str) for p in parts])
        except Exception:
            # fallback: simple join
            return cache.make_key(*[str(p) for p in parts])

    def process_collection(collection_id):
        key = canonical_cache_key("catalog", collection_id, {"bbox": bbox}, {"start": start, "end": end},
                                  {"limit": limit, "max_cloud": max_cloud, "use_quick_ndvi": use_quick_ndvi})
        cached = cache.get_json(key)
        if cached is not None:
            current_app.logger.debug("catalog-score: cache hit key=%s (collection=%s)", key, collection_id)
            # cached should be a list; return it
            return cached, None

        # Query STAC / catalog
        try:
            stac = catalog_search(collection_id, bbox, start, end, limit=limit, max_cloud=max_cloud)
        except Exception as e:
            current_app.logger.exception("catalog-search failed for collection=%s", collection_id)
            return [], str(e)

        features = stac.get("features") if isinstance(stac, dict) else None
        if not isinstance(features, list):
            current_app.logger.warning("catalog-score: unexpected STAC response structure for collection=%s: %r", collection_id, stac)
            features = []

        out_list = []
        for feat in features:
            try:
                sc, meta = score_scene(feat, now_dt, use_quick_ndvi, bbox, start, end)
                it_out = {"id": feat.get("id"), "properties": feat.get("properties", {}), "score": sc, "meta": meta}
                out_list.append(it_out)
            except Exception:
                current_app.logger.exception("catalog-score: failed scoring scene id=%s", feat.get("id"))
                continue

        # sort descending by score
        out_list.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        # persist scenes (upsert) if farm_id provided
        if farm_id and out_list:
            for it_out in out_list:
                try:
                    repos.save_scene(
                        farm_id=int(farm_id),
                        scene_id=it_out["id"],
                        collection=collection_id,
                        properties=it_out.get("properties", {}),
                        score=float(it_out.get("score") or 0.0),
                        meta=it_out.get("meta")
                    )
                except Exception:
                    current_app.logger.exception("failed saving scene %s for farm %s", it_out.get("id"), farm_id)

        # store in cache (small JSON only)
        try:
            cache.set_json(key, out_list, ttl=3600 * 6)
        except Exception:
            current_app.logger.exception("failed setting catalog cache for key=%s", key)

        # cache best-scene per-farm and overall
        if farm_id and isinstance(out_list, list) and len(out_list) > 0:
            best = out_list[0]
            per_key = f"scene:best:{farm_id}:{collection_id}"
            try:
                cache.set_json(per_key, best, ttl=86400)
            except Exception:
                current_app.logger.exception("failed caching per-collection best scene %s for farm=%s", collection_id, farm_id)

            overall_key = f"scene:best:{farm_id}:overall"
            try:
                prev = cache.get_json(overall_key)
                prev_score = None
                if prev and isinstance(prev, dict):
                    prev_score = float(prev.get("scene", {}).get("score", 0.0) or 0.0)
                cur_score = float(best.get("score", 0.0) or 0.0)
                if prev is None or cur_score > prev_score:
                    cache.set_json(overall_key, {"collection": collection_id, "scene": best}, ttl=86400)
            except Exception:
                current_app.logger.exception("failed updating overall best scene cache for farm=%s", farm_id)

        return out_list, None

    # Only S2 supported here; ensure config.COL_S2 is set
    if not getattr(config, "COL_S2", None):
        return jsonify({"S2_error": "sentinel-2 not configured on server"}), 500

    lst, err = process_collection(config.COL_S2)
    response["S2"] = lst if not err else {"error": err}

    return jsonify(response), 200



# -----------------------
# Process: call processing endpoint
# ---------------------
# constants (ensure config.PROCESS_API_URL exists or fallback)





# Constants / config-driven

REQUEST_TIMEOUT = getattr(config, "PROCESS_API_TIMEOUT", 30)
CACHE_TTL = getattr(config, "CACHE_TTL_SECONDS", 86400)  # default 1 day
S3_BUCKET = getattr(config, "S3_BUCKET", None)
S3_PREFIX = getattr(config, "S3_PREFIX", "ndvi")
ENV_TAG = getattr(config, "ENV_TAG", "dev")  # used to namespace cache keys / dev/prod
COL_S2 = str(getattr(config, "COL_S2", "sentinel-2-l2a"))

def upload_to_s3(data_bytes: bytes, key: str, content_type: str = "application/octet-stream"):
    """
    Upload bytes to S3 and return public URL (or S3 URL).
    Requires S3_BUCKET and boto3 available and configured with credentials.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not configured")

    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data_bytes, ContentType=content_type)
    # construct a URL (presumes public or presigned may be needed)
    region = s3.meta.region_name
    if region:
        return f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{key}"
    return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"


# -------------------------------------------------------------------------
# /process route — uses process_s2 and farm stored in DB
# -------------------------------------------------------------------------
@api.route("/process", methods=["POST"])
def process_using_process_s2():
    """
    POST body:
      { "farm_id": <int> }
    Optional:
      { "start": "<ISO8601>", "end": "<ISO8601>" }
    Behavior:
      - load farm geometry from repos.get_farm(farm_id)
      - prefer best scene from redis for exact timeRange (scene.datetime)
      - fallback to start/end from request body
      - final fallback: last 30 days
      - call process_s2(geojson, start, end) and return the result as JSON
    """
    body = request.get_json(force=True, silent=True) or {}
    farm_id = body.get("farm_id")
    if not farm_id:
        return jsonify({"error": "farm_id is required"}), 400

    try:
        farm_id = int(farm_id)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid_farm_id"}), 400

    # 1) fetch farm from DB
    try:
        farm = repos.get_farm(farm_id)
    except Exception as e:
        current_app.logger.exception("Failed to fetch farm from DB for id=%s", farm_id)
        return jsonify({"error": "db_error", "detail": str(e)}), 500

    if not farm:
        return jsonify({"error": "farm_not_found"}), 404

    geojson = farm.get("geojson") or farm.get("geom")
    if not geojson:
        return jsonify({"error": "farm_missing_geometry"}), 400

    # 2) determine time window: prefer best scene (redis), else body start/end, else last 30 days
    start = None
    end = None
    # try best scene in redis (overall then per-collection)
    try:
        best_key = f"scene:best:{farm_id}:overall"
        current_app.logger.info("[PROCESS_S2] Looking for best scene in redis: %s", best_key)
        cached_best = cache.get_json(best_key)
        if not cached_best or not cached_best.get("scene"):
            per_key = f"scene:best:{farm_id}:{config.COL_S2}"
            cached_s2 = cache.get_json(per_key)
            if cached_s2 and isinstance(cached_s2, dict):
                cached_best = {"collection": config.COL_S2, "scene": cached_s2}
                current_app.logger.info("[PROCESS_S2] overall best missing; using per-collection best: %s", per_key)
        if cached_best and cached_best.get("scene"):
            best_scene = cached_best["scene"]
            scene_dt = best_scene.get("properties", {}).get("datetime")
            if scene_dt:
                # Use exact scene time (can use ±1 minute when calling process_s2, but process_s2 expects ISO strings)
                try:
                    dt = datetime.fromisoformat(scene_dt.replace("Z", "+00:00"))
                    # use a small window (±1 minute) as strings
                    start = (dt - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
                    end = (dt + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
                    current_app.logger.info("[PROCESS_S2] Using best scene datetime for timeRange: %s - %s", start, end)
                except Exception:
                    # if parse fails, fall back to raw scene_dt for both start and end
                    start = scene_dt
                    end = scene_dt
    except Exception:
        current_app.logger.exception("[PROCESS_S2] error reading best scene from redis (ignored)")

    # If redis didn't yield times, check request body start/end
    if not start or not end:
        req_start = body.get("start")
        req_end = body.get("end")
        if req_start and req_end:
            start = req_start
            end = req_end
            current_app.logger.info("[PROCESS_S2] Using start/end from request body: %s - %s", start, end)

    # Final fallback: last 30 days window
    if not start or not end:
        now = datetime.now(timezone.utc)
        end = now.isoformat().replace("+00:00", "Z")
        start = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")
        current_app.logger.info("[PROCESS_S2] Falling back to last 30 days: %s - %s", start, end)

    # 3) call existing process_s2 function
    try:
        # process_s2 expects geojson, start, end and returns the JSON-like dictionary with stats & preview
        out = process_s2(geojson, start, end)
        # Add provenance info so client knows what was used
        out.setdefault("provenance", {})
        out["provenance"].update({
            "farm_id": farm_id,
            "used_start": start,
            "used_end": end
        })
        return jsonify(out), 200
    except requests.exceptions.HTTPError as he:
        # process_s2 uses requests and may raise HTTPError
        resp_text = getattr(he.response, "text", str(he))
        current_app.logger.error("[PROCESS_S2] HTTPError from SH PROCESS: %s", resp_text[:2000])
        return jsonify({"error": "process_api_failed", "detail": resp_text}), 502
    except Exception as e:
        current_app.logger.exception("[PROCESS_S2] processing failed for farm_id=%s", farm_id)
        return jsonify({"error": "process_failed", "detail": str(e)}), 500

# -----------------------
# Query process results
# -----------------------
@api.route("/process-result", methods=["GET"])
def list_process_results():
    """
    Query params: ?farm_id=...&limit=10
    Returns recent process_result rows (full merged JSON per row).
    """
    farm_id = request.args.get("farm_id")
    if not farm_id:
        return jsonify({"error": "farm_id required"}), 400
    try:
        farm_id = int(farm_id)
    except Exception:
        return jsonify({"error": "invalid_farm_id"}), 400

    try:
        limit = int(request.args.get("limit", 10))
    except Exception:
        limit = 10

    try:
        rows = repos.get_latest_process_result(farm_id=farm_id)
        return jsonify(rows), 200
    except Exception as e:
        current_app.logger.exception("list_process_results failed")
        return jsonify({"error": "list_failed", "detail": str(e)}), 500





# Tunables (adjust)
RECENT_MAX_AGE_DAYS = getattr(config, "RECENT_MAX_AGE_DAYS", 5)
CATALOG_LOOKBACK_DAYS = getattr(config, "CATALOG_LOOKBACK_DAYS", 30)
CATALOG_LIMIT = getattr(config, "CATALOG_LIMIT", 10)
CATALOG_MAX_CLOUD = getattr(config, "CATALOG_MAX_CLOUD", 60.0)

def _iso_to_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _is_recent_iso(iso_ts, days=RECENT_MAX_AGE_DAYS):
    dt = _iso_to_dt(iso_ts)
    if not dt:
        return False
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=days)

def find_best_scene_from_catalog(farm_geojson, farm_id=None):
    """
    Call catalog-score (internal) for bbox built from geojson and return top scene dict or None.
    Reuses your catalog_score_route logic; adapt if your internal helper name differs.
    """
    bbox = polygon_to_bbox(farm_geojson)
    end_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    start_iso = (datetime.now(timezone.utc) - timedelta(days=CATALOG_LOOKBACK_DAYS)).isoformat().replace("+00:00", "Z")

    # Prefer to call internal function `catalog_search` or `catalog_score` directly if available (avoid HTTP)
    try:
        # If you have a helper that returns scored list, call it directly
        if "catalog_search" in globals() or "score_scene" in globals():
            # call existing catalog_score code path: use process_collection snippet (simplified)
            stac_resp = catalog_search(config.COL_S2, bbox, start_iso, end_iso, limit=CATALOG_LIMIT, max_cloud=CATALOG_MAX_CLOUD)
            feats = stac_resp.get("features", []) if isinstance(stac_resp, dict) else []
            if not feats:
                return None
            # score each feature using existing score_scene (reuse)
            now_dt = datetime.now(timezone.utc)
            scored = []
            for f in feats:
                try:
                    sc, meta = score_scene(f, now_dt, use_quick_ndvi=False, bbox=bbox, start=start_iso, end=end_iso)
                    scored.append({"id": f.get("id"), "properties": f.get("properties", {}), "score": sc, "feature": f})
                except Exception:
                    current_app.logger.exception("scoring failed for feature (ignored)")
            if not scored:
                return None
            scored.sort(key=lambda x: x["score"], reverse=True)
            best = scored[0]["feature"]
        else:
            # Fallback: call your internal http endpoint /catalog-score (local call)
            payload = {"geojson": farm_geojson, "start": start_iso, "end": end_iso, "limit": CATALOG_LIMIT, "max_cloud": CATALOG_MAX_CLOUD, "use_quick_ndvi": False, "farm_id": farm_id}
            r = requests.post(f"http://127.0.0.1:{os.environ.get('PORT',5000)}/catalog-score", json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            # expected shape: {"S2": [list of scored scenes...]}
            cand_list = j.get("S2") or []
            if isinstance(cand_list, dict) and cand_list.get("error"):
                return None
            best = cand_list[0]["feature"] if cand_list and "feature" in cand_list[0] else (cand_list[0] if cand_list else None)

        # cache best scene in redis for farm if farm_id present
        if farm_id and best:
            try:
                cache.set_json(f"scene:best:{farm_id}:overall", {"collection": config.COL_S2, "scene": best}, ttl=86400)
            except Exception:
                current_app.logger.exception("failed caching best scene (ignored)")
        return best
    except Exception as e:
        current_app.logger.exception("find_best_scene_from_catalog failed: %s", e)
        return None
    

@api.route("/process-or-refresh", methods=["POST"])
def process_or_refresh_for_farm():
    """
    Smart real-time-ish monitor for a single farm.

    POST body: { "farm_id": <int> }

    Logic:
      1) Load farm (geojson) from DB.
      2) Get latest stored process_result for this farm from Postgres.
      3) Find best scene from catalog (via find_best_scene_from_catalog) for farm bbox.
      4) Compare:
           - if latest.scene_id == best_scene.id => RETURN LATEST (no new scene).
           - else if latest_end_ts >= best_scene.datetime => RETURN LATEST.
           - else => PROCESS NEW SCENE (process_s2), save to DB, cache, and return.
      5) If catalog finds no scenes:
           - if latest exists => return latest (but it's old).
           - else => 404 no_recent_scene_found.
    """
    body = request.get_json(force=True, silent=True) or {}
    farm_id = body.get("farm_id")
    if not farm_id:
        return jsonify({"error": "farm_id required"}), 400
    try:
        farm_id = int(farm_id)
    except Exception:
        return jsonify({"error": "invalid_farm_id"}), 400

    # ------------------------------------------------------------------
    # 1) Fetch farm from DB
    # ------------------------------------------------------------------
    try:
        farm = repos.get_farm(farm_id)
    except Exception as e:
        current_app.logger.exception("[PROCESS_OR_REFRESH] failed fetching farm")
        return jsonify({"error": "db_error", "detail": str(e)}), 500

    if not farm:
        return jsonify({"error": "farm_not_found"}), 404

    geojson = farm.get("geojson") or farm.get("geom")
    if not geojson:
        return jsonify({"error": "farm_missing_geometry"}), 400

    # ------------------------------------------------------------------
    # 2) Latest process_result from DB (if any)
    #    Expecting repos.get_latest_process_result to return dict like:
    #    {
    #      "id", "farm_id", "source", "scene_id",
    #      "start_ts", "end_ts", "created_at",
    #      "health_score", "result": {...}
    #    }
    # ------------------------------------------------------------------
    latest_row = None
    latest_result_out = None
    latest_scene_id = None
    latest_end_dt = None

    try:
        if hasattr(repos, "get_latest_process_result"):
            latest_row = repos.get_latest_process_result(farm_id)
    except Exception:
        current_app.logger.exception("[PROCESS_OR_REFRESH] DB get_latest_process_result failed (ignored)")
        latest_row = None

    if latest_row and isinstance(latest_row, dict):
        # unwrap stored "result" (the original process_s2 output)
        latest_result_out = latest_row.get("result") or {}
        latest_scene_id = latest_row.get("scene_id") or latest_result_out.get("provenance", {}).get("scene_id")

        # end_ts is usually ISO string from _process_result_to_dict
        end_ts_iso = latest_row.get("end_ts")
        if end_ts_iso:
            try:
                latest_end_dt = _iso_to_dt(end_ts_iso)
            except Exception:
                latest_end_dt = None

        current_app.logger.info(
            "[PROCESS_OR_REFRESH] Latest DB result for farm=%s scene_id=%s end_ts=%s",
            farm_id, latest_scene_id, end_ts_iso
        )

    # ------------------------------------------------------------------
    # 3) Find best scene from catalog for this farm (uses bbox + lookback)
    #    This also caches into Redis: scene:best:{farm_id}:overall
    # ------------------------------------------------------------------
    chosen_scene = find_best_scene_from_catalog(geojson, farm_id=farm_id)
    if not chosen_scene:
        # No catalog scene found in lookback window.
        if latest_result_out is not None:
            # Return stale/latest result with a flag.
            out = dict(latest_result_out)  # shallow copy to avoid mutating DB object
            prov = out.setdefault("provenance", {})
            prov.setdefault("farm_id", farm_id)
            prov.setdefault("collection", latest_row.get("source") or config.COL_S2)
            if latest_scene_id:
                prov.setdefault("scene_id", latest_scene_id)
            if latest_row.get("start_ts"):
                prov.setdefault("used_start", latest_row["start_ts"])
            if latest_row.get("end_ts"):
                prov.setdefault("used_end", latest_row["end_ts"])
            prov.setdefault("note", "no_new_scene_found_in_catalog")
            current_app.logger.info(
                "[PROCESS_OR_REFRESH] No catalog scenes for farm=%s; returning latest stored result.",
                farm_id,
            )
            return jsonify(out), 200

        # Truly nothing to show
        return jsonify({
            "error": "no_recent_scene_found",
            "detail": "No catalog scenes in lookback window and no previous process_result in DB."
        }), 404

    # Extract scene datetime
    scene_props = chosen_scene.get("properties", {}) or {}
    scene_dt_iso = (
        scene_props.get("datetime")
        or scene_props.get("start_datetime")
        or scene_props.get("end_datetime")
    )
    chosen_scene_id = chosen_scene.get("id")
    chosen_dt = _iso_to_dt(scene_dt_iso) if scene_dt_iso else None

    current_app.logger.info(
        "[PROCESS_OR_REFRESH] Best catalog scene for farm=%s: id=%s datetime=%s",
        farm_id, chosen_scene_id, scene_dt_iso
    )

    # ------------------------------------------------------------------
    # 4) Smart comparison: decide whether we really need to call SH PROCESS
    #
    #    Cases where we can safely reuse latest DB result:
    #      - latest_scene_id == chosen_scene_id  -> same scene already processed
    #      - latest_end_dt >= chosen_dt         -> we have processed same or newer date
    # ------------------------------------------------------------------
    if latest_result_out is not None:
        same_scene = latest_scene_id and chosen_scene_id and (str(latest_scene_id) == str(chosen_scene_id))
        newer_or_equal_time = False
        if latest_end_dt and chosen_dt:
            newer_or_equal_time = latest_end_dt >= chosen_dt

        if same_scene or newer_or_equal_time:
            # Reuse latest DB result
            out = dict(latest_result_out)
            prov = out.setdefault("provenance", {})
            prov.setdefault("farm_id", farm_id)
            prov.setdefault("collection", latest_row.get("source") or config.COL_S2)
            if latest_scene_id:
                prov.setdefault("scene_id", latest_scene_id)
            if latest_row.get("start_ts"):
                prov.setdefault("used_start", latest_row["start_ts"])
            if latest_row.get("end_ts"):
                prov.setdefault("used_end", latest_row["end_ts"])
            prov.setdefault("note", "reused_latest_db_result")
            current_app.logger.info(
                "[PROCESS_OR_REFRESH] Reusing latest DB result for farm=%s (same_scene=%s newer_or_equal_time=%s)",
                farm_id, same_scene, newer_or_equal_time
            )
            return jsonify(out), 200

    # ------------------------------------------------------------------
    # 5) We DO need to process a new scene (chosen_scene is newer/better)
    #    Build a tight time window around scene datetime (±1 minute)
    # ------------------------------------------------------------------
    if chosen_dt:
        start_iso = (chosen_dt - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        end_iso = (chosen_dt + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    else:
        # very unlikely, but fallback: small 30-minute window around "now"
        now = datetime.now(timezone.utc)
        start_iso = (now - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
        end_iso = (now + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")

    current_app.logger.info(
        "[PROCESS_OR_REFRESH] Processing new scene for farm=%s scene_id=%s window=%s..%s",
        farm_id, chosen_scene_id, start_iso, end_iso
    )

    # ------------------------------------------------------------------
    # 6) Call your existing process_s2() to hit SH PROCESS and compute NDVI
    # ------------------------------------------------------------------
    try:
        out = process_s2(geojson, start_iso, end_iso)
    except requests.exceptions.HTTPError as he:
        current_app.logger.exception("[PROCESS_OR_REFRESH] process_s2 HTTPError")
        return jsonify({"error": "process_api_failed", "detail": getattr(he.response, "text", str(he))}), 502
    except Exception as e:
        current_app.logger.exception("[PROCESS_OR_REFRESH] process_s2 failed")
        return jsonify({"error": "process_failed", "detail": str(e)}), 500

    # Attach provenance
    out.setdefault("provenance", {})
    out["provenance"].update({
        "farm_id": farm_id,
        "collection": config.COL_S2,
        "scene_id": chosen_scene_id,
        "used_start": start_iso,
        "used_end": end_iso
    })

    # ------------------------------------------------------------------
    # 7) Save to DB best-effort
    # ------------------------------------------------------------------
    try:
        start_dt = _iso_to_dt(start_iso)
        end_dt = _iso_to_dt(end_iso)
        health_score = None
        try:
            health_score = float(out.get("health_score")) if out.get("health_score") is not None else None
        except Exception:
            health_score = None

        repos.save_process_result(
            farm_id=farm_id,
            source=config.COL_S2,
            scene_id=chosen_scene_id,
            start_ts=start_dt,
            end_ts=end_dt,
            result=out,
            health_score=health_score
        )
    except Exception:
        current_app.logger.exception("[PROCESS_OR_REFRESH] failed saving process_result (ignored)")

    # ------------------------------------------------------------------
    # 8) Cache final result for fast re-use
    # ------------------------------------------------------------------
    try:
        cache_key = cache.make_key("process", farm_id, config.COL_S2, start_iso, end_iso)
        cache.set_json(cache_key, out, ttl=getattr(config, "CACHE_TTL_SECONDS", 86400))
    except Exception:
        current_app.logger.exception("[PROCESS_OR_REFRESH] failed caching process result (ignored)")

    return jsonify(out), 200


from AdvisoryEngine import build_crop_advice, analyze_trend, build_current_snapshot, _extract_scene_dt_from_pr, _extract_ndvi_stats_from_pr, get_or_refresh_latest
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

@api.route("/farm-insights", methods=["GET"])
def farm_insights_route():
    farm_id_param = request.args.get("farm_id")
    if not farm_id_param:
        return jsonify({"error": "farm_id_required", "detail": "Pass farm_id as query param"}), 400
    try:
        farm_id = int(farm_id_param)
    except ValueError:
        return jsonify({"error": "invalid_farm_id"}), 400

    try:
        history_limit = int(request.args.get("history_limit", 5))
    except Exception:
        history_limit = 5
    history_limit = max(1, min(30, history_limit))

    # 1) fetch farm
    try:
        farm = repos.get_farm(farm_id)
    except Exception as e:
        current_app.logger.exception("farm_insights: failed to fetch farm")
        return jsonify({"error": "db_error", "detail": str(e)}), 500
    if not farm:
        return jsonify({"error": "farm_not_found"}), 404

    # 2) Ensure we have a recent/latest process_result (try refresh if stale)
    try:
        # pass force_refresh via query param ?refresh=true
        force_refresh = str(request.args.get("refresh", "")).lower() in ("1", "true", "yes")
        latest_pr = get_or_refresh_latest(farm_id, force_refresh=force_refresh)
    except Exception as e:
        current_app.logger.exception("farm_insights: failed to obtain latest process_result")
        return jsonify({"error": "no_process_result", "detail": str(e)}), 502

    # 3) Get history (DB) for trend/plot (we still read from DB for history)
    try:
        history = repos.get_process_results(farm_id, limit=history_limit)
    except Exception:
        current_app.logger.exception("farm_insights: failed fetching history process_results")
        # continue with single latest only
        history = [latest_pr] if isinstance(latest_pr, dict) else []

    # 4) Build current snapshot from latest_pr and farm meta
    current_snapshot = build_current_snapshot(farm, latest_pr)

    # 5) Trend analysis (history: newest-first expected)
    trend_block = analyze_trend(history or [latest_pr])

    # 6) Advisory
    advisory_block = build_crop_advice(
        crop_info=current_snapshot.get("crop", {}),
        health_bucket=current_snapshot.get("health_bucket", current_snapshot.get("health_bucket", {})),
        ndvi_level=current_snapshot.get("ndvi_level", {}),
        trend_block=trend_block
    )

    # 7) compact history for UI
    history_out = []
    for pr in history:
        scene_dt = _extract_scene_dt_from_pr(pr)
        ndvi_stats = _extract_ndvi_stats_from_pr(pr)
        res = pr.get("result") or {}
        hs = pr.get("health_score", res.get("health_score"))
        history_out.append({
            "id": pr.get("id"),
            "scene_id": pr.get("scene_id"),
            "timestamp": scene_dt.isoformat() if scene_dt else None,
            "health_score": hs,
            "ndvi_mean": ndvi_stats.get("mean"),
            "ndvi_min": ndvi_stats.get("min"),
            "ndvi_max": ndvi_stats.get("max")
        })

    # 8) farm summary
    farm_summary = {
        "id": farm.get("id"),
        "name": farm.get("name"),
        "area_m2": farm.get("area_m2"),
        "created_at": farm.get("created_at").isoformat() if isinstance(farm.get("created_at"), datetime) else farm.get("created_at"),
        "crop": current_snapshot.get("crop"),
        "meta": farm.get("meta") or {}
    }

    return jsonify({
        "farm": farm_summary,
        "current": current_snapshot,
        "trend": trend_block,
        "advisory": advisory_block,
        "history": history_out
    }), 200