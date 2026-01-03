# repos.py
import json
import logging
from typing import Dict, List, Optional, Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from geoalchemy2.elements import WKTElement
from shapely.geometry import shape
from datetime import datetime
import models
from db import SessionLocal 
import hashlib
from io import BytesIO
from PIL import Image
import numpy as np
from flask import current_app

LOG = logging.getLogger("sh_app.repos")


def get_session() -> Session:
    """Return a new DB Session (caller is responsible for closing if not using helper wrappers)."""
    return SessionLocal()

def get_user_by_uid(session: Session, uid: str) -> Optional[models.AppUser]:
    """Return AppUser instance or None."""
    return session.execute(select(models.AppUser).where(models.AppUser.uid == uid)).scalar_one_or_none()


def create_user_with_session(session: Session, uid: Optional[str] = None, email: Optional[str] = None,
                             phone: Optional[str] = None) -> models.AppUser:
    """Create a user using the provided session (does not commit)."""
    user = models.AppUser(uid=uid, email=email, phone=phone)
    session.add(user)
    session.flush()  
    return user

from datetime import datetime
from typing import Optional
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
import models  # your models module

def get_or_create_user_by_uid(
    session,
    uid: str,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    display_name: Optional[str] = None,
    photo_url: Optional[str] = None,
) -> models.AppUser:
    """
    New version matching the new app_user schema.

    - Uses app_user.firebase_uid instead of app_user.uid
    - Only updates email/phone if provided and changed
    - No display_name/photo_url/roles/last_seen/etc (columns removed)
    """
    if not uid:
        raise ValueError("uid is required")

    user = (
        session.execute(
            select(models.AppUser).where(models.AppUser.firebase_uid == uid)
        )
        .scalar_one_or_none()
    )

    if user:
        updated = False
        if email and user.email != email:
            user.email = email
            updated = True
        if phone and user.phone != phone:
            user.phone = phone
            updated = True

        if updated:
            session.commit()
            session.refresh(user)
        return user

    try:
        new_user = models.AppUser(
            firebase_uid=uid,
            email=email,
            phone=phone,
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        return new_user
    except IntegrityError:
        session.rollback()
        user = (
            session.execute(
                select(models.AppUser).where(models.AppUser.firebase_uid == uid)
            )
            .scalar_one_or_none()
        )
        if user:
            return user
        raise

def update_user_profile(session: Session, uid: str, updates: Dict[str, Any]) -> Optional[models.AppUser]:
    """
    Update allowed user profile fields (display_name, photo_url, phone, meta).
    Returns updated AppUser or None if user not found.
    Caller must manage session closing.
    """
    user = session.execute(select(models.AppUser).where(models.AppUser.uid == uid)).scalar_one_or_none()
    if not user:
        return None

    allowed = {"display_name", "photo_url", "phone", "meta", "is_active"}
    changed = False
    for k, v in updates.items():
        if k in allowed:
            setattr(user, k, v)
            changed = True

    if changed:
        user.last_seen = datetime.utcnow()
        session.commit()
        session.refresh(user)
    else:
        session.commit()
        session.refresh(user)

    return user

def get_or_create_user(uid: str, email: Optional[str] = None, phone: Optional[str] = None) -> models.AppUser:
    session = get_session()
    try:
        return get_or_create_user_by_uid(session, uid, email=email, phone=phone)
    finally:
        session.close()


def create_farm(user_id: int, name: str, geojson: Dict, meta: Optional[Dict] = None) -> models.Farm:
    """
    Insert a farm polygon (GeoJSON) with computed bbox and area_m2.
    - geojson: a GeoJSON Polygon/MultiPolygon python dict
    - meta: optional JSONB metadata
    Returns the created Farm model instance.
    """
    session = get_session()
    try:
        geom_shape = shape(geojson)
        if not geom_shape.is_valid:
            geom_shape = geom_shape.buffer(0)

        wkt = geom_shape.wkt
        geom_wkt = WKTElement(wkt, srid=4326)

        geom_expr = func.ST_GeomFromText(wkt, 4326)

        area_m2 = session.scalar(func.ST_Area(func.ST_Transform(geom_expr, 3857)))

        bbox_wkt = session.scalar(func.ST_AsText(func.ST_Envelope(geom_expr)))
        bbox_elem = WKTElement(bbox_wkt, srid=4326) if bbox_wkt else None

        farm = models.Farm(
            user_id=user_id,
            name=name,
            geom=geom_wkt,
            bbox=bbox_elem,
            area_m2=float(area_m2) if area_m2 is not None else None,
            meta=meta or {}
        )
        session.add(farm)
        session.commit()
        session.refresh(farm)
        return farm
    except Exception as e:
        session.rollback()
        LOG.exception("create_farm failed: %s", e)
        raise
    finally:
        session.close()


def get_farm(farm_id: int) -> Optional[Dict[str, Any]]:
   
    session = get_session()
    try:
        q = session.query(
            models.Farm,
            func.ST_AsGeoJSON(models.Farm.geom).label("geom_geojson"),
            func.ST_AsGeoJSON(models.Farm.bbox).label("bbox_geojson")
        ).filter(models.Farm.id == farm_id)
        row = q.one_or_none()
        if not row:
            return None
        farm_obj = row[0]
        geom_json = json.loads(row.geom_geojson) if row.geom_geojson else None
        bbox_json = json.loads(row.bbox_geojson) if row.bbox_geojson else None

        return {
            "id": farm_obj.id,
            "user_id": farm_obj.user_id,
            "name": farm_obj.name,
            "geojson": geom_json,
            "geom": geom_json,         # legacy alias
            "bbox": bbox_json,
            "area_m2": float(farm_obj.area_m2) if farm_obj.area_m2 is not None else None,
            "meta": farm_obj.meta,
            "created_at": farm_obj.created_at.isoformat() if farm_obj.created_at else None
        }
    finally:
        session.close()



def list_farms(user_id: int) -> List[Dict[str, Any]]:
   
    session = get_session()
    try:
        q = session.query(
            models.Farm,
            func.ST_AsGeoJSON(models.Farm.geom).label("geom_geojson")
        ).filter(models.Farm.user_id == user_id)
        results: List[Dict[str, Any]] = []
        for farm_obj, geom_json_str in q.all():
            geom = json.loads(geom_json_str) if geom_json_str else None
            results.append({
                "id": farm_obj.id,
                "name": farm_obj.name,
                "meta": farm_obj.meta
            })

        return results
    finally:
        session.close()


# -------------------------
# Scenes / Process results
# -------------------------
def save_scene(farm_id: int, scene_id: str, collection: str, properties: Dict, score: float,
               meta: Optional[Dict] = None) -> models.Scene:
    """
    Save scene (catalog result) for a farm. If scene exists, update fields.
    """
    session = get_session()
    try:
        existing = session.query(models.Scene).filter_by(farm_id=farm_id, scene_id=scene_id).one_or_none()
        if existing:
            existing.properties = properties
            existing.score = score
            if meta is not None:
                existing.meta = meta
            session.commit()
            session.refresh(existing)
            return existing

        scene = models.Scene(
            farm_id=farm_id,
            scene_id=scene_id,
            collection=collection,
            properties=properties,
            score=score,
            meta=meta or {}
        )
        session.add(scene)
        session.commit()
        session.refresh(scene)
        return scene
    except Exception as e:
        session.rollback()
        LOG.exception("save_scene failed: %s", e)
        raise
    finally:
        session.close()


def save_process_result(farm_id: int, source: str, scene_id: Optional[str], start_ts, end_ts,
                        result: Dict, health_score: Optional[float] = None) -> models.ProcessResult:
  
    session = get_session()
    try:
        pr = models.ProcessResult(
            farm_id=farm_id,
            source=source,
            scene_id=scene_id,
            start_ts=start_ts,
            end_ts=end_ts,
            result=result,
            health_score=health_score
        )
        session.add(pr)
        session.commit()
        session.refresh(pr)
        return pr
    except Exception as e:
        session.rollback()
        LOG.exception("save_process_result failed: %s", e)
        raise
    finally:
        session.close()


# -------------------------
# Logging / quota
# -------------------------
def log_api(user_id: Optional[int], endpoint: str, payload: Dict, response_status: int = 200) -> None:
    session = get_session()
    try:
        entry = models.ApiLog(user_id=user_id, endpoint=endpoint, payload=payload, response_status=response_status)
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
        LOG.exception("log_api failed for endpoint=%s user_id=%s", endpoint, user_id)
    finally:
        session.close()


def increment_quota(user_id: int, calls: int = 1) -> None:
    session = get_session()
    try:
        uq = session.query(models.UserQuota).filter_by(user_id=user_id).one_or_none()
        if not uq:
            uq = models.UserQuota(user_id=user_id, monthly_calls=calls)
            session.add(uq)
        else:
            uq.monthly_calls = (uq.monthly_calls or 0) + calls
        session.commit()
    except Exception:
        session.rollback()
        LOG.exception("increment_quota failed for user_id=%s", user_id)
        raise
    finally:
        session.close()

def _eval_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _png_mostly_gray(png_bytes: bytes, gray_rgb=(128,128,128), threshold=0.90, var_threshold=5.0) -> bool:
    """
    Returns True if image appears mostly gray.
    Criteria:
      - fraction of pixels exactly equal to gray_rgb >= threshold OR
      - image RGB variance across channels is very small (< var_threshold)
    """
    try:
        im = Image.open(BytesIO(png_bytes)).convert("RGB")
        arr = np.array(im)
        h,w,_ = arr.shape
        total = h*w
        # exact match fraction
        matches = np.all(arr == np.array(gray_rgb, dtype=np.uint8), axis=2)
        frac = float(np.count_nonzero(matches)) / float(total)
        if frac >= threshold:
            return True
        # else check variance (low variance -> monochrome)
        # compute avg per-channel std
        std = float(np.mean(np.std(arr.reshape(-1,3), axis=0)))
        if std < var_threshold:
            return True
    except Exception:
        current_app.logger.exception("Failed to analyze PNG bytes for grayness; will not treat as gray")
    return False

from rasterio.io import MemoryFile


def compute_ndvi_stats_from_tiff_bytes(tiff_bytes: bytes):
    """
    Read a single-band FLOAT32 GeoTIFF returned by SH (our EVAL_S2_NDVI) and return:
      - ndvi_arr: 2D numpy float32 array (NaN for invalid)
      - profile: rasterio profile dict
      - stats: dict with basic stats: min, max, mean, median, valid_count, invalid_count
    """
    try:
        with MemoryFile(tiff_bytes) as mem:
            with mem.open() as ds:
                arr = ds.read(1).astype(np.float32)  # single band
                profile = ds.profile.copy()
                # sentinel-hub may encode NaN as some nodata; detect nodata if set
                nodata = ds.nodatavals[0]
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                # basic stats on finite values
                valid_mask = np.isfinite(arr)
                vals = arr[valid_mask]
                if vals.size:
                    stats = {
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                        "mean": float(np.nanmean(vals)),
                        "median": float(np.nanmedian(vals)),
                        "std": float(np.nanstd(vals)),
                        "valid_count": int(vals.size),
                        "invalid_count": int(arr.size - vals.size)
                    }
                else:
                    stats = {"min": None, "max": None, "mean": None, "median": None, "std": None, "valid_count": 0, "invalid_count": int(arr.size)}
                return arr, profile, {"stats": stats}
    except Exception as e:
        current_app.logger.exception("compute_ndvi_stats_from_tiff_bytes failed")
        raise

def ndvi_array_to_colormap_png(ndvi_arr: np.ndarray, invalid_mask: np.ndarray = None, out_size=(512, 512)):
    """
    Convert NDVI float32 array into a color-mapped PNG bytes.
    Color scheme matches EVAL_S2_NDVI_COLORMAP:
      ndvi < 0.0 -> gray (128,128,128)
      [0.0,0.2) -> red (255,0,0)
      [0.2,0.4) -> orange (255,166,0)
      [0.4,0.6) -> yellow (255,255,0)
      [0.6,0.8) -> green (51,204,51)
      >=0.8 -> bright green (0,255,0)
    Returns PNG bytes.
    """
    try:
        arr = ndvi_arr.astype(np.float32)
        h, w = arr.shape
        # scale/resample to out_size if needed (use PIL nearest)
        img = Image.new("RGB", (w, h))
        # prepare color map
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if invalid_mask is None:
            invalid_mask = ~np.isfinite(arr)
        # handle finite values
        finite_mask = np.isfinite(arr) & (~invalid_mask)
        # default gray
        rgb[:, :, :] = np.array([128, 128, 128], dtype=np.uint8)
        # assign colors by bins
        m = finite_mask & (arr < 0.0)
        rgb[m] = np.array([128, 128, 128], dtype=np.uint8)
        m = finite_mask & (arr >= 0.0) & (arr < 0.2)
        rgb[m] = np.array([255, 0, 0], dtype=np.uint8)
        m = finite_mask & (arr >= 0.2) & (arr < 0.4)
        rgb[m] = np.array([255, 166, 0], dtype=np.uint8)
        m = finite_mask & (arr >= 0.4) & (arr < 0.6)
        rgb[m] = np.array([255, 255, 0], dtype=np.uint8)
        m = finite_mask & (arr >= 0.6) & (arr < 0.8)
        rgb[m] = np.array([51, 204, 51], dtype=np.uint8)
        m = finite_mask & (arr >= 0.8)
        rgb[m] = np.array([0, 255, 0], dtype=np.uint8)
        pil = Image.fromarray(rgb, mode="RGB")
        if out_size and (out_size[0] != w or out_size[1] != h):
            pil = pil.resize(out_size, resample=Image.BILINEAR)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        current_app.logger.exception("ndvi_array_to_colormap_png failed")
        raise

from typing import Optional, Dict, List

# ... existing imports ...
# from .database import get_session  # whatever you already have
# import models
# LOG = logging.getLogger(__name__)
def _process_result_to_dict(pr: "models.ProcessResult") -> Dict:
    """
    Internal helper to convert a ProcessResult ORM object into a clean dict
    that matches what the rest of the app expects.
    """
    if pr is None:
        return {}

    # pr.result already contains the full NDVI JSON you saved (provenance, indices_stats, previews_base64, etc.)
    # We merge in some top-level fields for convenience.
    base = pr.result.copy() if isinstance(pr.result, dict) else {}

    # Keep original result under a key for backwards compatibility
    base["result"] = pr.result if isinstance(pr.result, dict) else None

    # Ensure provenance exists
    prov = base.get("provenance") or {}
    prov.setdefault("farm_id", pr.farm_id)
    prov.setdefault("scene_id", pr.scene_id)
    prov.setdefault("used_start", pr.start_ts.isoformat() if pr.start_ts else None)
    prov.setdefault("used_end", pr.end_ts.isoformat() if pr.end_ts else None)
    base["provenance"] = prov

    # Surface these fields at top-level for easy access
    base.setdefault("health_score", float(pr.health_score) if pr.health_score is not None else None)
    base.setdefault("scene_id", pr.scene_id)
    base.setdefault("source", pr.source)
    base.setdefault("start_ts", pr.start_ts.isoformat() if pr.start_ts else None)
    base.setdefault("end_ts", pr.end_ts.isoformat() if pr.end_ts else None)
    base.setdefault("created_at", pr.created_at.isoformat() if pr.created_at else None)

    return base

def get_latest_process_result(farm_id: int) -> Optional[Dict]:
    """
    Return the most recent ProcessResult for a farm_id as a dict
    (merged result JSON + metadata), or None if none exist.
    """
    session = get_session()
    try:
        pr = (
            session.query(models.ProcessResult)
            .filter(models.ProcessResult.farm_id == farm_id)
            .order_by(models.ProcessResult.end_ts.desc(), models.ProcessResult.created_at.desc())
            .first()
        )
        if not pr:
            return None
        return _process_result_to_dict(pr)
    except Exception as e:
        LOG.exception("get_latest_process_result failed for farm_id=%s: %s", farm_id, e)
        raise
    finally:
        session.close()

def get_process_results(farm_id: int, limit: int = 10) -> List[Dict]:
    """
    Return a list of recent ProcessResult dicts for a farm_id, newest first.
    Each dict is the merged result JSON + metadata (same shape as get_latest_process_result).
    """
    session = get_session()
    try:
        q = (
            session.query(models.ProcessResult)
            .filter(models.ProcessResult.farm_id == farm_id)
            .order_by(models.ProcessResult.end_ts.desc(), models.ProcessResult.created_at.desc())
            .limit(int(limit))
        )
        rows = q.all()
        out: List[Dict] = []
        for pr in rows:
            out.append(_process_result_to_dict(pr))
        return out
    except Exception as e:
        LOG.exception("get_process_results failed for farm_id=%s: %s", farm_id, e)
        raise
    finally:
        session.close()






from sqlalchemy import desc
from models import UserDetectedDisease, CropHealthAnalysis

def list_user_disease_history(user_id: int):
    session = SessionLocal()
    try:
        rows = (
            session.query(
                UserDetectedDisease.disease_name,
                UserDetectedDisease.scientific_name,
                UserDetectedDisease.detection_count,
                UserDetectedDisease.last_detected_at,
                CropHealthAnalysis.id.label("analysis_id"),
                CropHealthAnalysis.farm_id
            )
            .join(
                CropHealthAnalysis,
                CropHealthAnalysis.detected_disease == UserDetectedDisease.disease_name
            )
            .filter(UserDetectedDisease.user_id == user_id)
            .order_by(UserDetectedDisease.last_detected_at.desc())
            .all()
        )

        return [
            {
                "analysis_id": r.analysis_id,
                "farm_id": r.farm_id,
                "disease_name": r.disease_name,
                "scientific_name": r.scientific_name,
                "detection_count": r.detection_count,
                "last_detected_at": r.last_detected_at.isoformat()
            }
            for r in rows
        ]
    finally:
        session.close()
