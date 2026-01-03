from typing import Dict, Any, Optional, List
from statistics import mean
from datetime import datetime

from sqlalchemy.orm import Session
import models

def clamp(v: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, v))

def normalize_score(value: Optional[float], min_v: float, max_v: float) -> Optional[float]:
    if value is None:
        return None
    return clamp((value - min_v) / (max_v - min_v))


def extract_satellite_signal(
    pr: Optional[models.ProcessResult]
) -> Dict[str, Any]:
    # 1. Basic Safety Checks
    if not pr or not pr.result:
        return {
            "available": False,
            "health_score": None,
            "confidence": 0.0,
            "notes": ["No recent satellite data"]
        }

    # 2. MATCHING YOUR ACTUAL JSON STRUCTURE
    # DB: "indices_stats": {"ndvi": {"mean": 0.22, ...}}
    stats = pr.result.get("indices_stats", {})
    ndvi_stats = stats.get("ndvi", {})
    
    ndvi_mean = ndvi_stats.get("mean")

    # 3. Handle the Health Score Scale
    # Your DB stores it as 0-100 (e.g., 38.22), but we need 0.0-1.0 for fusion
    db_score = pr.health_score # Column: 38.22
    
    final_health_score = None
    if db_score is not None:
        final_health_score = clamp(db_score / 100.0) # Convert 38.22 -> 0.38
    
    # Fallback: Calculate from NDVI if DB score is missing
    if final_health_score is None and ndvi_mean is not None:
        final_health_score = normalize_score(ndvi_mean, 0.2, 0.8)

    return {
        "available": True,
        "ndvi_mean": ndvi_mean,
        "health_score": final_health_score,
        "confidence": 0.7, # Sentinel-2 is reliable
        "notes": [
            f"Average NDVI = {ndvi_mean:.2f}" if ndvi_mean is not None else "NDVI unavailable"
        ]
    }

# --- DISEASE SIGNAL (Unchanged) ---
def extract_crop_disease_signal(
    analysis: Optional[models.CropHealthAnalysis]
) -> Dict[str, Any]:
    if not analysis:
        return {
            "available": False,
            "risk": None,
            "confidence": 0.0,
            "notes": ["No crop disease analysis"]
        }

    disease = analysis.detected_disease
    severity = (analysis.advisory or {}).get("severity_assessment", "UNKNOWN")

    severity_map = {
        "LOW": 0.3,
        "MODERATE": 0.6,
        "HIGH": 0.9
    }
    risk_score = severity_map.get(severity.upper(), 0.5)

    return {
        "available": True,
        "disease": disease,
        "severity": severity,
        "risk": risk_score,
        "confidence": 0.85,
        "notes": [
            f"Disease detected: {disease}",
            f"Severity assessed as {severity}"
        ]
    }
def extract_soil_signal(
    soil_report: Optional[models.SoilAnalysisReport]
) -> Dict[str, Any]:
    """
    Extracts signals directly from the SoilAnalysisReport.soil_parameters JSON.
    """
    if not soil_report:
        return {
            "available": False,
            "nutrient_risk": None,
            "confidence": 0.0,
            "notes": ["No soil report available"]
        }

    # 1. USE THE CORRECT COLUMN NAME: 'soil_parameters'
    # This matches the JSONB column in your models.py
    data = soil_report.soil_parameters or {}

    # 2. Extract values (Adjust keys if your JSON uses different casing, e.g. "Nitrogen")
    # Using .get() prevents crashes if a key is missing
    nitrogen = data.get("nitrogen") or data.get("Nitrogen")
    phosphorus = data.get("phosphorus") or data.get("Phosphorus")
    potassium = data.get("potassium") or data.get("Potassium")

    deficiencies = []
    
    # 3. Safe check logic
    try:
        if nitrogen is not None and float(nitrogen) < 280:
            deficiencies.append("Nitrogen")
        if phosphorus is not None and float(phosphorus) < 10:
            deficiencies.append("Phosphorus")
        if potassium is not None and float(potassium) < 120:
            deficiencies.append("Potassium")
    except (ValueError, TypeError):
        pass # Ignore malformed data (e.g., if value is "Low" string instead of number)

    # Calculate Risk
    risk_score = clamp(len(deficiencies) / 3)

    return {
        "available": True,
        "deficiencies": deficiencies,
        "nutrient_risk": risk_score,
        "confidence": 0.9,
        "notes": [
            f"Deficient nutrients: {', '.join(deficiencies)}"
            if deficiencies else "Soil nutrients within acceptable range"
        ]
    }# --- FUSION LOGIC (Unchanged) ---
def fuse_crop_health(
    *,
    satellite: Dict[str, Any],
    disease: Dict[str, Any],
    soil: Dict[str, Any]
) -> Dict[str, Any]:

    scores = []
    confidences = []
    explain = []

    if satellite["available"] and satellite["health_score"] is not None:
        scores.append(1 - satellite["health_score"])
        confidences.append(satellite["confidence"])
        explain.extend(satellite["notes"])

    if disease["available"]:
        scores.append(disease["risk"])
        confidences.append(disease["confidence"])
        explain.extend(disease["notes"])

    if soil["available"]:
        scores.append(soil["nutrient_risk"])
        confidences.append(soil["confidence"])
        explain.extend(soil["notes"])

    overall_risk = clamp(mean(scores)) if scores else None
    overall_conf = clamp(mean(confidences)) if confidences else 0.0

    if overall_risk is None:
        risk_level = "UNKNOWN"
    elif overall_risk < 0.3:
        risk_level = "LOW"
    elif overall_risk < 0.6:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    return {
        "overall_health_score": 1 - overall_risk if overall_risk is not None else None,
        "risk_level": risk_level,
        "signals": {
            "satellite": satellite,
            "crop_disease": disease,
            "soil": soil
        },
        "confidence": {
            "satellite": satellite["confidence"],
            "crop_disease": disease["confidence"],
            "soil": soil["confidence"],
            "overall": overall_conf
        },
        "explainability": explain
    }

# --- UPDATED RUNNER (Removed Snapshot Query) ---
def run_fusion_for_farm(
    session: Session,
    *,
    farm_id: int,
    user_id: int
) -> Dict[str, Any]:

    # 1. Latest Satellite
    pr = (
        session.query(models.ProcessResult)
        .filter(models.ProcessResult.farm_id == farm_id)
        .order_by(models.ProcessResult.created_at.desc())
        .first()
    )

    # 2. Latest Crop Disease
    disease = (
        session.query(models.CropHealthAnalysis)
        .filter(models.CropHealthAnalysis.farm_id == farm_id)
        .order_by(models.CropHealthAnalysis.created_at.desc())
        .first()
    )

    # 3. Latest Soil Report (Metadata is inside here now)
    soil_report = (
        session.query(models.SoilAnalysisReport)
        .filter(models.SoilAnalysisReport.farm_id == farm_id)
        .order_by(models.SoilAnalysisReport.created_at.desc())
        .first()
    )

    # No more snapshot query needed!

    satellite_signal = extract_satellite_signal(pr)
    disease_signal = extract_crop_disease_signal(disease)
    
    # Pass only the report
    soil_signal = extract_soil_signal(soil_report)

    return fuse_crop_health(
        satellite=satellite_signal,
        disease=disease_signal,
        soil=soil_signal
    )