# processing_crop_health.py
import base64
import io
import requests
from typing import List, Dict, Any
from PIL import Image

KINDWISE_ENDPOINT_DEFAULT = "https://crop.kindwise.com/api/v1/identification"

from crophealthadvisoryengine import (
    infer_disease_type,
    generate_immediate_actions,
    generate_fertilizer_plan,
    generate_control_plan,
    generate_application_guidelines,
    generate_preventive_care
)



def images_to_base64_flask(files) -> List[str]:
    images_b64 = []
    for f in files:
        img = Image.open(f.stream).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return images_b64


def call_kindwise_identify(
    images_base64: List[str],
    api_key: str,
    endpoint: str = KINDWISE_ENDPOINT_DEFAULT,
    latitude: float | None = None,
    longitude: float | None = None,
    datetime_iso: str | None = None,
) -> Dict[str, Any]:

    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "images": images_base64,
          "similar_images": True 
    }

    if latitude is not None and longitude is not None:
        payload["latitude"] = latitude
        payload["longitude"] = longitude

    if datetime_iso:
        payload["datetime"] = datetime_iso

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)

    # ✅ FIX #1 — Accept BOTH 200 and 201
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Kindwise error {resp.status_code}: {resp.text}"
        )

    return resp.json()


def generate_advisory_from_kindwise(kindwise_resp: Dict[str, Any]) -> Dict[str, Any]:

    result = kindwise_resp.get("result", {})

    is_plant = result.get("is_plant", {})

    if not is_plant.get("binary", False):
        return {
            "detection_summary": {
                "is_plant": False,
                "plant_probability": round(is_plant.get("probability", 0.0), 3),
                "message": "The uploaded image does not appear to contain a crop or plant."
            },
            "advisory": {
                "disease_name": None,
                "disease_type": None,
                "severity_assessment": "No plant detected — agronomic advisory not applicable.",
                "recommended_actions": {
                    "immediate_treatment": [],
                    "fertilizer_management": [],
                    "chemical_or_biological_control": [],
                    "application_guidelines": [],
                    "preventive_care_next_stage": [
                        "Upload clear photos of crop leaves, stems, or affected plant parts.",
                        "Avoid people, animals, tools, or background objects.",
                        "Take images in daylight with the camera focused on the plant."
                    ]
                }
            },
            "confidence_note": {
                "level": "LOW",
                "message": "Plant presence could not be confirmed."
            }
        }

    crop_suggestions = result.get("crop", {}).get("suggestions", [])
    top_crop = crop_suggestions[0] if crop_suggestions else None

    crop_probability = top_crop.get("probability", 0.0) if top_crop else 0.0
    crop_name = top_crop.get("name").title() if top_crop else "Unknown crop"
    crop_confidence_low = crop_probability < 0.2

    disease_suggestions = result.get("disease", {}).get("suggestions", [])

    if not disease_suggestions:
        return {
            "detection_summary": {
                "is_plant": True,
                "crop_detected": crop_name,
                "message": "Plant detected but no disease signal identified."
            },
            "advisory": {
                "disease_name": None,
                "disease_type": None,
                "severity_assessment": "No visible disease detected from supplied images.",
                "recommended_actions": {
                    "immediate_treatment": [],
                    "fertilizer_management": [],
                    "chemical_or_biological_control": [],
                    "application_guidelines": [],
                    "preventive_care_next_stage": [
                        "Continue routine crop monitoring.",
                        "Upload clearer images if symptoms appear."
                    ]
                }
            }
        }

    
    top_disease = max(disease_suggestions, key=lambda d: d.get("probability", 0.0))

    disease_name = top_disease.get("name", "").title()
    scientific_name = top_disease.get("scientific_name", "")
    disease_probability = top_disease.get("probability", 0.0)

    if disease_probability >= 0.6:
        severity_text = (
            "High likelihood disease detected; unmanaged spread may impact yield."
        )
    elif disease_probability >= 0.3:
        severity_text = (
            "Moderate disease signal detected; early-stage intervention recommended."
        )
    else:
        severity_text = (
            "Low-confidence disease signal; confirm with clearer images before action."
        )

    advisory = {
        "disease_name": disease_name,
        "disease_type": infer_disease_type(scientific_name),
        "severity_assessment": severity_text,
        "recommended_actions": {
            "immediate_treatment": generate_immediate_actions(scientific_name),
            "fertilizer_management": generate_fertilizer_plan(scientific_name),
            "chemical_or_biological_control": generate_control_plan(scientific_name),
            "application_guidelines": generate_application_guidelines(scientific_name),
            "preventive_care_next_stage": generate_preventive_care(scientific_name),
        }
    }

    return {
        "detection_summary": {
            "is_plant": True,
            "plant_probability": round(is_plant.get("probability", 0.0), 3),
            "crop_detected": crop_name,
            "crop_probability": round(crop_probability, 3),
            "disease_detected": disease_name,
            "scientific_name": scientific_name,
            "disease_probability": round(disease_probability, 3),
            "note": (
                "Crop confidence is low; upload clearer leaf images."
                if crop_confidence_low
                else "Disease identified from visible plant symptoms."
            )
        },
        "advisory": advisory,
        "confidence_note": {
            "level": "LOW" if crop_confidence_low else "MEDIUM",
            "message": (
                "Upload 2–3 close-up leaf images for higher diagnostic confidence."
                if crop_confidence_low
                else "Diagnosis confidence acceptable."
            )
        }
    }

def extract_similar_images_from_kindwise(kindwise_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts similar image references from Kindwise response
    (safe for missing fields).
    """
    out = []

    disease_suggestions = (
        kindwise_resp
        .get("result", {})
        .get("disease", {})
        .get("suggestions", [])
    )

    for disease in disease_suggestions:
        sim_imgs = disease.get("similar_images") or []
        for img in sim_imgs:
            out.append({
                "disease_name": disease.get("name"),
                "scientific_name": disease.get("scientific_name"),
                "image_url": img.get("url"),
                "thumbnail_url": img.get("url_small"),
                "license_name": img.get("license_name"),
                "license_url": img.get("license_url"),
                "citation": img.get("citation"),
                "similarity": img.get("similarity"),
            })

    return out
