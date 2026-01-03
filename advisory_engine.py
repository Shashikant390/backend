import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Keep your existing imports for data fetching/parsing
from dateutil import parser as date_parser
import requests
import config
import repos


LOG = logging.getLogger("advisory_engine")

# Initialize Gemini
GEMINI_API_KEY = getattr(config, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
genai.configure(api_key=GEMINI_API_KEY)

# Use Flash for speed and low cost
MODEL_NAME = "gemini-2.5-flash"

def generate_ai_analysis(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends farm data to Gemini and gets a structured JSON advisory.
    """
    
    # 1. Define the Output Schema (Structured Output)
    # This ensures Gemini always returns valid JSON matching this shape
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.4, # Low temperature for factual, consistent advice
    )

    # 2. Construct the Prompt
    prompt = f"""
    You are an expert Senior Agronomist AI. Analyze the following satellite and weather data for a farm and provide a professional advisory.

    --- FARM CONTEXT ---
    Crop: {context_data.get('crop_type', 'Unknown')}
    Stage: {context_data.get('crop_stage', 'Unknown')}
    Location (Lat/Lon): {context_data.get('location', 'Unknown')}
    
    --- SATELLITE DATA (Sentinel-2) ---
    Current NDVI Mean: {context_data.get('cur_ndvi')}
    Baseline NDVI (Historical): {context_data.get('baseline_ndvi')}
    NDVI Change: {context_data.get('ndvi_drop_abs')} (Positive means drop/stress)
    Spatial Variance (Patchiness): {context_data.get('spread')} (High means uneven growth)
    
    --- TEMPORAL TREND ---
    Trend Slope: {context_data.get('temporal_slope')} (Negative means declining health)
    
    --- WEATHER CONTEXT ---
    Recent Rainfall Deficit: {context_data.get('rain_deficit_norm')} (0-1 scale, 1 is severe drought)
    Temperature Stress: {context_data.get('temp_norm')} (0-1 scale, 1 is extreme heat)

    --- INSTRUCTIONS ---
    1. Analyze the correlation between weather, NDVI trends, and crop stage.
    2. Determine if the stress is abiotic (water/nutrient) or biotic (pest/disease) based on patchiness (high variance usually means pest/disease/soil issues, uniform decline usually means water/nutrient).
    3. Provide 3-4 specific, actionable recommendations suitable for the crop stage.
    4. Assign a severity level (INFO, WARNING, CRITICAL) and a health score (0-100).

    --- OUTPUT FORMAT (JSON) ---
    {{
        "severity": "INFO" | "WARNING" | "CRITICAL",
        "stress_score": <float 0.0-1.0>,
        "title": "<Short, punchy headline for the advisory>",
        "explanation": {{
            "diagnosis": "<2-3 sentences explaining the root cause>",
            "evidence": ["<Bullet point 1 based on data>", "<Bullet point 2>"]
        }},
        "recommended_actions": ["<Action 1>", "<Action 2>", "<Action 3>"]
    }}
    """

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt, 
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return json.loads(response.text)
    except Exception as e:
        LOG.error(f"AI Generation failed: {e}")
        return None # Signal to fall back to rule-based system

def generate_advisory_for_farm(farm_id: int, force_refresh: bool = False, persist: bool = True) -> Dict[str, Any]:
    """
    Orchestrator: Fetches data -> Calculates Math -> Calls AI -> Saves result.
    """
    
    # 1. Data Fetching (Keep your existing robust logic)
    latest_pr = get_or_refresh_latest(farm_id, force_refresh=force_refresh)
    history_prs = repos.get_process_results(farm_id, limit=8) or []
    weather = repos.get_recent_weather_for_farm(farm_id) # Assuming this exists
    farm = repos.get_farm(farm_id)
    crop_info = farm.get("meta", {}) if farm else {}
    
    # ... [Insert your existing crop stage detection logic here] ...
    # Assume 'crop_type_norm' and 'crop_stage' are determined
    
    # 2. Math / Signal Calculation (Keep this! AI needs these numbers)
    signals, meta = compute_signals(latest_pr, history_prs, weather, crop_info)
    
    # 3. Prepare Context for AI
    context_payload = {
        "crop_type": crop_info.get("crop_name", "Unknown"),
        "crop_stage": crop_info.get("sowing_date", "Unknown"), # Or calculated stage
        "location": farm.get("geom", "Unknown"), # Pass rough coords if safe
        "cur_ndvi": meta.get("cur_ndvi"),
        "baseline_ndvi": meta.get("baseline_ndvi"),
        "ndvi_drop_abs": meta.get("ndvi_drop_abs"),
        "spread": meta.get("spread"),
        "temporal_slope": signals.get("temporal_slope"),
        "rain_deficit_norm": meta.get("weather_norms", {}).get("rain_deficit_norm"),
        "temp_norm": meta.get("weather_norms", {}).get("temp_norm")
    }

    # 4. Generate AI Advisory
    ai_result = generate_ai_analysis(context_payload)

    # 5. Fallback Strategy (If AI fails/times out, use your old logic)
    if not ai_result:
        LOG.warning("Falling back to static rules for farm %s", farm_id)
        # ... Call your original compute_stress_score, recommend_actions etc ...
        # For brevity, assuming you kept the old logic functions available
        score = compute_stress_score(signals)
        severity = classify_severity(score)
        actions = recommend_actions(severity, [], crop_info.get("crop_name"), None)
        
        ai_result = {
            "severity": severity,
            "stress_score": score,
            "title": f"{severity} Alert: Crop Stress Detected",
            "explanation": {
                "diagnosis": "Automated signal analysis detected anomalies.",
                "evidence": ["AI unavailable, using fallback rules."]
            },
            "recommended_actions": actions
        }

    # 6. Construct Final Payload
    advisory_payload = {
        "farm_id": farm_id,
        "scene_id": latest_pr.get("scene_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "severity": ai_result.get("severity", "INFO").upper(),
        "stress_score": ai_result.get("stress_score", 0.0),
        "confidence": "HIGH", # AI is confident
        "signals": signals, # Keep raw signals for debugging/charts
        "meta": meta,
        "explanation": {
            "title": ai_result.get("title"),
            "diagnosis": ai_result.get("explanation", {}).get("diagnosis"),
            "evidence": ai_result.get("explanation", {}).get("evidence", [])
        },
        "recommended_actions": ai_result.get("recommended_actions", []),
        "crop_type": crop_info.get("crop_name"),
    }

    # 7. Persist
    if persist and hasattr(repos, "save_advisory_history"):
        repos.save_advisory_history(advisory_payload)

    return advisory_payload

# ... [Keep your helper functions like get_or_refresh_latest] ...