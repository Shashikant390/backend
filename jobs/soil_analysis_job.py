from Soil_Analysis.processing_soil_analysis import (
    extract_text_with_fallback,
    normalize_soil_text,
    call_gemini_soil_analysis,
)
from models import SoilAnalysisReport
from db import SessionLocal
import logging

LOG = logging.getLogger(__name__)

def run_soil_analysis(farm_id: int, pdf_path: str):
    session = SessionLocal()

    try:
        raw_text = extract_text_with_fallback(pdf_path)
        cleaned = normalize_soil_text(raw_text)

        llm_result = call_gemini_soil_analysis(cleaned)

        report = SoilAnalysisReport(
            farm_id=farm_id,
            advisory=llm_result
        )

        session.add(report)
        session.commit()

        return {"status": "done"}

    except Exception as e:
        session.rollback()
        LOG.exception("soil analysis failed")
        raise

    finally:
        session.close()
