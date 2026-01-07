from Soil_Analysis.processing_soil_analysis import (
    extract_text_with_fallback,
    normalize_soil_text,
    call_gemini_soil_analysis,
)
from models import SoilAnalysisReport, UserSoilHistory
from db import SessionLocal
import tempfile
import os
import logging

LOG = logging.getLogger(__name__)

def run_soil_analysis(
    farm_id: int,
    pdf_bytes: bytes,
    file_hash: str,
    file_name: str,
    user_id: int | None = None,
):
    session = SessionLocal()

    try:
        # write PDF inside worker container
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        raw_text = extract_text_with_fallback(tmp_path)
        cleaned = normalize_soil_text(raw_text)
        llm = call_gemini_soil_analysis(cleaned)

        report = SoilAnalysisReport(
            user_id=user_id,
            farm_id=farm_id,
            file_name=file_name,
            file_hash=file_hash,
            raw_text=raw_text,
            cleaned_text=cleaned,
            soil_parameters=llm.get("soil_parameters"),
            advisory=llm.get("advisory"),
            confidence=llm.get("confidence"),
        )

        session.add(report)
        session.flush()

        if user_id:
            session.add(UserSoilHistory(
                user_id=user_id,
                report_id=report.id,
                summary=report.advisory.get("summary"),
                confidence=report.confidence
            ))

        session.commit()
        return {"status": "done", "report_id": report.id}

    except Exception:
        session.rollback()
        LOG.exception("soil analysis failed")
        raise

    finally:
        session.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
