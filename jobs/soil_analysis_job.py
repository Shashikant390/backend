import tempfile
import os
import logging
from db import SessionLocal
from models import SoilAnalysisReport
from Soil_Analysis.processing_soil_analysis import (
    extract_text_with_fallback,
    normalize_soil_text,
    call_gemini_soil_analysis
)

LOG = logging.getLogger(__name__)

def run_soil_analysis(
    farm_id: int,
    pdf_bytes: bytes,
    file_hash: str,
    file_name: str,
    user_id: int | None,
):
    session = SessionLocal()
    tmp_path = None

    try:
        # 1️⃣ Write bytes INSIDE worker container
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # 2️⃣ Process
        raw_text = extract_text_with_fallback(tmp_path)
        cleaned = normalize_soil_text(raw_text)
        llm_result = call_gemini_soil_analysis(cleaned)

        # 3️⃣ Save DB
        report = SoilAnalysisReport(
            user_id=user_id,
            farm_id=farm_id,
            file_name=file_name,
            file_hash=file_hash,
            raw_text=raw_text,
            cleaned_text=cleaned,
            soil_parameters=llm_result.get("soil_parameters"),
            advisory=llm_result.get("advisory"),
            confidence=llm_result.get("confidence"),
        )

        session.add(report)
        session.commit()

        return {"status": "done", "report_id": report.id}

    except Exception:
        session.rollback()
        LOG.exception("soil analysis failed")
        raise

    finally:
        session.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
