from db import SessionLocal
from models import ProcessResult

def save_process_result(farm_id, source, scene_id, start_ts, end_ts, result_json, health_score):
    sess = SessionLocal()
    try:
        pr = ProcessResult(
            farm_id=farm_id,
            source=source,
            scene_id=scene_id,
            start_ts=start_ts,
            end_ts=end_ts,
            result=result_json,
            health_score=health_score
        )
        sess.add(pr)
        sess.commit()
        sess.refresh(pr)
        return pr
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
