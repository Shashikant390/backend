from sqlalchemy import text, select, func
from db import SessionLocal
from models import Farm

def update_area(farm_id):
    sess = SessionLocal()
    try:
        # compute in DB to avoid projection issues
        area_q = sess.execute(
            select(func.ST_Area(func.ST_Transform(Farm.geom, 3857))).where(Farm.id == farm_id)
        )
        area_val = area_q.scalar_one()
        if area_val is not None:
            sess.execute(
                text("UPDATE farm SET area_m2 = :area WHERE id = :fid"),
                {"area": float(area_val), "fid": farm_id}
            )
            sess.commit()
        return area_val
    finally:
        sess.close()
