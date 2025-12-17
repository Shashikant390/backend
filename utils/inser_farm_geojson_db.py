from sqlalchemy import text
import json
from db import SessionLocal

def create_farm_geojson_db(user_id, name, geojson_obj, meta=None):
    sess = SessionLocal()
    try:
        geojson_str = json.dumps(geojson_obj)
        # Use ST_SetSRID(ST_GeomFromGeoJSON(...),4326)
        stmt = text("""
        INSERT INTO farm (user_id, name, geom, bbox, area_m2, meta)
        VALUES (:user_id, :name,
                ST_SetSRID(ST_GeomFromGeoJSON(:geojson)::geometry, 4326),
                ST_Envelope(ST_SetSRID(ST_GeomFromGeoJSON(:geojson)::geometry,4326)),
                ST_Area(ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(:geojson)::geometry,4326),3857)),
                :meta)
        RETURNING id;
        """)
        res = sess.execute(stmt, {"user_id": user_id, "name": name, "geojson": geojson_str, "meta": json.dumps(meta or {})})
        farm_id = res.scalar_one()
        sess.commit()
        return farm_id
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
