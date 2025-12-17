# requires shapely install: pip install shapely
import json
from shapely.geometry import shape
from geoalchemy2 import WKTElement
from db import SessionLocal
from models import Farm

def create_farm_from_geojson(user_id, name, geojson_obj, meta=None):
    sess = SessionLocal()
    try:
        # convert to WKT + set SRID 4326
        geom_wkt = shape(geojson_obj).wkt
        geom = WKTElement(geom_wkt, srid=4326)

        farm = Farm(user_id=user_id, name=name, geom=geom, meta=meta)
        sess.add(farm)
        sess.commit()
        sess.refresh(farm)
        return farm
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
