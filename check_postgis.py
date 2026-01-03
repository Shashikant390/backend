from db import engine
from sqlalchemy import text

with engine.connect() as conn:
    print("DB URL used by SQLAlchemy engine:", engine.url)
    try:
        print("postgis_full_version():", conn.execute(text("SELECT postgis_full_version()")).scalar())
    except Exception as e:
        print("postgis_full_version() error:", e)
    try:
        print("SRID 4326 count:", conn.execute(text("SELECT count(*) FROM spatial_ref_sys WHERE srid = 4326")).scalar())
    except Exception as e:
        print("spatial_ref_sys query error:", e)
