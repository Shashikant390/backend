# init_app.py
import config
from catalog import list_collections
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)



def init_available_collections() -> None:
    try:
        j = list_collections()
        config.AVAILABLE_COLLECTIONS = [c.get("id") for c in j.get("collections", [])]
        print("AVAILABLE_COLLECTIONS:", config.AVAILABLE_COLLECTIONS)

        # Sanity: ensure main constants refer to actual available values; adjust/fallback/disable as needed
        if config.COL_S2 not in config.AVAILABLE_COLLECTIONS:
            if "sentinel-2-l1c" in config.AVAILABLE_COLLECTIONS:
                print("COL_S2 'sentinel-2-l2a' not present, falling back to sentinel-2-l1c")
                config.COL_S2 = "sentinel-2-l1c"
            else:
                print("COL_S2 not available; disabling S2 processing")
                config.COL_S2 = None

        if config.COL_S1 not in config.AVAILABLE_COLLECTIONS:
            print("COL_S1 not available; disabling S1 processing")
            config.COL_S1 = None

        if config.COL_L8 not in config.AVAILABLE_COLLECTIONS:
            print("COL_L8 not available; disabling L8 processing")
            config.COL_L8 = None

    except Exception as e:
        print("init_available_collections failed:", str(e))
        # leave AVAILABLE_COLLECTIONS empty -> validation will be bypassed but functions will error clearly later
