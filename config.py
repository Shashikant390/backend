import os
from dotenv import load_dotenv

load_dotenv()

SH_CLIENT_ID: str = os.environ.get("SH_CLIENT_ID", "")
SH_CLIENT_SECRET: str = os.environ.get("SH_CLIENT_SECRET", "")
SH_ACCESS_TOKEN: str | None = os.environ.get("SH_ACCESS_TOKEN")  # optional long-lived token

TOKEN_URL: str = os.environ.get(
    "TOKEN_URL",
    "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
)
CATALOG_URL: str = os.environ.get(
    "CATALOG_URL",
    "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
)
PROCESS_URL: str = os.environ.get(
    "PROCESS_URL",
    "https://services.sentinel-hub.com/api/v1/process"
)

COL_S2: str | None = "sentinel-2-l2a"


AVAILABLE_COLLECTIONS: list[str] = []

TOKEN_CACHE = {"access_token": None, "expires_at": 0}

DATABASE_URL = os.environ.get("DATABASE_URL", "")
REDIS_URL = os.environ.get("REDIS_URL")

PROCESS_OR_REFRESH_URL = os.environ.get("PROCESS_OR_REFRESH_URL")

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 432000))

FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

AVAILABLE_COLLECTIONS = []


PROCESS_STALE_DAYS = 5     
PROCESS_OR_REFRESH_TIMEOUT = 120