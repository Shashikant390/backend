# sh_app/cache.py
import json
import os
from typing import Any, Optional
import redis
import logging

logger = logging.getLogger("cache")

# Use config constants if available; fall back to env
REDIS_URL = os.environ.get("REDIS_URL", None)
DEFAULT_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 432000))  # 1 hour default

_redis_client: Optional[redis.Redis] = None

def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        logger.info("REDIS_URL not configured, cache disabled")
        return None
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping to fail fast
        _redis_client.ping()
        logger.info("Connected to Redis")
        return _redis_client
    except Exception as e:
        logger.exception("Failed to connect to redis: %s", e)
        _redis_client = None
        return None

def build_key(*parts: Any) -> str:
    """
    Safe key builder: joins parts with ':' after JSON-serializing complex structures.
    """
    out = []
    for p in parts:
        if p is None:
            out.append("null")
        elif isinstance(p, (str, int, float, bool)):
            out.append(str(p))
        else:
            # JSON canonical form is safer to avoid ordering issues
            try:
                out.append(json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            except Exception:
                out.append(str(p))
    return "shapp:" + ":".join(out)

make_key = build_key

def get_json(key: str) -> Optional[Any]:
    client = get_redis_client()
    if not client:
        return None
    try:
        v = client.get(key)
        if v is None:
            return None
        return json.loads(v)
    except Exception as e:
        logger.exception("cache get error for key=%s: %s", key, e)
        return None

def set_json(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    client = get_redis_client()
    if not client:
        return False
    try:
        payload = json.dumps(value, default=str, ensure_ascii=False)
        tt = DEFAULT_TTL if ttl is None else int(ttl)
        client.setex(key, tt, payload)
        return True
    except Exception as e:
        logger.exception("cache set error for key=%s: %s", key, e)
        return False
    
def delete(key: str) -> bool:
    client = get_redis_client()
    if not client:
        return False
    try:
        client.delete(key)
        return True
    except Exception as e:
        logger.exception("cache delete error for key=%s: %s", key, e)
        return False
