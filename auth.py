# auth.py
import os
import time
import logging
from typing import Optional

import firebase_admin
from firebase_admin import auth as fb_auth, credentials
from flask import request as flask_request
import repos
from utils import request_with_retries
import config as config

logger = logging.getLogger("auth")

# ------------ Sentinel Hub token helper (kept here for convenience) ------------
def _fetch_new_token() -> str:
    if not config.SH_CLIENT_ID or not config.SH_CLIENT_SECRET:
        raise RuntimeError("Set SH_CLIENT_ID & SH_CLIENT_SECRET or SH_ACCESS_TOKEN in environment")

    payload = {
        "grant_type": "client_credentials",
        "client_id": config.SH_CLIENT_ID,
        "client_secret": config.SH_CLIENT_SECRET
    }
    r = request_with_retries("POST", config.TOKEN_URL, data=payload, timeout=30)
    try:
        r.raise_for_status()
    except Exception:
        body = getattr(r, "text", str(r))
        logger.exception("TOKEN ENDPOINT ERROR: %s", body[:2000])
        raise RuntimeError(f"token request failed: {body[:2000]}")

    j = r.json()
    tok = j.get("access_token")
    exp = j.get("expires_in", 3600)
    if not tok:
        raise RuntimeError("token response did not contain access_token")

    config.TOKEN_CACHE["access_token"] = tok
    config.TOKEN_CACHE["expires_at"] = time.time() + int(exp) - 30
    return tok


def get_sh_token() -> str:
    if getattr(config, "SH_ACCESS_TOKEN", None):
        return config.SH_ACCESS_TOKEN

    if config.TOKEN_CACHE.get("access_token") and time.time() < config.TOKEN_CACHE.get("expires_at", 0):
        return config.TOKEN_CACHE["access_token"]

    return _fetch_new_token()


# ------------ Firebase init ------------
_firebase_app = None
def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    cred_path = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if not cred_path:
        logger.info("FIREBASE_CREDENTIALS_JSON not set; Firebase Admin will not be initialized.")
        return None
    try:
        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin initialized.")
        return _firebase_app
    except Exception as e:
        logger.exception("Failed to initialize Firebase Admin: %s", e)
        return None

# Try initialize but don't crash import if something fails
try:
    _init_firebase()
except Exception:
    pass


# ------------ Helpers exposed to routes ------------

def _get_auth_header_token() -> Optional[str]:
    auth = flask_request.headers.get("Authorization", "")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    # alternative dev header
    return flask_request.headers.get("X-Firebase-Token")  # optional dev header


def verify_firebase_token(id_token: str) -> Optional[dict]:
    """
    Verify a Firebase ID token and return decoded claims dict or None on failure.
    """
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.warning("Firebase token verification failed: %s", e)
        return None


def get_current_user_or_none(req=None) -> Optional[repos.models.AppUser]:
    """
    Primary function routes expect. Verify Firebase token in the incoming Flask request
    (or provided `req` flask request-like object). Returns AppUser SQLAlchemy object
    (creates row if missing) or None when unauthenticated.

    Dev/testing fallback:
      - If environment variable DEV_AUTH=true, you can also provide header X-DEV-UID: <firebase-uid>
        or query param ?dev_uid=<firebase-uid> to emulate authentication (ONLY for local dev).
    """
    # allow passing request object for testability; otherwise use flask.request
    r = req or flask_request

    # Dev override (useful for Postman local tests)
    dev_ok = os.environ.get("DEV_AUTH", "false").lower() in ("1", "true", "yes")
    if dev_ok:
        dev_uid = r.headers.get("X-DEV-UID") or r.args.get("dev_uid")
        if dev_uid:
            # return db user (create if missing)
            session = repos.get_session()
            try:
                user = repos.get_or_create_user_by_uid(session, dev_uid)
                return user
            finally:
                session.close()

    # normal flow: check Authorization header for Firebase ID token
    token = _get_auth_header_token()
    if not token:
        return None

    # verify token with Firebase Admin SDK
    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
        # invalid/expired token
        try:
            import logging
            logging.getLogger("auth").warning("Firebase token verify failed: %s", e)
        except Exception:
            pass
        return None

    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        return None

    # create or fetch corresponding AppUser row using a session provided to repos
    session = repos.get_session()
    try:
        user = repos.get_or_create_user_by_uid(
            session,
            firebase_uid,
            email=decoded.get("email"),
            phone=decoded.get("phone_number")
        )
        return user
    finally:
        session.close()
