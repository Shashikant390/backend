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
import json

logger = logging.getLogger("auth")

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



_firebase_app = None
import json

_firebase_app = None

def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    firebase_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")

    try:
        if firebase_json:
            # ✅ Railway / prod (env variable)
            creds_dict = json.loads(firebase_json)
            cred = credentials.Certificate(creds_dict)
        else:
            # ✅ Local dev (file on disk)
            cred = credentials.Certificate("firebase-adminsdk.json")

        _firebase_app = firebase_admin.initialize_app(cred)
        return _firebase_app

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Firebase Admin: {e}")
    


def _get_auth_header_token() -> Optional[str]:
    auth = flask_request.headers.get("Authorization", "")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return flask_request.headers.get("X-Firebase-Token")


def verify_firebase_token(id_token: str) -> Optional[dict]:
    
    try:
        decoded = fb_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.warning("Firebase token verification failed: %s", e)
        return None


def get_current_user_or_none(req=None) -> Optional[repos.models.AppUser]:
   
 
    r = req or flask_request

  
    dev_ok = os.environ.get("DEV_AUTH", "false").lower() in ("1", "true", "yes")
    if dev_ok:
        dev_uid = r.headers.get("X-DEV-UID") or r.args.get("dev_uid")
        if dev_uid:
            session = repos.get_session()
            try:
                user = repos.get_or_create_user_by_uid(session, dev_uid)
                return user
            finally:
                session.close()

    token = _get_auth_header_token()
    if not token:
        return None

   
    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
      
        try:
            import logging
            logging.getLogger("auth").warning("Firebase token verify failed: %s", e)
        except Exception:
            pass
        return None

    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        return None

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
