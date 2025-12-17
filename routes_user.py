# routes_user.py
from flask import Blueprint, request, jsonify, current_app
import repos
from auth import get_current_user_or_none
from typing import Dict, Any

user_bp = Blueprint("user", __name__, url_prefix="/user")


def _serialize_user(u) -> Dict[str, Any]:
    return {
        "id": u.id,
        "uid": u.uid,
        "email": u.email,
        "phone": u.phone,
        "display_name": getattr(u, "display_name", None),
        "photo_url": getattr(u, "photo_url", None),
        "meta": getattr(u, "meta", {}) or {},
        "quota": getattr(u, "quota", {}) or {},
        "roles": getattr(u, "roles", []) or [],
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_seen": u.last_seen.isoformat() if u.last_seen else None,
        "is_active": bool(u.is_active)
    }


@user_bp.route("/me", methods=["GET"])
def get_me():
    """
    Verify auth (firebase token or dev UID). If user row doesn't exist, create it.
    Returns serialized user object.
    """
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    # user is already a DB object thanks to auth.get_current_user_or_none()
    return jsonify(_serialize_user(user)), 200


@user_bp.route("/me", methods=["PATCH"])
def patch_me():
    """
    Update allowed profile fields: display_name, phone, meta (address etc.), is_active.
    Caller must be authenticated.
    """
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    # Disallow changing uid, id, roles, created_at from client
    for forbidden in ("uid", "id", "roles", "created_at"):
        payload.pop(forbidden, None)

    allowed_keys = {"display_name", "phone", "meta", "is_active"}
    updates = {k: v for k, v in payload.items() if k in allowed_keys}

    if not updates:
        return jsonify({"error": "no_allowed_fields"}), 400

    session = repos.get_session()
    try:
        updated_user = repos.update_user_profile(session, user.uid, updates)
        if not updated_user:
            return jsonify({"error": "not_found"}), 404

        # optional: invalidate cache here if you use cache keys
        try:
            import cache
            cache.delete(f"user:{user.uid}")
        except Exception:
            current_app.logger.debug("cache invalidation skipped or failed for user:%s", user.uid)

        return jsonify(_serialize_user(updated_user)), 200
    except Exception as e:
        current_app.logger.exception("update profile failed")
        return jsonify({"error": "update_failed", "detail": str(e)}), 500
    finally:
        session.close()


@user_bp.route("/signup", methods=["POST"])
def signup():
    """
    Optional endpoint: a single place to create and populate onboarding fields.
    Accepts display_name, phone, meta (e.g. address).
    This will create user if missing and update fields.
    """
    # authenticate first (this will create user row if missing because auth.get_current_user_or_none does that)
    user = get_current_user_or_none(request)
    if not user:
        return jsonify({"error": "unauthenticated"}), 401

    payload = request.get_json(force=True, silent=True) or {}
    allowed_keys = {"display_name", "phone", "meta"}
    updates = {k: v for k, v in payload.items() if k in allowed_keys}

    session = repos.get_session()
    try:
        # ensure user exists and apply updates
        updated_user = repos.update_user_profile(session, user.uid, updates) if updates else user
        return jsonify(_serialize_user(updated_user)), 201
    except Exception as e:
        current_app.logger.exception("signup failed")
        print("SIGNUP FAILED:", str(e))
        return jsonify({"error": "signup_failed", "detail": str(e)}), 500
    finally:
        session.close()
