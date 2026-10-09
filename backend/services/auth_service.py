import os
import re
import time
import uuid
import jwt
from typing import Dict, Any, Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.config import is_supabase_configured, SUPABASE_URL, SUPABASE_ANON_KEY
from backend.services.supabase_service import get_supabase_client, get_supabase_anon_client
from backend.database import DatabaseService

# Local JWT Fallback Secret & Expiry (used only when Supabase is not configured)
JWT_SECRET = os.getenv("JWT_SECRET", "vidara_secret_key_8923489237498234")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 30  # 30 days persistent session

security = HTTPBearer(auto_error=False)


class AuthService:
    """Production Supabase Authentication Service with local offline fallback."""

    EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,30}$")

    @classmethod
    def validate_password_policy(cls, password: str) -> None:
        if not password or len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
        if not any(c.isalpha() for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one letter.")
        if not any(c.isdigit() for c in password):
            raise HTTPException(status_code=400, detail="Password must contain at least one number.")

    @classmethod
    def validate_signup_inputs(cls, username: str, email: str, password: str, confirm_password: str) -> None:
        clean_username = (username or "").strip()
        clean_email = (email or "").strip().lower()

        if not clean_username:
            raise HTTPException(status_code=400, detail="Username is required.")
        if not cls.USERNAME_REGEX.match(clean_username):
            raise HTTPException(status_code=400, detail="Username must be 3-30 characters and contain only letters, numbers, hyphens, or underscores.")

        if not clean_email or not cls.EMAIL_REGEX.match(clean_email):
            raise HTTPException(status_code=400, detail="Please provide a valid email address.")

        if not password:
            raise HTTPException(status_code=400, detail="Password is required.")
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match.")

        cls.validate_password_policy(password)

    @classmethod
    def sign_up(cls, username: str, email: str, password: str, confirm_password: str) -> Dict[str, Any]:
        cls.validate_signup_inputs(username, email, password, confirm_password)
        clean_username = username.strip()
        clean_email = email.strip().lower()

        if is_supabase_configured():
            auth_client = get_supabase_anon_client() or get_supabase_client()
            admin_client = get_supabase_client()
            try:
                res = auth_client.auth.sign_up({
                    "email": clean_email,
                    "password": password,
                    "options": {
                        "data": {
                            "username": clean_username,
                            "display_name": clean_username
                        }
                    }
                })

                user = res.user
                if not user:
                    raise HTTPException(status_code=400, detail="Could not create user account.")

                # Supabase Gotrue returns an empty identities list when the email is already registered
                if getattr(user, "identities", None) is not None and len(user.identities) == 0:
                    raise HTTPException(status_code=400, detail="An account with this email address already exists. Please log in.")

                # Ensure profile exists in profiles table using admin client
                if admin_client:
                    try:
                        admin_client.table("profiles").upsert({
                            "id": user.id,
                            "display_name": clean_username,
                            "username": clean_username,
                            "email": clean_email,
                            "updated_at": "now()"
                        }).execute()
                    except Exception:
                        try:
                            admin_client.table("profiles").upsert({
                                "id": user.id,
                                "display_name": clean_username,
                                "updated_at": "now()"
                            }).execute()
                        except Exception as p_err:
                            print(f"Profile upsert note on signup: {p_err}")

                # Check if session token was returned (auto-confirmed) or email verification pending
                if res.session and res.session.access_token:
                    return {
                        "status": "authenticated",
                        "token": res.session.access_token,
                        "refresh_token": res.session.refresh_token,
                        "user": {
                            "id": user.id,
                            "email": user.email,
                            "username": clean_username,
                            "display_name": clean_username,
                            "avatar_url": ""
                        }
                    }
                else:
                    return {
                        "status": "verification_pending",
                        "message": "Account created! Please check your email inbox to verify your account before logging in.",
                        "user": {
                            "id": user.id,
                            "email": user.email,
                            "username": clean_username
                        }
                    }

            except HTTPException:
                raise
            except Exception as e:
                err_str = str(e)
                if "already registered" in err_str.lower() or "unique constraint" in err_str.lower() or "user already exists" in err_str.lower():
                    raise HTTPException(status_code=400, detail="An account with this email address already exists. Please log in.")
                raise HTTPException(status_code=400, detail=f"Signup failed: {err_str}")

        # Local Offline Fallback
        existing = DatabaseService.get_user_by_email(clean_email)
        if existing:
            raise HTTPException(status_code=400, detail="An account with this email already exists.")

        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        user = DatabaseService.upsert_user(
            user_id=user_id,
            name=clean_username,
            email=clean_email,
            auth_provider="email"
        )
        token = cls.create_fallback_jwt(user["id"], {"name": clean_username, "email": clean_email})
        return {
            "status": "authenticated",
            "token": token,
            "user": {
                "id": user["id"],
                "email": clean_email,
                "username": clean_username,
                "display_name": clean_username
            }
        }

    @classmethod
    def login(cls, email: str, password: str) -> Dict[str, Any]:
        clean_email = (email or "").strip().lower()
        if not clean_email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required.")

        if is_supabase_configured():
            auth_client = get_supabase_anon_client() or get_supabase_client()
            admin_client = get_supabase_client()
            try:
                res = auth_client.auth.sign_in_with_password({
                    "email": clean_email,
                    "password": password
                })
                if not res.session or not res.user:
                    raise HTTPException(status_code=401, detail="Invalid email or password.")

                user = res.user
                meta = user.user_metadata or {}
                display_name = meta.get("display_name") or meta.get("username") or clean_email.split("@")[0]
                username = meta.get("username") or display_name
                avatar_url = meta.get("avatar_url") or meta.get("picture") or ""

                # Ensure profile in profiles table
                if admin_client:
                    try:
                        admin_client.table("profiles").upsert({
                            "id": user.id,
                            "display_name": display_name,
                            "username": username,
                            "email": clean_email,
                            "avatar_url": avatar_url,
                            "updated_at": "now()"
                        }).execute()
                    except Exception:
                        try:
                            admin_client.table("profiles").upsert({
                                "id": user.id,
                                "display_name": display_name,
                                "avatar_url": avatar_url,
                                "updated_at": "now()"
                            }).execute()
                        except Exception as p_err:
                            print(f"Profile upsert note on login: {p_err}")

                return {
                    "status": "authenticated",
                    "token": res.session.access_token,
                    "refresh_token": res.session.refresh_token,
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": username,
                        "display_name": display_name,
                        "avatar_url": avatar_url
                    }
                }
            except HTTPException:
                raise
            except Exception as e:
                err_str = str(e)
                if "invalid login credentials" in err_str.lower() or "invalid_credentials" in err_str.lower():
                    raise HTTPException(status_code=401, detail="Invalid email or password. Please try again.")
                if "email not confirmed" in err_str.lower():
                    raise HTTPException(status_code=400, detail="Please verify your email address before logging in.")
                raise HTTPException(status_code=400, detail=f"Login error: {err_str}")

        # Local Offline Fallback
        user = DatabaseService.get_user_by_email(clean_email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = cls.create_fallback_jwt(user["id"], {"name": user["name"], "email": clean_email})
        return {
            "status": "authenticated",
            "token": token,
            "user": user
        }

    @classmethod
    def forgot_password(cls, email: str, redirect_url: Optional[str] = None) -> Dict[str, Any]:
        clean_email = (email or "").strip().lower()
        if not clean_email or not cls.EMAIL_REGEX.match(clean_email):
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")

        if is_supabase_configured():
            auth_client = get_supabase_anon_client() or get_supabase_client()
            try:
                options = {"redirect_to": redirect_url} if redirect_url else None
                auth_client.auth.reset_password_for_email(clean_email, options=options)
                return {
                    "status": "success",
                    "message": f"Password reset instructions have been sent to {clean_email}."
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to send reset link: {str(e)}")

        return {
            "status": "success",
            "message": f"Password reset instructions sent to {clean_email}."
        }

    @classmethod
    def reset_password(cls, new_password: str, confirm_password: str, token: Optional[str] = None) -> Dict[str, Any]:
        if new_password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match.")
        cls.validate_password_policy(new_password)

        if is_supabase_configured():
            admin_client = get_supabase_client()
            try:
                if token:
                    user_res = admin_client.auth.get_user(token)
                    if not user_res or not user_res.user:
                        raise HTTPException(status_code=401, detail="Invalid or expired reset token.")
                    admin_client.auth.admin.update_user_by_id(user_res.user.id, {"password": new_password})
                else:
                    admin_client.auth.update_user({"password": new_password})
                return {
                    "status": "success",
                    "message": "Password updated successfully. You can now log in with your new password."
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to reset password: {str(e)}")

        return {"status": "success", "message": "Password updated successfully."}

    @classmethod
    def get_google_oauth_url(cls, redirect_url: Optional[str] = None) -> Dict[str, Any]:
        if not is_supabase_configured():
            raise HTTPException(status_code=400, detail="Supabase is not configured.")

        auth_client = get_supabase_anon_client() or get_supabase_client()
        try:
            target_redirect = redirect_url or f"http://127.0.0.1:8000/"
            res = auth_client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": target_redirect
                }
            })
            return {
                "status": "success",
                "url": res.url
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Google OAuth initialization error: {str(e)}")

    @classmethod
    def verify_supabase_token(cls, token: str) -> Optional[Dict[str, Any]]:
        """Verifies JWT with Supabase Auth server-side."""
        if not token:
            return None
        client = get_supabase_client() or get_supabase_anon_client()
        if not client:
            return None

        try:
            user_res = client.auth.get_user(token)
            if not user_res or not user_res.user:
                return None

            user = user_res.user
            meta = user.user_metadata or {}
            display_name = meta.get("display_name") or meta.get("username") or meta.get("name") or (user.email.split("@")[0] if user.email else "User")
            username = meta.get("username") or display_name
            avatar_url = meta.get("avatar_url") or meta.get("picture") or ""

            return {
                "id": str(user.id),
                "email": user.email or "",
                "name": display_name,
                "display_name": display_name,
                "username": username,
                "avatar_url": avatar_url,
                "auth_provider": "supabase"
            }
        except Exception:
            return None

    @classmethod
    def create_fallback_jwt(cls, user_id: str, payload_data: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time() + (JWT_EXPIRATION_HOURS * 3600)),
        }
        if payload_data:
            payload.update(payload_data)
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @classmethod
    def verify_fallback_jwt(cls, token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except Exception:
            return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reusable FastAPI authentication dependency.
    Validates Supabase Auth token (from Bearer Authorization header or ?token= query param)
    and enforces authenticated user identity server-side.
    """
    resolved_token = None
    if credentials and credentials.credentials:
        resolved_token = credentials.credentials.strip()
    elif token:
        resolved_token = token.strip()

    if not resolved_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    # 1. Try Supabase Auth verification
    if is_supabase_configured():
        verified_user = AuthService.verify_supabase_token(resolved_token)
        if verified_user:
            return verified_user

    # 2. Try Fallback JWT verification
    decoded = AuthService.verify_fallback_jwt(resolved_token)
    if decoded and "sub" in decoded:
        user = DatabaseService.get_user_by_id(decoded["sub"])
        if user:
            return user
        return {
            "id": decoded["sub"],
            "name": decoded.get("name", "Vidara User"),
            "display_name": decoded.get("name", "Vidara User"),
            "email": decoded.get("email", ""),
            "avatar_url": "",
            "auth_provider": "local"
        }

    raise HTTPException(status_code=401, detail="Invalid or expired session token. Please log in again.")


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Optional authentication dependency; returns user dict if valid, else None."""
    resolved_token = None
    if credentials and credentials.credentials:
        resolved_token = credentials.credentials.strip()
    elif token:
        resolved_token = token.strip()

    if not resolved_token:
        return None

    if is_supabase_configured():
        verified = AuthService.verify_supabase_token(resolved_token)
        if verified:
            return verified

    decoded = AuthService.verify_fallback_jwt(resolved_token)
    if decoded and "sub" in decoded:
        user = DatabaseService.get_user_by_id(decoded["sub"])
        if user:
            return user
        return {
            "id": decoded["sub"],
            "name": decoded.get("name", "Vidara User"),
            "display_name": decoded.get("name", "Vidara User"),
            "email": decoded.get("email", ""),
            "avatar_url": ""
        }

    return None
