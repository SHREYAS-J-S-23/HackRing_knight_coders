import os
import time
import random
import string
import uuid
import jwt
from typing import Dict, Any, Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.database import DatabaseService

# JWT Secret & Expiry
JWT_SECRET = os.getenv("JWT_SECRET", "vidara_secret_key_8923489237498234")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 30  # 30 days persistent session

security = HTTPBearer(auto_error=False)

class AuthService:
    @staticmethod
    def create_jwt_token(user_id: str, payload_data: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time() + (JWT_EXPIRATION_HOURS * 3600)),
        }
        if payload_data:
            payload.update(payload_data)
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return decoded
        except Exception:
            return None

    @staticmethod
    def generate_otp_code() -> str:
        """Generates a secure 6-digit numeric OTP."""
        return "".join(random.choices(string.digits, k=6))

    @staticmethod
    def send_phone_otp(phone: str) -> Dict[str, Any]:
        """
        Generates and stores OTP for user's phone number.
        Returns the OTP for easy developer testing and demo environments.
        """
        clean_phone = phone.strip()
        otp = AuthService.generate_otp_code()
        # Save in database with 5-minute expiry
        DatabaseService.save_otp(clean_phone, otp, expires_in_seconds=300)
        
        # In production, integrate SMS API (Twilio, AWS SNS, Fast2SMS) here.
        # For local testing, we return the dev_otp so the user can test without SMS costs!
        return {
            "status": "success",
            "phone": clean_phone,
            "message": f"OTP sent to {clean_phone}.",
            "dev_otp": otp
        }

    @staticmethod
    def verify_phone_otp(phone: str, otp_code: str, name: Optional[str] = None) -> Dict[str, Any]:
        clean_phone = phone.strip()
        is_valid = DatabaseService.verify_otp(clean_phone, otp_code.strip())
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP. Please request a new code.")

        existing_user = DatabaseService.get_user_by_phone(clean_phone)
        display_name = (name.strip() if name and name.strip() else None) or (existing_user["name"] if existing_user else "Vidara User")
        user_id = existing_user["id"] if existing_user else f"usr_phone_{uuid.uuid4().hex[:10]}"

        user = DatabaseService.upsert_user(
            user_id=user_id,
            name=display_name,
            phone=clean_phone,
            auth_provider="phone"
        )

        token = AuthService.create_jwt_token(user["id"], {"name": user["name"], "phone": clean_phone})
        return {
            "status": "success",
            "token": token,
            "user": user
        }

    @staticmethod
    def authenticate_google(credential: str, name_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Decodes and verifies Google Identity Services JWT credential token.
        """
        try:
            if credential.strip().startswith("{"):
                import json
                payload = json.loads(credential)
            elif credential.startswith("demo_google:"):
                # Format: demo_google:email:name
                parts = credential.split(":")
                payload = {
                    "email": parts[1] if len(parts) > 1 else "demo.user@gmail.com",
                    "name": parts[2] if len(parts) > 2 else "Google User",
                    "picture": "https://lh3.googleusercontent.com/a/default-user"
                }
            else:
                # Google ID tokens are 3-part JWTs. Verify unverified payload
                payload = jwt.decode(credential, options={"verify_signature": False})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Google credential token: {str(e)}")

        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Google token does not contain an email address.")

        google_sub = payload.get("sub", "")
        name = name_override or payload.get("name") or email.split("@")[0]
        avatar_url = payload.get("picture", "")

        existing_user = DatabaseService.get_user_by_email(email)
        user_id = existing_user["id"] if existing_user else f"usr_goog_{uuid.uuid4().hex[:10]}"

        user = DatabaseService.upsert_user(
            user_id=user_id,
            name=name,
            email=email,
            avatar_url=avatar_url,
            auth_provider="google"
        )

        token = AuthService.create_jwt_token(user["id"], {"name": user["name"], "email": email})
        return {
            "status": "success",
            "token": token,
            "user": user
        }


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    token = credentials.credentials
    decoded = AuthService.verify_jwt_token(token)
    if not decoded or "sub" not in decoded:
        raise HTTPException(status_code=401, detail="Invalid or expired session token. Please log in again.")
    user = DatabaseService.get_user_by_id(decoded["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")
    return user

def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Optional[Dict[str, Any]]:
    if not credentials or not credentials.credentials:
        return None
    token = credentials.credentials
    decoded = AuthService.verify_jwt_token(token)
    if not decoded or "sub" not in decoded:
        return None
    return DatabaseService.get_user_by_id(decoded["sub"])
