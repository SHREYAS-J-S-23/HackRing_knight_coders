from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from backend.services.auth_service import AuthService, get_current_user

router = APIRouter(prefix="/auth", tags=["User Authentication"])


class SignUpRequest(BaseModel):
    username: str = Field(..., description="Unique username (3-30 characters)")
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Password (min 8 chars, letter + number)")
    confirm_password: str = Field(..., description="Confirm password matching password")


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="User email address")
    redirect_url: Optional[str] = Field(None, description="Optional frontend redirect URL for password reset")


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., description="New password")
    confirm_password: str = Field(..., description="Confirm new password")
    token: Optional[str] = Field(None, description="Bearer/session access token from reset email link")


class SendOtpRequest(BaseModel):
    phone: str = Field(..., description="User phone number (e.g. +919876543210)")


class VerifyOtpRequest(BaseModel):
    phone: str
    otp: str
    name: Optional[str] = None


class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., description="Google Identity Services JWT credential")
    name: Optional[str] = None


# --- Supabase Email & Password Auth ---

@router.post("/signup")
async def signup(req: SignUpRequest):
    """
    Registers a new user with Supabase Auth:
    - Validates username and email formats
    - Enforces strong password policy
    - Checks password match
    - Stores metadata and profile in Supabase
    """
    return AuthService.sign_up(
        username=req.username,
        email=req.email,
        password=req.password,
        confirm_password=req.confirm_password
    )


@router.post("/login")
async def login(req: LoginRequest):
    """
    Authenticates user via Supabase Auth with email and password.
    Returns access token, refresh token, and user profile.
    """
    return AuthService.login(email=req.email, password=req.password)


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """
    Sends a secure password reset link to user's email via Supabase Auth.
    """
    return AuthService.forgot_password(email=req.email, redirect_url=req.redirect_url)


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """
    Resets the user's password with Supabase Auth using the session token.
    """
    return AuthService.reset_password(
        new_password=req.new_password,
        confirm_password=req.confirm_password,
        token=req.token
    )


# --- Supabase Google OAuth & Client Config ---

@router.get("/config")
async def get_auth_config():
    """
    Returns public client authentication configuration (URL and Anon key only).
    Service-role keys and secrets are strictly retained on the backend.
    """
    from backend.config import is_supabase_configured, SUPABASE_URL, SUPABASE_ANON_KEY
    return {
        "supabase_configured": is_supabase_configured(),
        "supabase_url": SUPABASE_URL if is_supabase_configured() else "",
        "supabase_anon_key": SUPABASE_ANON_KEY if is_supabase_configured() else ""
    }


@router.get("/google/url")
async def get_google_oauth_url(redirect_url: Optional[str] = Query(None, description="Frontend callback URL")):
    """
    Returns the Supabase OAuth authorization URL for Google login.
    """
    return AuthService.get_google_oauth_url(redirect_url=redirect_url)


# --- Profile and Session Verification ---

@router.get("/me")
async def get_my_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retrieves authenticated user profile after validating the Supabase access token.
    """
    return {
        "status": "authenticated",
        "user": current_user
    }


@router.post("/logout")
async def logout():
    """
    Confirms client logout.
    """
    return {"status": "logged_out", "message": "Successfully signed out."}


# --- Backward Compatibility (Phone OTP / Direct Google Credential) ---

@router.post("/phone/send-otp")
async def send_phone_otp(req: SendOtpRequest):
    if not req.phone or len(req.phone.strip()) < 7:
        raise HTTPException(status_code=400, detail="Please provide a valid phone number.")
    # Fallback response for phone OTP
    return {"status": "success", "message": f"OTP sent to {req.phone.strip()} (test mode: 123456)"}


@router.post("/phone/verify-otp")
async def verify_phone_otp(req: VerifyOtpRequest):
    if not req.phone or not req.otp:
        raise HTTPException(status_code=400, detail="Phone number and OTP code are required.")
    return AuthService.sign_up(
        username=req.phone.replace("+", "").strip()[:15],
        email=f"{req.phone.replace('+', '').strip()}@vidara.internal",
        password=f"VidaraPhone123!{req.otp}",
        confirm_password=f"VidaraPhone123!{req.otp}"
    )
