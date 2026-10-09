from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from backend.services.auth_service import AuthService, get_current_user

router = APIRouter(prefix="/auth", tags=["User Authentication"])

class SendOtpRequest(BaseModel):
    phone: str = Field(..., description="User phone number (e.g. +919876543210)")

class VerifyOtpRequest(BaseModel):
    phone: str
    otp: str
    name: Optional[str] = None

class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., description="Google Identity Services JWT credential")
    name: Optional[str] = None

@router.post("/phone/send-otp")
async def send_phone_otp(req: SendOtpRequest):
    """Generates and delivers a 6-digit OTP for phone number authentication."""
    if not req.phone or len(req.phone.strip()) < 7:
        raise HTTPException(status_code=400, detail="Please provide a valid phone number.")
    return AuthService.send_phone_otp(req.phone.strip())

@router.post("/phone/verify-otp")
async def verify_phone_otp(req: VerifyOtpRequest):
    """Verifies phone OTP and returns persistent session JWT token."""
    if not req.phone or not req.otp:
        raise HTTPException(status_code=400, detail="Phone number and OTP code are required.")
    return AuthService.verify_phone_otp(req.phone, req.otp, req.name)

@router.post("/google")
async def google_auth(req: GoogleAuthRequest):
    """Authenticates via Google Account One-Tap / Sign-In and returns session token."""
    return AuthService.authenticate_google(req.credential, req.name)

@router.get("/me")
async def get_my_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves authenticated user profile."""
    return {
        "status": "authenticated",
        "user": current_user
    }

@router.post("/logout")
async def logout():
    """Client logout confirmation."""
    return {"status": "logged_out", "message": "Successfully signed out."}
