from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.database import DatabaseService
from backend.services.auth_service import get_current_user

router = APIRouter(prefix="/library", tags=["User Library & Saved Clips"])

class SaveClipRequest(BaseModel):
    video_id: str
    clip_id: str
    title: str
    video_title: Optional[str] = "Video"
    start_time: float
    end_time: float
    duration: float
    download_url: Optional[str] = ""
    video_url: Optional[str] = ""
    filepath: Optional[str] = None

@router.get("/clips")
async def get_saved_clips(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves all permanently saved clips for the logged-in user."""
    user_id = current_user["id"]
    clips = DatabaseService.get_user_saved_clips(user_id)
    return {
        "status": "success",
        "user_id": user_id,
        "count": len(clips),
        "clips": clips
    }

@router.post("/clips/save")
async def save_clip_to_library(req: SaveClipRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Saves a generated or previewed topic clip permanently to the user's dashboard."""
    user_id = current_user["id"]
    
    # Auto-resolve URLs if missing
    download_url = req.download_url or f"/api/videos/{req.video_id}/clip/{req.clip_id}"
    video_url = req.video_url or f"/api/videos/{req.video_id}/clip/{req.clip_id}"

    saved_record = DatabaseService.save_user_clip(
        user_id=user_id,
        video_id=req.video_id,
        clip_id=req.clip_id,
        title=req.title,
        video_title=req.video_title or "Video Highlight",
        start_time=req.start_time,
        end_time=req.end_time,
        duration=req.duration,
        download_url=download_url,
        video_url=video_url,
        filepath=req.filepath
    )

    return {
        "status": "success",
        "message": f"'{req.title}' permanently saved to your library.",
        "clip": saved_record
    }

@router.delete("/clips/{clip_id}")
async def delete_saved_clip(clip_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Permanently deletes a saved clip from the user's dashboard."""
    user_id = current_user["id"]
    deleted = DatabaseService.delete_user_saved_clip(user_id, clip_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Clip not found in your library.")

    return {
        "status": "success",
        "message": "Clip successfully removed from your library."
    }

@router.get("/videos")
async def get_user_videos(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves all previously uploaded or analyzed videos for this user."""
    user_id = current_user["id"]
    videos = DatabaseService.get_user_videos(user_id)
    return {
        "status": "success",
        "count": len(videos),
        "videos": videos
    }
