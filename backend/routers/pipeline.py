import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, AUDIO_DIR, OUTPUT_DIR
from backend.services.audio_extractor import AudioExtractor
from backend.services.groq_stt import GroqSTTService
from backend.services.semantic_engine import SemanticEngine
from backend.services.meaning_validator import MeaningValidator
from backend.services.video_cutter import VideoCutter
from backend.routers.audience import DEFAULT_PROFILES
from backend.models.schemas import (
    EnrichedTranscript,
    IntentGraph,
    AudienceProfile,
    EditDecisionList,
    EditDecision,
    MeaningValidationReport
)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# In-memory store for jobs/videos (can be backed by SQLite/JSON)
JOBS_DB: Dict[str, Dict[str, Any]] = {}

class GenerateEDLRequest(BaseModel):
    video_id: str
    audience_id: str
    custom_profile: Optional[AudienceProfile] = None

class RenderVideoRequest(BaseModel):
    video_id: str
    audience_id: str
    overrides: Optional[Dict[int, str]] = None  # {segment_id: "KEEP" | "CUT"}

@router.post("/upload")
async def upload_video(file: UploadFile = File(...), groq_api_key: Optional[str] = Form(None)):
    """
    Step 1: Uploads video, extracts audio, transcribes with Groq Whisper, and builds Intent Graph.
    """
    from backend.config import set_groq_api_key, get_groq_api_key
    if groq_api_key and groq_api_key.strip():
        set_groq_api_key(groq_api_key.strip())

    if not get_groq_api_key():
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is missing! Click the 'Keys' button in the navbar or set it in .env so ReelForge can perform 100% real transcription and analysis on your video."
        )
    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    saved_video_path = UPLOAD_DIR / f"{video_id}{ext}"

    # Save uploaded file
    with open(saved_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract audio using FFmpeg
        audio_path = AudioExtractor.extract_audio(saved_video_path, f"{video_id}.mp3")
        duration = AudioExtractor.get_video_duration(saved_video_path)

        # Transcribe with Groq Whisper
        stt = GroqSTTService()
        transcript = stt.transcribe(audio_path, video_id, duration)

        # Build Intent Graph with Groq / Agnes LLM
        semantic_engine = SemanticEngine()
        intent_graph = semantic_engine.build_intent_graph(transcript)

        JOBS_DB[video_id] = {
            "video_id": video_id,
            "filename": file.filename,
            "video_path": str(saved_video_path),
            "audio_path": str(audio_path),
            "duration": duration,
            "transcript": transcript,
            "intent_graph": intent_graph,
            "edls": {},
            "rendered_outputs": {}
        }

        return {
            "status": "success",
            "video_id": video_id,
            "filename": file.filename,
            "duration_seconds": duration,
            "transcript": transcript.model_dump(),
            "intent_graph": intent_graph.model_dump()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@router.post("/generate-edl")
async def generate_edl(req: GenerateEDLRequest):
    """
    Step 2: Builds audience-specific Edit Decision List and performs Meaning-Preservation Check.
    """
    job = JOBS_DB.get(req.video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video session not found.")

    # Match audience profile
    audience = None
    if req.custom_profile:
        audience = req.custom_profile
    else:
        for p in DEFAULT_PROFILES:
            if p.id == req.audience_id:
                audience = p
                break
    
    if not audience:
        audience = DEFAULT_PROFILES[0]

    semantic_engine = SemanticEngine()
    validator = MeaningValidator()

    # Generate EDL
    edl = semantic_engine.generate_edl(job["transcript"], job["intent_graph"], audience)
    
    # Run Meaning-Preservation Validation Check
    val_report = validator.validate_edl(job["transcript"], job["intent_graph"], edl)

    job["edls"][req.audience_id] = {
        "edl": edl,
        "validation_report": val_report
    }

    return {
        "status": "success",
        "edl": edl.model_dump(),
        "validation_report": val_report.model_dump()
    }

@router.post("/render")
async def render_video(req: RenderVideoRequest):
    """
    Step 3: Deterministically renders final video using FFmpeg according to the EDL (with any creator overrides).
    """
    job = JOBS_DB.get(req.video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video session not found.")

    edl_data = job["edls"].get(req.audience_id)
    if not edl_data:
        raise HTTPException(status_code=400, detail="EDL has not been generated for this audience yet.")

    edl: EditDecisionList = edl_data["edl"]

    # Apply creator manual overrides if any
    if req.overrides:
        for decision in edl.decisions:
            if decision.segment_id in req.overrides:
                override_action = req.overrides[decision.segment_id]
                if override_action in ["KEEP", "CUT"]:
                    decision.action = override_action
                    decision.reason = f"Creator manual override to {override_action}"

    # Deterministic cutting with FFmpeg (both individual clips and concatenated merged video)
    video_path = Path(job["video_path"])
    output_filename = f"{req.video_id}_{req.audience_id}_final.mp4"
    rendered_path, clips = VideoCutter.render_edited_video(video_path, edl, output_filename)

    # Subtitles VTT synchronized with merged spans
    vtt_path = OUTPUT_DIR / f"{req.video_id}_{req.audience_id}.vtt"
    VideoCutter.export_subtitles_vtt(edl, vtt_path, video_path)

    job["rendered_outputs"][req.audience_id] = {
        "video_path": str(rendered_path),
        "vtt_path": str(vtt_path),
        "filename": output_filename,
        "clips": clips
    }

    return {
        "status": "success",
        "message": "Video successfully rendered deterministically.",
        "download_url": f"/api/pipeline/download/{req.video_id}/{req.audience_id}",
        "subtitles_url": f"/api/pipeline/subtitles/{req.video_id}/{req.audience_id}",
        "clips": clips,
        "stats": {
            "original_duration": edl.original_duration,
            "final_duration": edl.edited_duration,
            "compression_percent": edl.compression_percent,
            "segments_kept": edl.kept_count,
            "segments_cut": edl.cut_count,
            "clips_count": len(clips)
        }
    }

@router.get("/session/{video_id}")
async def get_session(video_id: str):
    """Retrieves full session data for inspectability."""
    job = JOBS_DB.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "video_id": video_id,
        "filename": job["filename"],
        "duration": job["duration"],
        "transcript": job["transcript"].model_dump(),
        "intent_graph": job["intent_graph"].model_dump(),
        "available_edls": list(job["edls"].keys())
    }

@router.get("/download/{video_id}/{audience_id}")
async def download_video(video_id: str, audience_id: str):
    """Download or stream the final edited video file."""
    job = JOBS_DB.get(video_id)
    target_path = None
    target_filename = f"{video_id}_{audience_id}_final.mp4"

    if job:
        out = job.get("rendered_outputs", {}).get(audience_id)
        if out and Path(out["video_path"]).exists():
            target_path = Path(out["video_path"])
            target_filename = out.get("filename", target_filename)

    # Fallback directly to disk in case backend reloaded
    if not target_path or not target_path.exists():
        fallback = OUTPUT_DIR / f"{video_id}_{audience_id}_final.mp4"
        if fallback.exists():
            target_path = fallback

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found. Please render first.")
    
    return FileResponse(
        path=str(target_path),
        filename=target_filename,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{target_filename}"',
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )

@router.get("/clip/{video_id}/{audience_id}/{clip_index}")
async def get_clip(video_id: str, audience_id: str, clip_index: int):
    """Download or stream an individual cut clip."""
    clip_filename = f"{video_id}_{audience_id}_clip_{clip_index}.mp4"
    clip_path = OUTPUT_DIR / clip_filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Individual clip not found.")
    
    return FileResponse(
        path=str(clip_path),
        filename=clip_filename,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{clip_filename}"',
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )

@router.get("/subtitles/{video_id}/{audience_id}")
async def get_subtitles(video_id: str, audience_id: str):
    """Fetch generated WebVTT subtitles."""
    job = JOBS_DB.get(video_id)
    target_path = None

    if job:
        out = job.get("rendered_outputs", {}).get(audience_id)
        if out and Path(out["vtt_path"]).exists():
            target_path = Path(out["vtt_path"])

    if not target_path or not target_path.exists():
        fallback = OUTPUT_DIR / f"{video_id}_{audience_id}.vtt"
        if fallback.exists():
            target_path = fallback

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found.")
    
    return FileResponse(
        path=str(target_path),
        filename=f"{video_id}_{audience_id}.vtt",
        media_type="text/vtt",
        headers={
            "Content-Disposition": f'attachment; filename="{video_id}_{audience_id}.vtt"',
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )

class ConfigRequest(BaseModel):
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None

@router.post("/config")
async def update_config(req: ConfigRequest):
    """Allows updating Groq API Key dynamically from the UI."""
    from backend.config import set_groq_api_key
    if req.groq_api_key and req.groq_api_key.strip():
        set_groq_api_key(req.groq_api_key.strip())
        return {"status": "success", "message": "Groq API key updated and stored successfully."}
    return {"status": "error", "message": "Key cannot be empty."}

