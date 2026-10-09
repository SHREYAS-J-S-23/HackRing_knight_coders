import uuid
import shutil
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from backend.services.auth_service import get_current_user, get_optional_current_user

from backend.config import (
    UPLOAD_DIR,
    AUDIO_DIR,
    OUTPUT_DIR,
    get_groq_api_key,
    get_groq_stt_api_key,
    get_groq_llm_api_key,
    set_groq_api_key,
    is_supabase_configured,
    SUPABASE_STORAGE_BUCKET_VIDEOS,
    SUPABASE_STORAGE_BUCKET_AUDIO,
    SUPABASE_STORAGE_BUCKET_CLIPS,
    SUPABASE_STORAGE_BUCKET_SUBTITLES,
    SUPABASE_STORAGE_BUCKET_EXPORTS,
)
from backend.database import DatabaseService
from backend.services.audio_extractor import AudioExtractor
from backend.services.groq_stt import GroqSTTService
from backend.services.topic_intelligence import VidaraTopicIntelligenceEngine
from backend.services.semantic_engine import SemanticEngine
from backend.services.video_cutter import VideoCutter
from backend.models.schemas import (
    IngestUrlRequest,
    DiscoveredTopic,
    TopicRelation,
    UserQueryRequest,
    UserQueryResponse,
    GenerateClipsRequest,
    MergeClipsRequest,
    VideoStatusResponse,
    EnrichedTranscript,
    ClipNotesRequest,
    ClipNotesResponse
)
from backend.services.notes_service import ClipNotesService, VALID_AUDIENCE_MODES

router = APIRouter(tags=["Vidara Video Intelligence"])


# In-memory fast cache for active sessions
VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}


def verify_video_access(video_id: str, user_id: str) -> Dict[str, Any]:
    """
    Strict ownership verification dependency helper.
    Ensures that only the authenticated owner can access this video and its derived assets.
    """
    vid = DatabaseService.get_video(video_id)
    if not vid:
        raise HTTPException(status_code=404, detail="Video session not found.")

    owner = vid.get("user_id")
    if owner:
        from backend.services.supabase_service import to_uuid
        is_owner = (str(owner) == str(user_id)) or (to_uuid(owner) == to_uuid(user_id))
        if not is_owner:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have permission to access this video.")
    else:
        # Legacy/unowned record: claim ownership for authenticated user
        DatabaseService.claim_video_ownership(video_id, user_id)
        vid["user_id"] = user_id

    return vid


def get_session_data(video_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    if user_id:
        verify_video_access(video_id, user_id)

    if video_id in VIDEO_CACHE:
        return VIDEO_CACHE[video_id]
    
    vid = DatabaseService.get_video(video_id)
    if not vid:
        raise HTTPException(status_code=404, detail="Video session not found.")
    
    segs = DatabaseService.get_segments(video_id)
    topics = DatabaseService.get_topics(video_id)
    clips = DatabaseService.get_clips(video_id)
    
    # Reconstruct EnrichedTranscript if segments exist
    transcript_obj = None
    if segs:
        from backend.models.schemas import TranscriptSegment
        t_segs = [
            TranscriptSegment(
                id=s["segment_index"],
                start=s["start_time"],
                end=s["end_time"],
                text=s["text"],
                speaker=s.get("speaker", "SPEAKER_01"),
                words=[]
            ) for s in segs
        ]
        transcript_obj = EnrichedTranscript(
            video_id=video_id,
            filename=vid["filename"],
            duration_seconds=vid["duration"],
            language=vid.get("language", "en"),
            segments=t_segs
        )

    video_path = vid.get("filepath", "")
    if not video_path or not Path(video_path).is_file():
        for ext in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
            cand = UPLOAD_DIR / f"{video_id}{ext}"
            if cand.is_file():
                video_path = str(cand)
                break

    audio_path = vid.get("audio_path", "")
    if not audio_path or not Path(audio_path).is_file():
        cand_audio = AUDIO_DIR / f"{video_id}.mp3"
        if cand_audio.is_file():
            audio_path = str(cand_audio)

    session = {
        "video_id": video_id,
        "filename": vid["filename"],
        "video_path": video_path,
        "audio_path": audio_path,
        "duration": vid["duration"],
        "audience_mode": vid.get("audience_mode", "education"),
        "transcript": transcript_obj,
        "topics": [DiscoveredTopic(**t) for t in topics] if topics else [],
        "clips": clips
    }
    VIDEO_CACHE[video_id] = session
    return session


def _async_sync_to_supabase_storage(
    video_id: str,
    user_id: Optional[str],
    video_path: Optional[Path],
    audio_path: Optional[Path],
    ext: str = ".mp4"
) -> None:
    """Non-blocking background helper to sync media to Supabase storage without delaying UI."""
    if not is_supabase_configured():
        return
    try:
        from backend.services.supabase_service import SupabaseStorageService, to_uuid
        u_prefix = to_uuid(user_id) if user_id else "anonymous"
        v_uuid = to_uuid(video_id)

        # Audio is always compact (16kHz mono, 32kbps) and fast to upload
        if audio_path and Path(audio_path).exists():
            SupabaseStorageService.upload_file(SUPABASE_STORAGE_BUCKET_AUDIO, audio_path, f"{u_prefix}/{v_uuid}/audio.mp3", "audio/mpeg")

        # Video is uploaded if within single-file limit
        if video_path and Path(video_path).exists():
            SupabaseStorageService.upload_file(SUPABASE_STORAGE_BUCKET_VIDEOS, video_path, f"{u_prefix}/{v_uuid}/original{ext}", "video/mp4")
    except Exception as supa_err:
        print(f"Notice: Background Supabase storage sync for {video_id}: {supa_err}")


@router.post("/upload")
@router.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    audience_mode: Optional[str] = Form("education"),
    groq_api_key: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Step 1: Upload video file, extract audio with FFmpeg, and initialize database record.
    Supports large video files (1GB - 2GB+) via streaming file save.
    """
    if groq_api_key and groq_api_key.strip():
        set_groq_api_key(groq_api_key.strip())

    if not get_groq_api_key():
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is missing! Click 'Keys' in the navigation bar to enter your key."
        )

    video_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    saved_video_path = UPLOAD_DIR / f"{video_id}{ext}"

    # Stream file to disk in 1MB chunks to support large files without RAM exhaustion
    bytes_written = 0
    with open(saved_video_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)
            bytes_written += len(chunk)

    try:
        # Extract audio using FFmpeg
        audio_path = AudioExtractor.extract_audio(saved_video_path, f"{video_id}.mp3")
        duration = AudioExtractor.get_video_duration(saved_video_path)

        user_id = current_user["id"] if current_user else None

        if audience_mode and audience_mode.strip():
            clean_mode = audience_mode.strip().lower()
            if clean_mode not in VALID_AUDIENCE_MODES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid audience_mode '{audience_mode}'. Valid modes: {list(VALID_AUDIENCE_MODES)}"
                )
        else:
            clean_mode = "education"

        # Save to database immediately
        DatabaseService.save_video(
            video_id=video_id,
            filename=file.filename,
            filepath=str(saved_video_path),
            audio_path=str(audio_path),
            duration=duration,
            filesize=bytes_written,
            user_id=user_id,
            audience_mode=clean_mode
        )

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="ingestion",
            status="uploaded",
            progress=20,
            stage="Audio Extracted",
            message=f"Uploaded {file.filename} ({round(duration, 1)}s). Ready for intelligence analysis.",
            user_id=user_id
        )

        VIDEO_CACHE[video_id] = {
            "video_id": video_id,
            "filename": file.filename,
            "video_path": str(saved_video_path),
            "audio_path": str(audio_path),
            "duration": duration,
            "audience_mode": clean_mode,
            "transcript": None,
            "topics": [],
            "clips": []
        }

        # Offload cloud storage sync to background task so the frontend is unblocked instantly
        background_tasks.add_task(
            _async_sync_to_supabase_storage,
            video_id=video_id,
            user_id=user_id,
            video_path=saved_video_path,
            audio_path=audio_path,
            ext=ext
        )

        return {
            "status": "success",
            "video_id": video_id,
            "filename": file.filename,
            "duration_seconds": duration,
            "filesize_bytes": bytes_written,
            "audience_mode": clean_mode
        }

    except Exception as e:
        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="ingestion",
            status="failed",
            progress=0,
            stage="Failed",
            message=str(e),
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Upload & extraction failed: {str(e)}")


@router.post("/ingest-url")
@router.post("/videos/ingest-url")
async def ingest_video_url(
    req: IngestUrlRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Step 1 (Alternative): Ingests video from a web link (YouTube, Vimeo, Loom, or direct .mp4),
    downloads safely with yt-dlp/streaming, extracts 16kHz audio with FFmpeg, and initializes database record.
    """
    if req.groq_api_key and req.groq_api_key.strip():
        set_groq_api_key(req.groq_api_key.strip())

    if not get_groq_api_key():
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is missing! Click 'Keys' in the navigation bar to enter your key."
        )

    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid video link.")

    from backend.services.link_downloader import LinkDownloader

    if req.audience_mode and req.audience_mode.strip():
        clean_mode = req.audience_mode.strip().lower()
        if clean_mode not in VALID_AUDIENCE_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audience_mode '{req.audience_mode}'. Valid modes: {list(VALID_AUDIENCE_MODES)}"
            )
    else:
        clean_mode = "education"

    try:
        user_id = current_user["id"]

        res = await asyncio.to_thread(LinkDownloader.download_video_from_url, req.url.strip(), user_id=user_id, audience_mode=clean_mode)
        video_id = res["video_id"]
        filename = res["filename"]
        duration = res["duration_seconds"]
        filesize = res["filesize"]

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="ingestion",
            status="uploaded",
            progress=20,
            stage="Audio Extracted",
            message=f"Imported from link: {filename} ({round(duration, 1)}s). Ready for intelligence analysis.",
            user_id=user_id
        )

        ext = Path(filename).suffix or ".mp4"
        background_tasks.add_task(
            _async_sync_to_supabase_storage,
            video_id=video_id,
            user_id=user_id,
            video_path=Path(res["filepath"]) if res.get("filepath") else None,
            audio_path=Path(res["audio_path"]) if res.get("audio_path") else None,
            ext=ext
        )

        VIDEO_CACHE[video_id] = {
            "video_id": video_id,
            "filename": filename,
            "video_path": res["filepath"],
            "audio_path": res["audio_path"],
            "duration": duration,
            "audience_mode": clean_mode,
            "transcript": None,
            "topics": [],
            "clips": []
        }

        return {
            "status": "success",
            "video_id": video_id,
            "filename": filename,
            "duration_seconds": duration,
            "filesize_bytes": filesize,
            "source_url": req.url.strip(),
            "audience_mode": clean_mode
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest video from link: {str(e)}")


def _run_analysis_pipeline(video_id: str, user_id: Optional[str] = None) -> None:
    """
    Internal background worker: Transcribes audio with Whisper, builds semantic index.
    Runs outside the request context so the HTTP response is not blocked.
    """
    session = VIDEO_CACHE.get(video_id) or {}

    # Reload from DB if not in cache
    if not session:
        try:
            session = get_session_data(video_id, user_id=user_id)
        except Exception:
            return

    raw_audio = session.get("audio_path") or ""
    audio_path = Path(raw_audio) if raw_audio else None
    if not audio_path or not audio_path.is_file():
        candidate_audio = AUDIO_DIR / f"{video_id}.mp3"
        if candidate_audio.is_file():
            audio_path = candidate_audio
            session["audio_path"] = str(candidate_audio)
        else:
            raw_video = session.get("video_path") or ""
            video_path = Path(raw_video) if raw_video else None
            if not video_path or not video_path.is_file():
                for ext in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
                    c_vid = UPLOAD_DIR / f"{video_id}{ext}"
                    if c_vid.is_file():
                        video_path = c_vid
                        session["video_path"] = str(c_vid)
                        break
            if video_path and video_path.is_file():
                audio_path = AudioExtractor.extract_audio(video_path, f"{video_id}.mp3")
                session["audio_path"] = str(audio_path)
            else:
                DatabaseService.set_job(
                    job_id=f"job_{video_id}",
                    video_id=video_id,
                    job_type="indexing",
                    status="failed",
                    progress=0,
                    stage="Failed",
                    message=f"Audio file not found for video {video_id}",
                    error=f"Audio file not found for video {video_id}",
                    user_id=user_id
                )
                return

    DatabaseService.set_job(
        job_id=f"job_{video_id}",
        video_id=video_id,
        job_type="transcription",
        status="processing",
        progress=40,
        stage="Transcribing Audio",
        message="Running Groq Whisper Large-v3 with word timestamps...",
        user_id=user_id
    )

    try:
        stt = GroqSTTService()
        transcript = stt.transcribe(audio_path, video_id, session["duration"])
        session["transcript"] = transcript
        VIDEO_CACHE[video_id] = session

        # Save transcript segments to DB
        segments_data = [
            {
                "segment_index": s.id,
                "start_time": s.start,
                "end_time": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "words": [w.model_dump() for w in s.words] if s.words else []
            }
            for s in transcript.segments
        ]
        DatabaseService.save_segments(video_id, segments_data)

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="indexing",
            status="processing",
            progress=65,
            stage="Building Semantic Graph",
            message=f"Transcribed {len(transcript.segments)} segments. Building Intent Graph...",
            user_id=user_id
        )

        # Build Core Thesis using SemanticEngine
        sem_engine = SemanticEngine()
        intent_graph = sem_engine.build_intent_graph(transcript)
        session["intent_graph"] = intent_graph
        VIDEO_CACHE[video_id] = session
        DatabaseService.update_video_status(video_id, "indexed", core_thesis=intent_graph.core_thesis)

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="indexing",
            status="processing",
            progress=85,
            stage="Discovering Key Moments",
            message="Evaluating podcast exchanges, takeaways & topic boundaries...",
            user_id=user_id
        )

        # Pre-seed and persist topics immediately from Intent Graph themes
        topic_engine = VidaraTopicIntelligenceEngine()
        themes = intent_graph.themes if hasattr(intent_graph, "themes") else []
        discovered_topics = topic_engine.discover_important_topics(transcript, intent_graph.core_thesis, themes=themes)
        session["topics"] = discovered_topics
        VIDEO_CACHE[video_id] = session
        DatabaseService.save_topics(video_id, [t.model_dump() for t in discovered_topics])

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="indexing",
            status="completed",
            progress=100,
            stage="Semantic Index Built",
            message=f"Indexed {len(transcript.segments)} segments & discovered {len(discovered_topics)} topics.",
            result={
                "core_thesis": intent_graph.core_thesis,
                "segment_count": len(transcript.segments),
                "topic_count": len(discovered_topics),
                "topics": [t.model_dump() for t in discovered_topics]
            },
            user_id=user_id
        )

    except Exception as e:
        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="indexing",
            status="failed",
            progress=0,
            stage="Failed",
            message=str(e),
            error=str(e),
            user_id=user_id
        )


@router.post("/videos/{video_id}/analyze")
async def analyze_video(
    video_id: str,
    background_tasks: BackgroundTasks,
    wait: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Step 2: Kicks off background transcription + semantic indexing and returns immediately.
    Poll /videos/{video_id}/status to track progress.
    Once status is 'completed', fetch full results from /videos/{video_id}/analyze-result.
    Supports ?wait=true for synchronous test calls.
    """
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    if not session:
        raise HTTPException(status_code=404, detail="Video session not found.")

    # If already completed (re-run guard), return result directly
    job = DatabaseService.get_job(f"job_{video_id}")

    # Guard 1: job record with result payload AND actual topics
    if job and job.get("status") == "completed" and job.get("job_type") in ("indexing", "analysis") and not wait:
        result = job.get("result") or {}
        topics = result.get("topics") or []
        if topics and len(topics) > 0:
            return {
                "status": "success",
                "video_id": video_id,
                "already_indexed": True,
                **result
            }

    # Guard 2: transcript already in cache/session (in-memory fast path) AND has topics
    if session.get("transcript") and not wait:
        topics = session.get("topics") or [t for t in DatabaseService.get_topics(video_id)]
        if topics and len(topics) > 0:
            thesis = ""
            ig = session.get("intent_graph")
            if ig:
                thesis = ig.core_thesis
            if not thesis:
                vid = DatabaseService.get_video(video_id)
                thesis = (vid or {}).get("core_thesis", "")
            return {
                "status": "success",
                "video_id": video_id,
                "already_indexed": True,
                "core_thesis": thesis,
                "segment_count": len(session["transcript"].segments) if hasattr(session["transcript"], "segments") else 0,
                "topic_count": len(topics),
                "topics": [t.model_dump() if hasattr(t, "model_dump") else t for t in topics]
            }

    # Guard 3: segments already in DB AND topics already in DB
    existing_segs = DatabaseService.get_segments(video_id)
    existing_topics = DatabaseService.get_topics(video_id)
    if existing_segs and len(existing_segs) > 0 and existing_topics and len(existing_topics) > 0 and not wait:
        vid = DatabaseService.get_video(video_id)
        return {
            "status": "success",
            "video_id": video_id,
            "already_indexed": True,
            "core_thesis": (vid or {}).get("core_thesis", ""),
            "segment_count": len(existing_segs),
            "topic_count": len(existing_topics),
            "topics": existing_topics
        }

    # Guard 4: already running — don't double-queue
    if job and job.get("status") in ("queued", "processing") and job.get("job_type") in ("transcription", "indexing", "analysis") and not wait:
        return {
            "status": "queued",
            "video_id": video_id,
            "message": "Analysis already in progress. Poll /api/videos/{video_id}/status for progress."
        }

    if wait:
        _run_analysis_pipeline(video_id, user_id=current_user["id"])
        job = DatabaseService.get_job(f"job_{video_id}")
        if job and job.get("status") == "failed":
            raise HTTPException(status_code=500, detail=job.get("error", "Analysis failed"))
        result = (job or {}).get("result") or {}
        if not result.get("topics"):
            result["topics"] = DatabaseService.get_topics(video_id)
            result["topic_count"] = len(result["topics"])
        if not result.get("core_thesis"):
            vid = DatabaseService.get_video(video_id)
            result["core_thesis"] = (vid or {}).get("core_thesis", "")
        if not result.get("segment_count"):
            result["segment_count"] = len(DatabaseService.get_segments(video_id))
        return {
            "status": "success",
            "video_id": video_id,
            **result
        }

    # Mark as queued
    DatabaseService.set_job(
        job_id=f"job_{video_id}",
        video_id=video_id,
        job_type="transcription",
        status="queued",
        progress=10,
        stage="Queued",
        message="Analysis pipeline queued. Starting Whisper transcription...",
        user_id=current_user["id"]
    )

    # Kick off the heavy work in the background
    background_tasks.add_task(_run_analysis_pipeline, video_id, current_user["id"])

    return {
        "status": "queued",
        "video_id": video_id,
        "message": "Analysis started in background. Poll /api/videos/{video_id}/status for progress."
    }


@router.get("/videos/{video_id}/analyze-result")
async def get_analyze_result(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Returns the full analysis result (topics, core_thesis, segment_count) once the pipeline is done.
    Returns 202 if still processing, 200 with data when complete.
    """
    verify_video_access(video_id, current_user["id"])
    job = DatabaseService.get_job(f"job_{video_id}")
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found. Did you call /analyze first?")

    status = job.get("status", "")
    job_type = job.get("job_type", "")
    stage = job.get("stage") or job.get("current_stage", "")
    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {job.get('message', 'Unknown error')}")

    if status != "completed" or stage in ("Audio Extracted", "Media ingestion"):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "status": status,
                "job_type": job_type,
                "progress": job.get("progress", 0),
                "stage": stage or "Processing",
                "message": job.get("message", "Still processing...")
            }
        )

    # Completed — try to get result from job record first, then rebuild from DB
    result = job.get("result") or {}
    if not result.get("topics"):
        topics_raw = DatabaseService.get_topics(video_id)
        result["topics"] = topics_raw
        result["topic_count"] = len(topics_raw)

    if not result.get("core_thesis"):
        vid = DatabaseService.get_video(video_id)
        result["core_thesis"] = (vid or {}).get("core_thesis", "")

    if not result.get("segment_count"):
        segs = DatabaseService.get_segments(video_id)
        result["segment_count"] = len(segs)

    return {
        "status": "success",
        "video_id": video_id,
        **result
    }


@router.post("/videos/{video_id}/discover-topics")
async def discover_topics(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    MODE B: DISCOVER IMPORTANT TOPICS
    Executes Vidara Topic Intelligence Engine to extract, cluster, graph, score, and rank topics.
    """
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    if not session.get("transcript"):
        # Auto-run analysis inline (blocking) so we have a transcript before proceeding
        _run_analysis_pipeline(video_id, user_id=current_user["id"])
        session = get_session_data(video_id, user_id=current_user["id"])

    transcript = session.get("transcript")
    core_thesis = session.get("intent_graph", None)
    thesis_str = core_thesis.core_thesis if core_thesis else ""
    themes = core_thesis.themes if (core_thesis and hasattr(core_thesis, "themes")) else []

    DatabaseService.set_job(
        job_id=f"job_{video_id}",
        video_id=video_id,
        job_type="discovery",
        status="processing",
        progress=80,
        stage="Discovering Topics",
        message="Computing Topic Graph, PageRank centrality, and multi-feature importance...",
        user_id=current_user["id"]
    )

    try:
        engine = VidaraTopicIntelligenceEngine()
        topics = engine.discover_important_topics(transcript, thesis_str, themes=themes)
        session["topics"] = topics

        # Persist discovered topics
        DatabaseService.save_topics(video_id, [t.model_dump() for t in topics])

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="discovery",
            status="completed",
            progress=100,
            stage="Topics Discovered",
            message=f"Discovered and ranked {len(topics)} meaningful topics.",
            result={"total_topics": len(topics)},
            user_id=current_user["id"]
        )

        return {
            "status": "success",
            "video_id": video_id,
            "total_topics": len(topics),
            "topics": [t.model_dump() for t in topics]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic discovery failed: {str(e)}")


@router.post("/videos/{video_id}/query")
async def query_video(
    video_id: str,
    req: UserQueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    MODE A: ASK VIDARA (Text or Voice)
    Retrieves semantically relevant sections matching the user's query without retranscribing.
    """
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    if not session.get("transcript"):
        _run_analysis_pipeline(video_id, user_id=current_user["id"])
        session = get_session_data(video_id, user_id=current_user["id"])

    engine = VidaraTopicIntelligenceEngine()
    existing_topics = session.get("topics") or []
    if not existing_topics:
        # Pre-seed topics
        existing_topics = engine.discover_important_topics(session["transcript"])
        session["topics"] = existing_topics
        DatabaseService.save_topics(video_id, [t.model_dump() for t in existing_topics])

    query_res = engine.query_video_topics(
        query=req.query,
        transcript=session["transcript"],
        indexed_topics=existing_topics
    )

    found = query_res.get("found", len(query_res.get("matched_topics", [])) > 0)

    return {
        "status": "success" if found else "not_found",
        "found": found,
        "video_id": video_id,
        "query": query_res["query"],
        "understood_concepts": query_res.get("understood_concepts", []),
        "matched_topics": [t.model_dump() for t in query_res.get("matched_topics", [])],
        "reasoning": query_res["reasoning"]
    }


def resolve_input_video_path(video_id: str, session: Dict[str, Any]) -> Optional[Path]:
    """Robustly resolves the physical input video path from session, DB, disk, or Supabase."""
    vp = session.get("video_path")
    if vp and Path(vp).is_file():
        return Path(vp)

    vid_record = DatabaseService.get_video(video_id)
    if vid_record and vid_record.get("filepath") and Path(vid_record["filepath"]).is_file():
        session["video_path"] = vid_record["filepath"]
        return Path(vid_record["filepath"])

    for ext in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
        cand = UPLOAD_DIR / f"{video_id}{ext}"
        if cand.is_file():
            session["video_path"] = str(cand)
            return cand

    matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if matches and matches[0].is_file():
        session["video_path"] = str(matches[0])
        return matches[0]

    if is_supabase_configured():
        try:
            from backend.services.supabase_service import SupabaseStorageService, to_uuid
            v_uuid = to_uuid(video_id)
            fn = session.get("filename") or (vid_record.get("filename") if vid_record else "video.mp4")
            supa_path = f"{v_uuid}/source/{fn}"
            dest = UPLOAD_DIR / f"{video_id}.mp4"
            down = SupabaseStorageService.download_file(SUPABASE_STORAGE_BUCKET_VIDEOS, supa_path, dest)
            if down and down.is_file():
                session["video_path"] = str(down)
                return down
        except Exception as e:
            print(f"Warning: Failed to fetch source video from Supabase: {e}")

    return None


@router.post("/videos/{video_id}/generate-clips")
async def generate_clips(
    video_id: str,
    req: GenerateClipsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Step 8: Deterministically cuts verified video clips for specified topics.
    Preserves +80ms lead-in, +160ms tail-out, PTS normalization, and faststart.
    """
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    all_topics = session.get("topics") or [DiscoveredTopic(**t) for t in DatabaseService.get_topics(video_id)]

    if req.topic_ids:
        def norm_id(i):
            return str(i).replace(f"{video_id}_", "")

        req_set = {norm_id(i) for i in req.topic_ids}
        selected_topics = [
            t for t in all_topics
            if norm_id(t.id if hasattr(t, "id") else t.get("id", "")) in req_set
        ]
    else:
        selected_topics = [
            t for t in all_topics
            if (t.is_selected if hasattr(t, "is_selected") else t.get("is_selected", True))
        ]

    if not selected_topics:
        selected_topics = all_topics[:3]

    video_path = resolve_input_video_path(video_id, session)
    if not video_path or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Original video file not found for cutting clips.")

    transcript = session.get("transcript")
    segments_data = None
    if transcript and hasattr(transcript, "segments"):
        segments_data = [s.model_dump() for s in transcript.segments]
    else:
        segments_data = DatabaseService.get_segments(video_id)

    rendered_clips = VideoCutter.render_topic_clips(
        input_video_path=video_path,
        video_id=video_id,
        topics=[t.model_dump() if hasattr(t, "model_dump") else t for t in selected_topics],
        segments=segments_data
    )

    # Persist clips
    DatabaseService.save_clips(video_id, rendered_clips)
    session["clips"] = rendered_clips

    return {
        "status": "success",
        "video_id": video_id,
        "clip_count": len(rendered_clips),
        "clips": rendered_clips
    }


@router.get("/videos/{video_id}/stream")
async def stream_full_video(
    video_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream full original video with HTTP 206 Partial Content Range support for scene preview."""
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    video_path = resolve_input_video_path(video_id, session)

    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Original video file not found.")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(
            path=str(video_path),
            filename=video_path.name,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="{video_path.name}"'
            }
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
    if start >= file_size or end >= file_size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    content_length = end - start + 1

    def file_iterator():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            chunk_size = 1024 * 1024
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = f.read(to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
        "Content-Disposition": f'inline; filename="{video_path.name}"'
    }
    return StreamingResponse(file_iterator(), status_code=206, headers=headers)


@router.post("/videos/{video_id}/merge")
async def merge_clips(
    video_id: str,
    req: MergeClipsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Step 9: Merges only the user's selected clips into a final unified master video.
    Preserves +80ms lead-in, +160ms tail-out, PTS normalization, and faststart.
    """
    verify_video_access(video_id, current_user["id"])
    session = get_session_data(video_id, user_id=current_user["id"])
    all_clips = session.get("clips") or DatabaseService.get_clips(video_id) or []
    all_topics = session.get("topics") or [DiscoveredTopic(**t) for t in DatabaseService.get_topics(video_id)]

    # 1. Determine requested topic keys with bidirectional normalization
    raw_requested_ids = list(req.topic_ids or [])
    if not raw_requested_ids and req.clip_ids:
        raw_requested_ids = list(req.clip_ids)

    selected_topic_keys = set()
    for item in raw_requested_ids:
        s = str(item).strip()
        selected_topic_keys.add(s)
        clean = s.replace(f"{video_id}_", "").replace(".mp4", "")
        selected_topic_keys.add(clean)
        selected_topic_keys.add(f"{video_id}_{clean}")

    # If no specific topics provided, default to all discovered topics
    if not selected_topic_keys:
        for t in all_topics:
            tid = str(t.id if hasattr(t, "id") else t["id"])
            selected_topic_keys.add(tid)
            clean = tid.replace(f"{video_id}_", "").replace(".mp4", "")
            selected_topic_keys.add(clean)
            selected_topic_keys.add(f"{video_id}_{clean}")

    # 2. Build comprehensive lookup of already-rendered clips on disk
    existing_clip_map: Dict[str, Dict[str, Any]] = {}
    resolved_clips_list = []

    for c in all_clips:
        c_dict = dict(c) if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else dict(c))
        fp = Path(c_dict.get("filepath", ""))
        cid = str(c_dict.get("id", ""))
        ctid = str(c_dict.get("topic_id", ""))
        clean_tid = ctid.replace(f"{video_id}_", "").replace(".mp4", "") or cid.replace(f"{video_id}_", "").replace(".mp4", "")

        if not fp.is_file():
            candidates = [
                OUTPUT_DIR / (c_dict.get("filename") or f"{cid}.mp4"),
                OUTPUT_DIR / f"{video_id}_{clean_tid}.mp4",
                OUTPUT_DIR / f"{clean_tid}.mp4"
            ]
            found = next((p for p in candidates if p.is_file()), None)
            if not found:
                glob_matches = list(OUTPUT_DIR.glob(f"*{clean_tid}*.mp4"))
                if glob_matches and glob_matches[0].is_file():
                    found = glob_matches[0]
            if found:
                c_dict["filepath"] = str(found)
                fp = found

        if fp.is_file():
            for k in [cid, clean_tid, f"{video_id}_{clean_tid}", ctid]:
                if k:
                    existing_clip_map[k] = c_dict
            resolved_clips_list.append(c_dict)

    # 3. Check which requested topics are missing and need on-the-fly rendering
    missing_topics = []
    for t in all_topics:
        tid = str(t.id if hasattr(t, "id") else t["id"])
        clean_tid = tid.replace(f"{video_id}_", "").replace(".mp4", "")
        if (tid in selected_topic_keys or clean_tid in selected_topic_keys):
            if tid not in existing_clip_map and clean_tid not in existing_clip_map and f"{video_id}_{clean_tid}" not in existing_clip_map:
                missing_topics.append(t)

    # 4. If any clips are missing from disk, attempt to render them on the fly!
    if missing_topics:
        video_path = resolve_input_video_path(video_id, session)
        if video_path and video_path.is_file():
            newly_rendered = VideoCutter.render_topic_clips(
                input_video_path=video_path,
                video_id=video_id,
                topics=[t.model_dump() if hasattr(t, "model_dump") else t for t in missing_topics]
            )
            for c in newly_rendered:
                cid = str(c.get("id", ""))
                ctid = str(c.get("topic_id", ""))
                clean_tid = ctid.replace(f"{video_id}_", "").replace(".mp4", "")
                for k in [cid, clean_tid, f"{video_id}_{clean_tid}", ctid]:
                    if k:
                        existing_clip_map[k] = c
                resolved_clips_list.append(c)

            all_clips = list({c["id"]: c for c in resolved_clips_list}.values())
            DatabaseService.save_clips(video_id, all_clips)
            session["clips"] = all_clips
        else:
            print(f"Notice: Source video file not found to cut {len(missing_topics)} missing topic clips; proceeding with available clips.")

    # 5. Collect target clips based on selected_topic_keys
    target_clips = []
    seen_target_ids = set()
    for c in resolved_clips_list:
        cid = str(c.get("id", ""))
        ctid = str(c.get("topic_id", ""))
        clean_cid = cid.replace(f"{video_id}_", "").replace(".mp4", "")
        clean_ctid = ctid.replace(f"{video_id}_", "").replace(".mp4", "")
        if (
            cid in selected_topic_keys or
            ctid in selected_topic_keys or
            clean_cid in selected_topic_keys or
            clean_ctid in selected_topic_keys
        ):
            if cid not in seen_target_ids:
                seen_target_ids.add(cid)
                target_clips.append(c)

    if not target_clips:
        target_clips = list({c["id"]: c for c in resolved_clips_list}.values())

    target_clips.sort(key=lambda c: float(c.get("start_time", 0.0)))
    clip_paths = []
    for c in target_clips:
        p = Path(c.get("filepath", ""))
        if p.is_file() and p.stat().st_size > 1000:
            clip_paths.append(p)
        else:
            alt = OUTPUT_DIR / (c.get("filename") or f"{c.get('id', '')}.mp4")
            if alt.is_file() and alt.stat().st_size > 1000:
                clip_paths.append(alt)

    if not clip_paths:
        raise HTTPException(
            status_code=400,
            detail="Could not find or render clips on disk. Please generate clips first or ensure the video is indexed."
        )

    # 6. Execute deterministic FFmpeg merge
    merged_filename = f"{video_id}_selected_master.mp4"
    output_path = OUTPUT_DIR / merged_filename

    final_merged = VideoCutter.merge_clip_files(clip_paths, output_path)

    export_storage_path = None
    if is_supabase_configured():
        try:
            from backend.services.supabase_service import SupabaseStorageService, to_uuid
            u_prefix = to_uuid(current_user["id"])
            v_uuid = to_uuid(video_id)
            export_storage_path = f"{u_prefix}/{v_uuid}/exports/{merged_filename}"
            SupabaseStorageService.upload_file(SUPABASE_STORAGE_BUCKET_EXPORTS, output_path, export_storage_path, "video/mp4")
        except Exception as supa_err:
            print(f"Warning: Supabase upload on merge_clips failed: {supa_err}")

    total_merged_dur = sum(float(c.get("duration", 0.0)) for c in target_clips)

    return {
        "status": "success",
        "video_id": video_id,
        "merged_filename": merged_filename,
        "storage_path": export_storage_path,
        "video_url": f"/api/videos/{video_id}/final",
        "download_url": f"/api/videos/{video_id}/final",
        "clip_count": len(clip_paths),
        "total_duration": round(total_merged_dur, 2)
    }


@router.get("/videos/{video_id}/topics")
async def get_topics(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves ranked topics for the video."""
    verify_video_access(video_id, current_user["id"])
    topics = DatabaseService.get_topics(video_id)
    return {"video_id": video_id, "topics": topics}


@router.get("/videos/{video_id}/clips")
async def get_clips(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves generated clips for the video."""
    verify_video_access(video_id, current_user["id"])
    clips = DatabaseService.get_clips(video_id)
    return {"video_id": video_id, "clips": clips}


@router.get("/videos/{video_id}/status")
@router.get("/videos/{video_id}/job-status")
async def get_job_status(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Retrieves full background processing job status and metrics.
    Exposes real stage, percentage, chunks, elapsed time, retries, and results.
    """
    verify_video_access(video_id, current_user["id"])
    job = DatabaseService.get_job(f"job_{video_id}")
    vid = DatabaseService.get_video(video_id)

    if not job:
        if vid:
            topics = DatabaseService.get_topics(video_id)
            segs = DatabaseService.get_segments(video_id)
            return {
                "video_id": video_id,
                "job_id": f"job_{video_id}",
                "status": "completed" if vid["status"] == "indexed" else vid["status"],
                "progress": 100 if vid["status"] == "indexed" else 20,
                "stage": "Ready" if vid["status"] == "indexed" else "Media ingestion",
                "message": "Video indexed and ready.",
                "chunks_completed": 0,
                "chunks_total": 0,
                "elapsed_seconds": 0.0,
                "retries": 0,
                "error": None,
                "core_thesis": vid.get("core_thesis", ""),
                "audience_mode": vid.get("audience_mode", "education") if vid else "education",
                "segment_count": len(segs),
                "topic_count": len(topics),
                "topics": topics
            }
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    topics = []
    segs = []
    core_thesis = ""
    if job.get("status") == "completed" or (vid and vid.get("status") == "indexed"):
        topics = DatabaseService.get_topics(video_id)
        segs = DatabaseService.get_segments(video_id)
        core_thesis = vid.get("core_thesis", "") if vid else ""

    return {
        "video_id": video_id,
        "job_id": job.get("id", f"job_{video_id}"),
        "job_type": job.get("job_type", "indexing"),
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage") or job.get("current_stage", "Processing"),
        "message": job.get("message", ""),
        "chunks_completed": job.get("chunks_completed", 0),
        "chunks_total": job.get("chunks_total", 0),
        "elapsed_seconds": job.get("elapsed_seconds", 0.0),
        "retries": job.get("retries", 0),
        "error": job.get("error"),
        "core_thesis": core_thesis,
        "audience_mode": vid.get("audience_mode", "education") if vid else "education",
        "segment_count": len(segs),
        "topic_count": len(topics),
        "topics": topics,
        "performance_stats": job.get("result", {})
    }


@router.get("/videos/{video_id}/progress/stream")
async def stream_progress(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Server-Sent Events (SSE) stream yielding real-time job progress events."""
    verify_video_access(video_id, current_user["id"])
    import asyncio
    from fastapi.responses import StreamingResponse

    async def event_generator():
        while True:
            job_info = await get_job_status(video_id, current_user=current_user)
            yield f"data: {json.dumps(job_info)}\n\n"
            if job_info.get("status") in ("completed", "failed"):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/videos/{video_id}/clip/{topic_id}")
async def stream_clip(
    video_id: str,
    topic_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream or download an individual topic clip with HTTP 206 Range support."""
    verify_video_access(video_id, current_user["id"])
    clean_tid = topic_id.replace(f"{video_id}_", "")
    candidates = [
        OUTPUT_DIR / f"{video_id}_{clean_tid}.mp4",
        OUTPUT_DIR / f"{video_id}_{topic_id}.mp4",
        OUTPUT_DIR / f"{video_id}_{video_id}_{clean_tid}.mp4",
        OUTPUT_DIR / f"{clean_tid}.mp4",
        OUTPUT_DIR / f"{topic_id}.mp4",
    ]
    clip_path = next((p for p in candidates if p.exists()), None)
    if not clip_path:
        matches = list(OUTPUT_DIR.glob(f"*{clean_tid}*.mp4"))
        if matches:
            clip_path = matches[0]

    if not clip_path or not clip_path.exists():
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseStorageService, to_uuid
                v_uuid = to_uuid(video_id)
                storage_path = f"{v_uuid}/clips/{video_id}_{clean_tid}.mp4"
                # Try downloading locally first for streaming
                download_dest = OUTPUT_DIR / f"{video_id}_{clean_tid}.mp4"
                downloaded = SupabaseStorageService.download_file(SUPABASE_STORAGE_BUCKET_CLIPS, storage_path, download_dest)
                if downloaded and downloaded.exists():
                    clip_path = downloaded
                else:
                    signed_url = SupabaseStorageService.create_signed_url(SUPABASE_STORAGE_BUCKET_CLIPS, storage_path)
                    if signed_url:
                        return RedirectResponse(url=signed_url)
            except Exception as supa_err:
                print(f"Warning: Supabase clip retrieval failed: {supa_err}")

    if not clip_path or not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found.")

    return FileResponse(
        path=str(clip_path),
        filename=clip_path.name,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{clip_path.name}"'}
    )


@router.get("/videos/{video_id}/clip/{topic_id}/subtitles")
async def get_clip_subtitles(
    video_id: str,
    topic_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream or download synchronized WebVTT subtitles with speaker tags for an individual clip."""
    verify_video_access(video_id, current_user["id"])
    clean_tid = topic_id.replace(f"{video_id}_", "")
    candidates = [
        OUTPUT_DIR / f"{video_id}_{clean_tid}.vtt",
        OUTPUT_DIR / f"{clean_tid}.vtt",
        OUTPUT_DIR / f"{video_id}_{topic_id}.vtt",
    ]
    vtt_path = next((p for p in candidates if p.exists()), None)
    if not vtt_path and is_supabase_configured():
        try:
            from backend.services.supabase_service import SupabaseStorageService, to_uuid
            v_uuid = to_uuid(video_id)
            sub_storage_path = f"{v_uuid}/subtitles/{clean_tid}.vtt"
            download_dest = OUTPUT_DIR / f"{video_id}_{clean_tid}.vtt"
            downloaded = SupabaseStorageService.download_file(SUPABASE_STORAGE_BUCKET_SUBTITLES, sub_storage_path, download_dest)
            if downloaded and downloaded.exists():
                vtt_path = downloaded
            else:
                signed_url = SupabaseStorageService.create_signed_url(SUPABASE_STORAGE_BUCKET_SUBTITLES, sub_storage_path)
                if signed_url:
                    return RedirectResponse(url=signed_url)
        except Exception as supa_err:
            print(f"Warning: Supabase subtitle retrieval failed: {supa_err}")

    if not vtt_path or not vtt_path.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found for this clip.")

    return FileResponse(
        path=str(vtt_path),
        filename=vtt_path.name,
        media_type="text/vtt",
        headers={"Content-Disposition": f'inline; filename="{vtt_path.name}"'}
    )


@router.get("/videos/{video_id}/final")
async def stream_final(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stream or download the final merged video."""
    verify_video_access(video_id, current_user["id"])
    merged_filename = f"{video_id}_selected_master.mp4"
    output_path = OUTPUT_DIR / merged_filename
    if not output_path.exists():
        # Fallback to any final mp4 for this video
        matches = list(OUTPUT_DIR.glob(f"{video_id}*final*.mp4"))
        if matches:
            output_path = matches[0]
            merged_filename = output_path.name
        else:
            raise HTTPException(status_code=404, detail="Merged video not found.")

    return FileResponse(
        path=str(output_path),
        filename=merged_filename,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{merged_filename}"'}
    )


@router.get("/videos/{video_id}/transcript")
async def get_transcript(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves full indexed transcript segments for the video."""
    verify_video_access(video_id, current_user["id"])
    segments = DatabaseService.get_segments(video_id)
    return {"video_id": video_id, "segment_count": len(segments), "segments": segments}


@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Deletes a video and all associated segments, topics, clips, and storage files."""
    verify_video_access(video_id, current_user["id"])
    DatabaseService.delete_video(video_id=video_id, user_id=current_user["id"])
    if video_id in VIDEO_CACHE:
        del VIDEO_CACHE[video_id]
    return {"status": "success", "message": "Video successfully deleted."}


@router.get("/videos")
async def list_videos(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves all videos uploaded or owned by the authenticated user."""
    videos = DatabaseService.get_user_videos(current_user["id"])
    return {"status": "success", "count": len(videos), "videos": videos}


class ApiKeysRequest(BaseModel):
    groq_api_key: Optional[str] = None
    groq_stt_key: Optional[str] = None
    groq_llm_key: Optional[str] = None


@router.get("/settings/keys")
async def get_keys_status():
    """Returns current key configuration status without revealing secrets."""
    master = get_groq_api_key()
    stt = get_groq_stt_api_key()
    llm = get_groq_llm_api_key()

    def mask(k: str) -> str:
        if not k:
            return ""
        return k[:4] + "..." + k[-4:] if len(k) > 10 else "***"

    return {
        "has_master_key": bool(master),
        "has_stt_key": bool(stt),
        "has_llm_key": bool(llm),
        "masked_master": mask(master),
        "masked_stt": mask(stt),
        "masked_llm": mask(llm)
    }


@router.post("/settings/keys")
async def update_keys(req: ApiKeysRequest):
    """Save or update Groq API keys (dual or master) at runtime and persist to .env."""
    set_groq_api_key(
        key=req.groq_api_key.strip() if req.groq_api_key else "",
        stt_key=req.groq_stt_key.strip() if req.groq_stt_key else "",
        llm_key=req.groq_llm_key.strip() if req.groq_llm_key else ""
    )
    return {
        "status": "success",
        "message": "Groq credentials successfully updated and active for this session."
    }


class ClipSelectionRequest(BaseModel):
    clip_id: str
    position: Optional[int] = 0
    selected: Optional[bool] = True


@router.post("/videos/{video_id}/select-clip")
async def save_clip_selection(
    video_id: str,
    req: ClipSelectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Persists user clip selection and playlist ordering in Supabase clip_selections."""
    verify_video_access(video_id, current_user["id"])
    res = DatabaseService.save_clip_selection(
        user_id=current_user["id"],
        video_id=video_id,
        clip_id=req.clip_id,
        position=req.position or 0,
        selected=req.selected if req.selected is not None else True
    )
    return {"status": "success", "selection": res}


@router.get("/videos/{video_id}/selections")
async def get_clip_selections(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves user clip selections and playlist ordering."""
    verify_video_access(video_id, current_user["id"])
    selections = DatabaseService.get_clip_selections(user_id=current_user["id"], video_id=video_id)
    return {"video_id": video_id, "selections": selections}


# ====================================================
# AUDIENCE-AWARE AI NOTES ENDPOINTS
# ====================================================

@router.post("/clips/{clip_id}/notes", response_model=ClipNotesResponse)
@router.post("/videos/{video_id}/clips/{clip_id}/notes", response_model=ClipNotesResponse)
async def generate_clip_notes_endpoint(
    clip_id: str,
    req: Optional[ClipNotesRequest] = None,
    video_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate structured Audience-Aware AI Notes for a specific clip.
    Extracts timestamp-aligned transcript segments and returns validated pedagogical,
    technical, or storytelling notes, persisted in DB cache.
    """
    req_mode = (req.audience_mode if req else "education") or "education"
    clean_mode = req_mode.strip().lower()
    if clean_mode not in VALID_AUDIENCE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audience_mode '{req_mode}'. Must be one of: {', '.join(sorted(VALID_AUDIENCE_MODES))}"
        )

    from backend.database import get_db_connection
    conn = get_db_connection()
    row = None

    # Try lookup in clips table
    if video_id:
        row = conn.execute(
            "SELECT * FROM clips WHERE video_id = ? AND (id = ? OR topic_id = ?)",
            (video_id, clip_id, clip_id)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM topics WHERE video_id = ? AND id = ?",
                (video_id, clip_id)
            ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM clips WHERE id = ? OR topic_id = ?",
            (clip_id, clip_id)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM topics WHERE id = ?",
                (clip_id,)
            ).fetchone()

    if not row:
        clean_search = clip_id.replace(".mp4", "")
        row = conn.execute(
            "SELECT * FROM clips WHERE id LIKE ? OR topic_id LIKE ? LIMIT 1",
            (f"%{clean_search}%", f"%{clean_search}%")
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM topics WHERE id LIKE ? LIMIT 1",
                (f"%{clean_search}%",)
            ).fetchone()

    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Clip or topic '{clip_id}' not found.")

    row_dict = dict(row)
    resolved_vid = row_dict.get("video_id") or video_id
    if not resolved_vid:
        raise HTTPException(status_code=404, detail="Associated video not found for this clip.")

    # Strict authorization & ownership check
    verify_video_access(resolved_vid, current_user["id"])

    start_time = float(row_dict.get("start_time", 0.0))
    end_time = float(row_dict.get("end_time", 0.0))
    title = row_dict.get("name") or row_dict.get("text") or "Clip Insight"

    try:
        notes = await asyncio.to_thread(
            ClipNotesService.generate_clip_notes,
            clip_id=clip_id,
            video_id=resolved_vid,
            audience_mode=clean_mode,
            clip_title=title,
            start_time=start_time,
            end_time=end_time,
            user_id=current_user["id"]
        )
        return notes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate clip notes: {str(e)}")


@router.get("/clips/{clip_id}/notes", response_model=ClipNotesResponse)
@router.get("/videos/{video_id}/clips/{clip_id}/notes", response_model=ClipNotesResponse)
async def get_clip_notes_endpoint(
    clip_id: str,
    audience_mode: Optional[str] = "education",
    video_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves cached AI notes for a clip, or generates them on demand."""
    clean_mode = (audience_mode or "education").strip().lower()
    if clean_mode not in VALID_AUDIENCE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audience_mode '{audience_mode}'. Must be one of: {', '.join(sorted(VALID_AUDIENCE_MODES))}"
        )

    cached = DatabaseService.get_clip_notes(clip_id, clean_mode)
    if cached:
        cached_vid = cached.get("video_id") or video_id
        if cached_vid:
            verify_video_access(cached_vid, current_user["id"])
        return cached

    req = ClipNotesRequest(audience_mode=clean_mode)
    return await generate_clip_notes_endpoint(
        clip_id=clip_id,
        req=req,
        video_id=video_id,
        current_user=current_user
    )


