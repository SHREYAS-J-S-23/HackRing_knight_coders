import uuid
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.auth_service import get_optional_current_user

from backend.config import (
    UPLOAD_DIR,
    AUDIO_DIR,
    OUTPUT_DIR,
    get_groq_api_key,
    get_groq_stt_api_key,
    get_groq_llm_api_key,
    set_groq_api_key
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
    EnrichedTranscript
)

router = APIRouter(tags=["Vidara Video Intelligence"])

# In-memory fast cache for active sessions
VIDEO_CACHE: Dict[str, Dict[str, Any]] = {}

def get_session_data(video_id: str) -> Dict[str, Any]:
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

    session = {
        "video_id": video_id,
        "filename": vid["filename"],
        "video_path": vid["filepath"],
        "audio_path": vid["audio_path"],
        "duration": vid["duration"],
        "transcript": transcript_obj,
        "topics": [DiscoveredTopic(**t) for t in topics] if topics else [],
        "clips": clips
    }
    VIDEO_CACHE[video_id] = session
    return session


@router.post("/upload")
@router.post("/videos/upload")
async def upload_video(
    file: UploadFile = File(...), 
    groq_api_key: Optional[str] = Form(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user)
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

        # Save to database
        DatabaseService.save_video(
            video_id=video_id,
            filename=file.filename,
            filepath=str(saved_video_path),
            audio_path=str(audio_path),
            duration=duration,
            filesize=bytes_written,
            user_id=user_id
        )

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="ingestion",
            status="completed",
            progress=20,
            stage="Audio Extracted",
            message=f"Uploaded {file.filename} ({round(duration, 1)}s). Ready for intelligence analysis."
        )

        VIDEO_CACHE[video_id] = {
            "video_id": video_id,
            "filename": file.filename,
            "video_path": str(saved_video_path),
            "audio_path": str(audio_path),
            "duration": duration,
            "transcript": None,
            "topics": [],
            "clips": []
        }

        return {
            "status": "success",
            "video_id": video_id,
            "filename": file.filename,
            "duration_seconds": duration,
            "filesize_bytes": bytes_written
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
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user)
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

    try:
        user_id = current_user["id"] if current_user else None
        res = LinkDownloader.download_video_from_url(req.url.strip(), user_id=user_id)
        video_id = res["video_id"]
        filename = res["filename"]
        duration = res["duration_seconds"]
        filesize = res["filesize"]

        DatabaseService.set_job(
            job_id=f"job_{video_id}",
            video_id=video_id,
            job_type="ingestion",
            status="completed",
            progress=20,
            stage="Audio Extracted",
            message=f"Imported from link: {filename} ({round(duration, 1)}s). Ready for intelligence analysis."
        )

        VIDEO_CACHE[video_id] = {
            "video_id": video_id,
            "filename": filename,
            "video_path": res["filepath"],
            "audio_path": res["audio_path"],
            "duration": duration,
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
            "source_url": req.url.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest video from link: {str(e)}")


def _run_analysis_pipeline(video_id: str) -> None:
    """
    Internal background worker: Transcribes audio with Whisper, builds semantic index.
    Runs outside the request context so the HTTP response is not blocked.
    """
    import copy
    session = VIDEO_CACHE.get(video_id) or {}

    # Reload from DB if not in cache
    if not session:
        try:
            session = get_session_data(video_id)
        except Exception:
            return

    audio_path = Path(session["audio_path"])

    DatabaseService.set_job(
        job_id=f"job_{video_id}",
        video_id=video_id,
        job_type="transcription",
        status="processing",
        progress=40,
        stage="Transcribing Audio",
        message="Running Groq Whisper Large-v3 with word timestamps..."
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
            message=f"Transcribed {len(transcript.segments)} segments. Building Intent Graph..."
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
            message="Evaluating podcast exchanges, takeaways & topic boundaries..."
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
            }
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
            error=str(e)
        )


@router.post("/videos/{video_id}/analyze")
async def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    """
    Step 2: Kicks off background transcription + semantic indexing and returns immediately.
    Poll /videos/{video_id}/status to track progress.
    Once status is 'completed', fetch full results from /videos/{video_id}/analyze-result.
    """
    session = get_session_data(video_id)

    # If already completed (re-run guard), return result directly
    job = DatabaseService.get_job(f"job_{video_id}")

    # Guard 1: job record with result payload (from new runs that store result)
    if job and job.get("status") == "completed" and job.get("result"):
        result = job["result"]
        return {
            "status": "success",
            "video_id": video_id,
            "already_indexed": True,
            **result
        }

    # Guard 2: transcript already in cache/session (in-memory fast path)
    if session.get("transcript"):
        topics = session.get("topics") or [t for t in DatabaseService.get_topics(video_id)]
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

    # Guard 3: segments already in DB (e.g., server restarted but DB is persistent)
    existing_segs = DatabaseService.get_segments(video_id)
    if existing_segs and len(existing_segs) > 0:
        existing_topics = DatabaseService.get_topics(video_id)
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
    if job and job.get("status") in ("queued", "processing"):
        return {
            "status": "queued",
            "video_id": video_id,
            "message": "Analysis already in progress. Poll /api/videos/{video_id}/status for progress."
        }

    # Mark as queued
    DatabaseService.set_job(
        job_id=f"job_{video_id}",
        video_id=video_id,
        job_type="transcription",
        status="queued",
        progress=10,
        stage="Queued",
        message="Analysis pipeline queued. Starting Whisper transcription..."
    )

    # Kick off the heavy work in the background
    background_tasks.add_task(_run_analysis_pipeline, video_id)

    return {
        "status": "queued",
        "video_id": video_id,
        "message": "Analysis started in background. Poll /api/videos/{video_id}/status for progress."
    }


@router.get("/videos/{video_id}/analyze-result")
async def get_analyze_result(video_id: str):
    """
    Returns the full analysis result (topics, core_thesis, segment_count) once the pipeline is done.
    Returns 202 if still processing, 200 with data when complete.
    """
    job = DatabaseService.get_job(f"job_{video_id}")
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found. Did you call /analyze first?")

    status = job.get("status", "")
    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Analysis failed: {job.get('message', 'Unknown error')}")

    if status != "completed":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "status": status,
                "progress": job.get("progress", 0),
                "stage": job.get("current_stage", "Processing"),
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
async def discover_topics(video_id: str):
    """
    MODE B: DISCOVER IMPORTANT TOPICS
    Executes Vidara Topic Intelligence Engine to extract, cluster, graph, score, and rank topics.
    """
    session = get_session_data(video_id)
    if not session.get("transcript"):
        # Auto-run analysis inline (blocking) so we have a transcript before proceeding
        _run_analysis_pipeline(video_id)
        session = get_session_data(video_id)

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
        message="Computing Topic Graph, PageRank centrality, and multi-feature importance..."
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
            result={"total_topics": len(topics)}
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
async def query_video(video_id: str, req: UserQueryRequest):
    """
    MODE A: ASK VIDARA (Text or Voice)
    Retrieves semantically relevant sections matching the user's query without retranscribing.
    """
    session = get_session_data(video_id)
    if not session.get("transcript"):
        # Run analysis inline (blocking) so we have a transcript before querying
        _run_analysis_pipeline(video_id)
        session = get_session_data(video_id)

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


@router.post("/videos/{video_id}/generate-clips")
async def generate_clips(video_id: str, req: GenerateClipsRequest):
    """
    Step 7 & 8: Generates individual video clips for selected topics with padding and faststart.
    """
    session = get_session_data(video_id)
    all_topics = session.get("topics") or []
    if not all_topics:
        raise HTTPException(status_code=400, detail="No topics have been discovered yet.")

    # Filter to selected topic IDs (handling prefix differences safely)
    if req.topic_ids:
        def norm_id(i: Any) -> str:
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

    video_path = Path(session["video_path"])
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
async def stream_full_video(video_id: str):
    """Stream full original video with HTTP 206 Partial Content Range support for scene preview."""
    session = get_session_data(video_id)
    video_path = Path(session["video_path"]) if session.get("video_path") else None
    if not video_path or not video_path.exists():
        video_path = UPLOAD_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if matches:
            video_path = matches[0]

    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Original video file not found.")

    return FileResponse(
        path=str(video_path),
        filename=video_path.name,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{video_path.name}"'}
    )


@router.post("/videos/{video_id}/merge")
async def merge_clips(video_id: str, req: MergeClipsRequest):
    """
    Step 9: Merges only the user's selected clips into a final unified master video.
    Preserves +80ms lead-in, +160ms tail-out, PTS normalization, and faststart.
    """
    session = get_session_data(video_id)
    all_clips = session.get("clips") or DatabaseService.get_clips(video_id)
    all_topics = session.get("topics") or [DiscoveredTopic(**t) for t in DatabaseService.get_topics(video_id)]

    # Determine which topics are requested
    selected_topic_ids = set(req.topic_ids or [])
    if not selected_topic_ids and req.clip_ids:
        selected_topic_ids = {c.replace(f"{video_id}_", "").replace(".mp4", "") for c in req.clip_ids}

    # If no specific topics provided, default to all discovered topics
    if not selected_topic_ids:
        selected_topic_ids = {t.id if hasattr(t, "id") else t["id"] for t in all_topics}

    # Check which clips already exist on disk
    existing_clip_map = {c.get("topic_id"): c for c in all_clips if Path(c.get("filepath", "")).exists()}
    missing_topics = [
        t for t in all_topics 
        if (t.id if hasattr(t, "id") else t["id"]) in selected_topic_ids 
        and (t.id if hasattr(t, "id") else t["id"]) not in existing_clip_map
    ]

    # If any clips are missing from disk, render them on the fly!
    if missing_topics:
        video_path = Path(session["video_path"])
        newly_rendered = VideoCutter.render_topic_clips(
            input_video_path=video_path,
            video_id=video_id,
            topics=[t.model_dump() if hasattr(t, "model_dump") else t for t in missing_topics]
        )
        for c in newly_rendered:
            existing_clip_map[c["topic_id"]] = c
        all_clips = list(existing_clip_map.values())
        DatabaseService.save_clips(video_id, all_clips)
        session["clips"] = all_clips

    # Collect and sort clips chronologically for concatenation
    target_clips = [c for c in all_clips if c.get("topic_id") in selected_topic_ids or c.get("id") in selected_topic_ids]
    if not target_clips:
        target_clips = all_clips

    target_clips.sort(key=lambda c: c.get("start_time", 0.0))
    clip_paths = [Path(c["filepath"]) for c in target_clips if Path(c.get("filepath", "")).exists()]
    if not clip_paths:
        raise HTTPException(status_code=400, detail="Could not find or render clips on disk.")

    merged_filename = f"{video_id}_selected_master.mp4"
    output_path = OUTPUT_DIR / merged_filename

    final_merged = VideoCutter.merge_clip_files(clip_paths, output_path)

    total_merged_dur = sum(c.get("duration", 0.0) for c in target_clips)

    return {
        "status": "success",
        "video_id": video_id,
        "merged_filename": merged_filename,
        "video_url": f"/api/videos/{video_id}/final",
        "download_url": f"/api/videos/{video_id}/final",
        "clip_count": len(clip_paths),
        "total_duration": round(total_merged_dur, 2)
    }


@router.get("/videos/{video_id}/topics")
async def get_topics(video_id: str):
    """Retrieves ranked topics for the video."""
    topics = DatabaseService.get_topics(video_id)
    return {"video_id": video_id, "topics": topics}


@router.get("/videos/{video_id}/clips")
async def get_clips(video_id: str):
    """Retrieves generated clips for the video."""
    clips = DatabaseService.get_clips(video_id)
    return {"video_id": video_id, "clips": clips}


@router.get("/videos/{video_id}/status")
async def get_status(video_id: str):
    """Retrieves current processing job status."""
    job = DatabaseService.get_job(f"job_{video_id}")
    if not job:
        vid = DatabaseService.get_video(video_id)
        if vid:
            return {"video_id": video_id, "status": vid["status"], "progress": 100, "stage": "Ready", "message": "Video indexed."}
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "video_id": video_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["current_stage"],
        "message": job["message"]
    }


@router.get("/videos/{video_id}/clip/{topic_id}")
async def stream_clip(video_id: str, topic_id: str):
    """Stream or download an individual topic clip with HTTP 206 Range support."""
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
        raise HTTPException(status_code=404, detail="Clip file not found.")

    return FileResponse(
        path=str(clip_path),
        filename=clip_path.name,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{clip_path.name}"'}
    )


@router.get("/videos/{video_id}/clip/{topic_id}/subtitles")
async def get_clip_subtitles(video_id: str, topic_id: str):
    """Stream or download synchronized WebVTT subtitles with speaker tags for an individual clip."""
    clean_tid = topic_id.replace(f"{video_id}_", "")
    candidates = [
        OUTPUT_DIR / f"{video_id}_{clean_tid}.vtt",
        OUTPUT_DIR / f"{clean_tid}.vtt",
        OUTPUT_DIR / f"{video_id}_{topic_id}.vtt",
    ]
    vtt_path = next((p for p in candidates if p.exists()), None)
    if not vtt_path:
        raise HTTPException(status_code=404, detail="Subtitles not found for this clip.")

    return FileResponse(
        path=str(vtt_path),
        filename=vtt_path.name,
        media_type="text/vtt",
        headers={"Content-Disposition": f'inline; filename="{vtt_path.name}"'}
    )


@router.get("/videos/{video_id}/final")
async def stream_final(video_id: str):
    """Stream or download the final merged video."""
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
