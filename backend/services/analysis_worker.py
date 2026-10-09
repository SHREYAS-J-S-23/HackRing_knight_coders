import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from backend.database import DatabaseService
from backend.services.audio_extractor import AudioExtractor
from backend.services.groq_stt import GroqSTTService
from backend.services.semantic_engine import SemanticEngine
from backend.services.topic_intelligence import VidaraTopicIntelligenceEngine
from backend.models.schemas import EnrichedTranscript, DiscoveredTopic

logger = logging.getLogger("vidara.worker")

# Thread lock to prevent concurrent redundant workers for the same video
WORKER_LOCKS: Dict[str, threading.Lock] = {}
ACTIVE_THREADS: Dict[str, threading.Thread] = {}
LOCKS_MUTEX = threading.Lock()


def get_video_lock(video_id: str) -> threading.Lock:
    with LOCKS_MUTEX:
        if video_id not in WORKER_LOCKS:
            WORKER_LOCKS[video_id] = threading.Lock()
        return WORKER_LOCKS[video_id]


class AnalysisWorker:
    """
    Persistent, durable background execution worker for video intelligence analysis.
    Stages:
    1. Media ingestion (10-20%)
    2. Audio extraction (20-30%)
    3. Transcription (30-65%, chunk-by-chunk progress)
    4. Semantic indexing (65-75%)
    5. Topic discovery (75-85%)
    6. Importance ranking (85-92%)
    7. Boundary validation (92-98%)
    8. Ready (100%)
    """

    @classmethod
    def start_analysis_job(
        cls,
        video_id: str,
        user_id: Optional[str] = None,
        video_cache_ref: Optional[Dict[str, Any]] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        Starts or resumes a background analysis job safely without duplication.
        Returns the job_id immediately.
        """
        job_id = f"job_{video_id}"

        # Check if already processing
        with LOCKS_MUTEX:
            if video_id in ACTIVE_THREADS and ACTIVE_THREADS[video_id].is_alive():
                logger.info(f"Analysis worker already running for video {video_id}.")
                return job_id

        # Initialize or update job status
        DatabaseService.set_job(
            job_id=job_id,
            video_id=video_id,
            job_type="analysis",
            status="processing",
            progress=10,
            stage="Media ingestion",
            message="Initializing video analysis pipeline...",
            chunks_completed=0,
            chunks_total=0,
            elapsed_seconds=0.0,
            user_id=user_id
        )

        def runner():
            lock = get_video_lock(video_id)
            if not lock.acquire(blocking=False):
                logger.warning(f"Could not acquire lock for video {video_id}; worker already active.")
                return

            try:
                cls._execute_analysis_pipeline(video_id, job_id, video_cache_ref)
                if on_complete:
                    try:
                        job_res = DatabaseService.get_job(job_id)
                        on_complete(job_res or {})
                    except Exception as cb_err:
                        logger.error(f"Worker callback error: {cb_err}")
            finally:
                lock.release()
                with LOCKS_MUTEX:
                    if video_id in ACTIVE_THREADS:
                        del ACTIVE_THREADS[video_id]

        t = threading.Thread(target=runner, name=f"VidaraWorker-{video_id}", daemon=True)
        with LOCKS_MUTEX:
            ACTIVE_THREADS[video_id] = t
        t.start()

        return job_id

    @classmethod
    def _execute_analysis_pipeline(
        cls,
        video_id: str,
        job_id: str,
        video_cache: Optional[Dict[str, Any]] = None
    ) -> None:
        start_time = time.time()
        logger.info(f"Starting analysis worker for video {video_id}...")

        try:
            # -------------------------------------------------------------
            # STAGE 1: Media Ingestion
            # -------------------------------------------------------------
            t_stage = time.time()
            vid_record = DatabaseService.get_video(video_id)
            if not vid_record:
                raise ValueError(f"Video {video_id} not found in database.")

            video_path = Path(vid_record["filepath"])
            audio_path_str = vid_record.get("audio_path")
            audio_path = Path(audio_path_str) if audio_path_str else None

            if not video_path.exists() and (not audio_path or not audio_path.exists()):
                raise FileNotFoundError(f"Media file {video_path} does not exist on disk.")

            duration = vid_record.get("duration") or (AudioExtractor.get_video_duration(video_path) if video_path.exists() else AudioExtractor.get_audio_duration(audio_path))
            filesize = vid_record.get("filesize") or (video_path.stat().st_size if video_path.exists() else (audio_path.stat().st_size if audio_path else 0))
            media_name = video_path.name if video_path.exists() else (audio_path.name if audio_path else "media")

            elapsed = time.time() - start_time
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=20,
                stage="Media ingestion",
                message=f"Media verified: {media_name} ({round(duration, 1)}s, {round(filesize / (1024*1024), 1)} MB).",
                elapsed_seconds=elapsed
            )

            # -------------------------------------------------------------
            # STAGE 2: Audio Extraction
            # -------------------------------------------------------------
            t_stage = time.time()
            audio_path_str = vid_record.get("audio_path")
            audio_path = Path(audio_path_str) if audio_path_str else None

            if not audio_path or not audio_path.exists():
                DatabaseService.set_job(
                    job_id=job_id,
                    video_id=video_id,
                    job_type="analysis",
                    status="processing",
                    progress=25,
                    stage="Audio extraction",
                    message="Extracting 16kHz mono audio track via FFmpeg...",
                    elapsed_seconds=time.time() - start_time
                )
                audio_path = AudioExtractor.extract_audio(video_path, f"{video_id}.mp3")
                # Update DB with audio path
                DatabaseService.save_video(
                    video_id=video_id,
                    filename=vid_record["filename"],
                    filepath=str(video_path),
                    audio_path=str(audio_path),
                    duration=duration,
                    filesize=filesize,
                    user_id=vid_record.get("user_id")
                )

            audio_extraction_time = time.time() - t_stage
            elapsed = time.time() - start_time
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=30,
                stage="Audio extraction",
                message=f"Audio extracted in {audio_extraction_time:.2f}s. Preparing Whisper STT...",
                elapsed_seconds=elapsed
            )

            # -------------------------------------------------------------
            # STAGE 3: Transcription
            # -------------------------------------------------------------
            t_stage = time.time()
            stt = GroqSTTService()

            def progress_callback(completed_chunks: int, total_chunks: int):
                pct = 30 + int(35 * (completed_chunks / max(1, total_chunks)))
                DatabaseService.set_job(
                    job_id=job_id,
                    video_id=video_id,
                    job_type="analysis",
                    status="processing",
                    progress=pct,
                    stage="Transcription",
                    message=f"Transcribing audio chunks: {completed_chunks}/{total_chunks} completed...",
                    chunks_completed=completed_chunks,
                    chunks_total=total_chunks,
                    elapsed_seconds=time.time() - start_time
                )

            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=35,
                stage="Transcription",
                message=f"Running {stt.mode} Whisper transcription...",
                elapsed_seconds=time.time() - start_time
            )

            transcript = stt.transcribe(
                audio_path=audio_path,
                video_id=video_id,
                duration=duration,
                progress_callback=progress_callback
            )

            # Persist transcript segments to DB
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

            transcription_time = time.time() - t_stage
            elapsed = time.time() - start_time
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=65,
                stage="Semantic indexing",
                message=f"Transcribed {len(transcript.segments)} segments in {transcription_time:.2f}s. Building semantic index...",
                elapsed_seconds=elapsed
            )

            # -------------------------------------------------------------
            # STAGE 4: Semantic Indexing
            # -------------------------------------------------------------
            t_stage = time.time()
            sem_engine = SemanticEngine()
            intent_graph = sem_engine.build_intent_graph(transcript)
            DatabaseService.update_video_status(video_id, "indexed", core_thesis=intent_graph.core_thesis)

            semantic_indexing_time = time.time() - t_stage
            elapsed = time.time() - start_time
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=75,
                stage="Topic discovery",
                message=f"Core thesis indexed in {semantic_indexing_time:.2f}s: '{intent_graph.core_thesis[:80]}...'. Discovering topics...",
                elapsed_seconds=elapsed
            )

            # -------------------------------------------------------------
            # STAGE 5: Topic Discovery & STAGE 6: Importance Ranking
            # -------------------------------------------------------------
            t_stage = time.time()
            topic_engine = VidaraTopicIntelligenceEngine()
            themes = intent_graph.themes if hasattr(intent_graph, "themes") else []

            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=82,
                stage="Topic discovery",
                message="Extracting conversational moments, podcast exchanges, and topic candidates...",
                elapsed_seconds=time.time() - start_time
            )

            discovered_topics = topic_engine.discover_important_topics(
                transcript,
                intent_graph.core_thesis,
                themes=themes
            )

            topic_discovery_time = time.time() - t_stage
            elapsed = time.time() - start_time
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=90,
                stage="Importance ranking",
                message=f"Scored & ranked {len(discovered_topics)} topics via PageRank centrality & editorial weighting.",
                elapsed_seconds=elapsed
            )

            # -------------------------------------------------------------
            # STAGE 7: Boundary Validation
            # -------------------------------------------------------------
            t_stage = time.time()
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="processing",
                progress=94,
                stage="Boundary validation",
                message="Verifying complete sentence boundaries and 8-point meaning preservation...",
                elapsed_seconds=time.time() - start_time
            )

            # Persist discovered topics to database
            DatabaseService.save_topics(video_id, [t.model_dump() for t in discovered_topics])
            boundary_validation_time = time.time() - t_stage

            # -------------------------------------------------------------
            # STAGE 8: Ready
            # -------------------------------------------------------------
            total_duration = time.time() - start_time
            performance_stats = {
                "total_processing_seconds": round(total_duration, 2),
                "audio_extraction_seconds": round(audio_extraction_time, 2),
                "transcription_seconds": round(transcription_time, 2),
                "diarization_seconds": round(transcript.diarization_time_seconds or 0.0, 2),
                "semantic_indexing_seconds": round(semantic_indexing_time, 2),
                "topic_discovery_seconds": round(topic_discovery_time, 2),
                "boundary_validation_seconds": round(boundary_validation_time, 2),
                "segment_count": len(transcript.segments),
                "topic_count": len(discovered_topics),
                "core_thesis": intent_graph.core_thesis,
                "video_duration_seconds": round(duration, 2),
                "mode": stt.mode
            }

            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="completed",
                progress=100,
                stage="Ready",
                message=f"Analysis complete in {total_duration:.2f}s! Discovered {len(discovered_topics)} high-value scenes.",
                result=performance_stats,
                elapsed_seconds=total_duration
            )

            # Update cache if reference was passed
            if video_cache is not None:
                video_cache[video_id] = {
                    "video_id": video_id,
                    "filename": vid_record["filename"],
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "duration": duration,
                    "transcript": transcript,
                    "intent_graph": intent_graph,
                    "topics": discovered_topics,
                    "clips": DatabaseService.get_clips(video_id)
                }

            logger.info(f"Analysis worker completed successfully for video {video_id} in {total_duration:.2f}s.")

        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Analysis worker error on video {video_id}: {err_msg}")
            DatabaseService.set_job(
                job_id=job_id,
                video_id=video_id,
                job_type="analysis",
                status="failed",
                progress=0,
                stage="Failed",
                message=f"Analysis failed: {err_msg}",
                error=err_msg,
                elapsed_seconds=time.time() - start_time
            )
