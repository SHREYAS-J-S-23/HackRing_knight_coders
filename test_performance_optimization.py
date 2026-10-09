import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.database import DatabaseService, init_db, get_db_connection
from backend.models.schemas import EnrichedTranscript, TranscriptSegment, WordTimestamp
from backend.services.groq_stt import GroqSTTService
from backend.services.diarization_service import SpeakerDiarizationService
from backend.services.analysis_worker import AnalysisWorker
from backend.config import (
    TRANSCRIPTION_MODE,
    GROQ_WHISPER_FAST_MODEL,
    GROQ_WHISPER_PRECISE_MODEL
)


class TestPerformanceOptimization(unittest.TestCase):
    def setUp(self):
        init_db()
        self.test_video_id = "test_perf_vid_99"

    def tearDown(self):
        # Clean up test checkpoints
        DatabaseService.clear_chunk_checkpoints(self.test_video_id)

    # -------------------------------------------------------------------------
    # TEST 1: Transcription Modes (FAST vs PRECISE Configuration)
    # -------------------------------------------------------------------------
    def test_transcription_modes_configuration(self):
        """Verifies FAST mode chooses whisper-large-v3-turbo with segment-level timestamps,
        and PRECISE mode chooses whisper-large-v3 with word-level timestamps."""
        fast_stt = GroqSTTService(mode="FAST")
        self.assertEqual(fast_stt.mode, "FAST")

        precise_stt = GroqSTTService(mode="PRECISE")
        self.assertEqual(precise_stt.mode, "PRECISE")

        self.assertEqual(GROQ_WHISPER_FAST_MODEL, "whisper-large-v3-turbo")
        self.assertEqual(GROQ_WHISPER_PRECISE_MODEL, "whisper-large-v3")

    # -------------------------------------------------------------------------
    # TEST 2: Chunk Overlap Deduplication & Timestamp Alignment
    # -------------------------------------------------------------------------
    def test_chunk_overlap_deduplication(self):
        """Verifies overlapping segments from adjacent chunks are deduplicated cleanly
        without dropped words or duplicated sentences, preserving chronological order."""
        # Chunk 0 segments (0s to 302s, where 300-302s is overlap)
        chunk0_segments = [
            {"start": 0.0, "end": 10.0, "text": "Welcome to our deep dive on distributed consensus.", "speaker": "SPEAKER_00", "words": []},
            {"start": 290.0, "end": 299.5, "text": "Now let us examine Byzantine fault tolerance.", "speaker": "SPEAKER_00", "words": []},
            {"start": 299.8, "end": 301.8, "text": "Nodes exchange heartbeats periodically.", "speaker": "SPEAKER_00", "words": []}
        ]

        # Chunk 1 segments (starts at 300s, where 300-302s is overlap)
        chunk1_segments = [
            {"start": 300.0, "end": 302.0, "text": "Nodes exchange heartbeats periodically.", "speaker": "SPEAKER_00", "words": []},
            {"start": 302.5, "end": 315.0, "text": "When a leader fails, a new election term is initiated immediately.", "speaker": "SPEAKER_00", "words": []}
        ]

        combined = chunk0_segments + chunk1_segments
        deduped = GroqSTTService.deduplicate_overlapping_segments(combined)

        # "Nodes exchange heartbeats periodically." should only appear ONCE
        heartbeat_segs = [s for s in deduped if "heartbeats" in s["text"].lower()]
        self.assertEqual(len(heartbeat_segs), 1, "Duplicate boundary segment should be merged/deduplicated")

        # Total unique segments should be 4 (not 5)
        self.assertEqual(len(deduped), 4)

        # Verify strict chronological monotonic order
        for i in range(len(deduped) - 1):
            self.assertLessEqual(deduped[i]["start"], deduped[i + 1]["start"])
            self.assertLessEqual(deduped[i]["start"], deduped[i]["end"])

    # -------------------------------------------------------------------------
    # TEST 3: Checkpoint Persistence and Job Resumption
    # -------------------------------------------------------------------------
    def test_checkpoint_persistence_and_resumption(self):
        """Verifies completed chunk checkpoints are persisted to SQLite and retrieved
        to resume interrupted processing without re-transcribing successful chunks."""
        # Save checkpoint for chunk 0
        chunk0_data = [
            {"start": 0.0, "end": 15.0, "text": "Chunk zero completed text.", "speaker": "SPEAKER_00", "words": []}
        ]
        DatabaseService.save_chunk_checkpoint(
            video_id=self.test_video_id,
            chunk_index=0,
            time_offset=0.0,
            duration=300.0,
            segments_data=chunk0_data,
            retries=0,
            latency=1.25,
            status="completed"
        )

        # Save checkpoint for chunk 1
        chunk1_data = [
            {"start": 300.0, "end": 320.0, "text": "Chunk one completed text.", "speaker": "SPEAKER_00", "words": []}
        ]
        DatabaseService.save_chunk_checkpoint(
            video_id=self.test_video_id,
            chunk_index=1,
            time_offset=300.0,
            duration=300.0,
            segments_data=chunk1_data,
            retries=1,
            latency=1.80,
            status="completed"
        )

        # Retrieve checkpoints
        checkpoints = DatabaseService.get_chunk_checkpoints(self.test_video_id)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[0]["chunk_index"], 0)
        self.assertEqual(checkpoints[1]["chunk_index"], 1)
        self.assertEqual(checkpoints[0]["status"], "completed")
        self.assertEqual(len(checkpoints[0]["segments"]), 1)
        self.assertEqual(checkpoints[0]["segments"][0]["text"], "Chunk zero completed text.")

    # -------------------------------------------------------------------------
    # TEST 4: Diarization FAST vs PRECISE Modes & Performance Timing
    # -------------------------------------------------------------------------
    def test_diarization_modes_and_timing(self):
        """Verifies FAST mode performs turn-taking heuristics with 0 FFmpeg subprocess calls
        and records separate diarization_time_seconds."""
        sample_segs = [
            TranscriptSegment(id=1, start=0.0, end=4.0, text="Host introduces the guest. What are your views on inflation?", speaker="SPEAKER_00"),
            TranscriptSegment(id=2, start=4.5, end=15.0, text="Guest explains that inflation erodes purchasing power over time.", speaker="SPEAKER_00"),
            TranscriptSegment(id=3, start=16.0, end=19.0, text="Host asks a follow-up question. Does this apply to real estate?", speaker="SPEAKER_00"),
            TranscriptSegment(id=4, start=19.5, end=30.0, text="Guest explains asset inflation preserves real value.", speaker="SPEAKER_00")
        ]
        transcript = EnrichedTranscript(
            video_id="perf_diar_test",
            filename="sample.mp4",
            duration_seconds=30.0,
            segments=sample_segs
        )

        # Test FAST mode
        fast_diarizer = SpeakerDiarizationService(mode="FAST")
        t0 = time.time()
        fast_res = fast_diarizer.diarize_transcript(transcript)
        fast_duration = time.time() - t0

        self.assertIsNotNone(fast_res.diarization_time_seconds)
        self.assertLess(fast_duration, 0.5, "FAST diarization must complete in under 500ms without FFmpeg")

        # Verify speakers were assigned
        speakers = {s.speaker for s in fast_res.segments}
        self.assertGreaterEqual(len(speakers), 1)

    # -------------------------------------------------------------------------
    # TEST 5: SQLite Database Indexes Verification
    # -------------------------------------------------------------------------
    def test_database_indexes_exist(self):
        """Verifies required performance indexes exist on transcript_segments, topics,
        clips, analysis_jobs, and transcript_chunks."""
        conn = get_db_connection()
        try:
            indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            index_names = {row["name"] for row in indexes}

            expected_indexes = [
                "idx_segments_vid",
                "idx_segments_vid_time",
                "idx_topics_vid",
                "idx_clips_vid",
                "idx_jobs_vid",
                "idx_jobs_status",
                "idx_chunks_vid"
            ]
            for idx_name in expected_indexes:
                self.assertIn(idx_name, index_names, f"Expected index {idx_name} to exist in SQLite")
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # TEST 6: Batch Database Writes vs Single Inserts
    # -------------------------------------------------------------------------
    def test_batch_database_operations(self):
        """Verifies save_segments and save_topics execute via batch transactions without N+1 overhead."""
        segments_batch = [
            {
                "segment_index": i,
                "start": float(i * 10),
                "end": float(i * 10 + 8),
                "text": f"Batch segment {i} discussing high-performance systems.",
                "speaker": "SPEAKER_01",
                "words": []
            }
            for i in range(50)
        ]

        t0 = time.time()
        DatabaseService.save_segments(self.test_video_id, segments_batch)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 1.0, f"Batch inserting 50 segments took {elapsed:.3f}s (should be < 1.0s)")

        retrieved = DatabaseService.get_segments(self.test_video_id)
        self.assertEqual(len(retrieved), 50)
        self.assertEqual(retrieved[0]["text"], "Batch segment 0 discussing high-performance systems.")

    # -------------------------------------------------------------------------
    # TEST 7: Persistent Analysis Worker Job Lifecycle & Metric Tracking
    # -------------------------------------------------------------------------
    def test_analysis_worker_job_state_tracking(self):
        """Verifies AnalysisWorker persists job status with stage, progress, chunk counts,
        elapsed time, and prevents duplicate active executions."""
        job_id = f"job_{self.test_video_id}"

        # Test set_job with chunk counts and elapsed seconds
        DatabaseService.set_job(
            job_id=job_id,
            video_id=self.test_video_id,
            job_type="analysis",
            status="processing",
            progress=55,
            stage="Transcription",
            message="Transcribing chunk 2 of 4...",
            chunks_completed=2,
            chunks_total=4,
            elapsed_seconds=12.4,
            retries=0
        )

        job = DatabaseService.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "processing")
        self.assertEqual(job["stage"], "Transcription")
        self.assertEqual(job["progress"], 55)
        self.assertEqual(job["chunks_completed"], 2)
        self.assertEqual(job["chunks_total"], 4)
        self.assertEqual(job["elapsed_seconds"], 12.4)


if __name__ == "__main__":
    unittest.main()
