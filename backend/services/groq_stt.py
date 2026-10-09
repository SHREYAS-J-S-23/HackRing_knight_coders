import os
import time
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from groq import Groq

from backend.config import (
    get_groq_stt_api_key,
    GROQ_WHISPER_MODEL,
    GROQ_WHISPER_FAST_MODEL,
    GROQ_WHISPER_PRECISE_MODEL,
    TRANSCRIPTION_MODE,
    TRANSCRIPTION_CHUNK_SECONDS,
    TRANSCRIPTION_OVERLAP_SECONDS,
    TRANSCRIPTION_MAX_CONCURRENCY,
    TRANSCRIPTION_MAX_RETRIES,
    TRANSCRIPTION_TIMEOUT_SECONDS
)
from backend.models.schemas import EnrichedTranscript, TranscriptSegment, WordTimestamp
from backend.database import DatabaseService


class GroqSTTService:
    """
    Optimized speech-to-text service for Groq LPUs.
    Features:
    - FAST mode (whisper-large-v3-turbo, segment-level timestamps) for ultra-low latency.
    - PRECISE mode (whisper-large-v3, word-level timestamps) for exact clip alignment.
    - Configurable chunk size and boundary overlap to prevent dropped boundary words.
    - Overlap deduplication and chronological ordering.
    - Bounded concurrency with exponential backoff and jitter on rate limits.
    - Persistent chunk-level checkpointing for resuming interrupted jobs.
    """

    def __init__(self, api_key: Optional[str] = None, mode: Optional[str] = None):
        self.api_key = api_key or get_groq_stt_api_key()
        self.mode = (mode or TRANSCRIPTION_MODE or "FAST").upper()

    def _get_client(self) -> Groq:
        key = self.api_key or get_groq_stt_api_key()
        if not key:
            raise ValueError(
                "GROQ_API_KEY / GROQ_STT_KEY is missing! Please provide your Groq API key in .env or via the Keys button."
            )
        return Groq(api_key=key, timeout=TRANSCRIPTION_TIMEOUT_SECONDS)

    def transcribe(
        self,
        audio_path: Path,
        video_id: str,
        duration: float = 0.0,
        mode: Optional[str] = None,
        progress_callback: Optional[Any] = None
    ) -> EnrichedTranscript:
        """
        Transcribes audio using configurable FAST or PRECISE mode, chunked processing,
        checkpoint resumption, and bounded concurrency.
        """
        active_mode = (mode or self.mode).upper()
        start_overall_time = time.time()

        from backend.services.audio_extractor import AudioExtractor

        client = self._get_client()

        # Determine model and granularities based on mode
        if active_mode == "FAST":
            primary_model = GROQ_WHISPER_FAST_MODEL or "whisper-large-v3-turbo"
            fallback_model = GROQ_WHISPER_PRECISE_MODEL or "whisper-large-v3"
            granularities = ["segment"]
        else:  # PRECISE
            primary_model = GROQ_WHISPER_PRECISE_MODEL or "whisper-large-v3"
            fallback_model = GROQ_WHISPER_FAST_MODEL or "whisper-large-v3-turbo"
            granularities = ["segment", "word"]

        chunk_dur = TRANSCRIPTION_CHUNK_SECONDS
        overlap_sec = TRANSCRIPTION_OVERLAP_SECONDS
        max_workers = max(1, min(TRANSCRIPTION_MAX_CONCURRENCY, 4))
        max_retries = max(1, TRANSCRIPTION_MAX_RETRIES)

        # Split audio into chunks with overlap
        chunks = AudioExtractor.split_audio(
            audio_path,
            chunk_duration=chunk_dur,
            overlap=overlap_sec
        )

        total_chunks = len(chunks)
        completed_chunk_segments: Dict[int, List[Dict[str, Any]]] = {}
        detected_languages: List[str] = []

        # Check existing checkpoints for resume capability
        existing_checkpoints = DatabaseService.get_chunk_checkpoints(video_id)
        for cp in existing_checkpoints:
            c_idx = cp.get("chunk_index")
            if c_idx is not None and cp.get("status") == "completed":
                completed_chunk_segments[c_idx] = cp.get("segments", [])

        # Filter chunks that need transcription
        chunks_to_process = []
        for idx, chunk_tuple in enumerate(chunks):
            if idx in completed_chunk_segments:
                continue
            chunks_to_process.append((idx, chunk_tuple))

        # Helper to transcribe a single chunk with exponential backoff
        def transcribe_chunk_with_retry(item):
            idx, (chunk_file, time_offset) = item
            chunk_t0 = time.time()
            retries = 0
            last_err = ""
            models_to_try = [primary_model, fallback_model]

            for attempt in range(max_retries):
                model = models_to_try[min(attempt, len(models_to_try) - 1)]
                try:
                    with open(chunk_file, "rb") as file_obj:
                        resp = client.audio.transcriptions.create(
                            file=(chunk_file.name, file_obj),
                            model=model,
                            response_format="verbose_json",
                            timestamp_granularities=granularities
                        )

                    chunk_latency = time.time() - chunk_t0
                    raw_segs = getattr(resp, "segments", []) or []
                    lang = getattr(resp, "language", "en")

                    # Convert raw segments to dictionary representations with offset correction
                    parsed_segs = []
                    for seg in raw_segs:
                        seg_dict = seg if isinstance(seg, dict) else seg.__dict__
                        text_cleaned = seg_dict.get("text", "").strip()
                        if not text_cleaned:
                            continue

                        words_list = []
                        if "words" in seg_dict and seg_dict["words"]:
                            for w in seg_dict["words"]:
                                w_dict = w if isinstance(w, dict) else w.__dict__
                                words_list.append({
                                    "word": w_dict.get("word", ""),
                                    "start": round(float(w_dict.get("start", 0.0)) + time_offset, 3),
                                    "end": round(float(w_dict.get("end", 0.0)) + time_offset, 3)
                                })

                        parsed_segs.append({
                            "start": round(float(seg_dict.get("start", 0.0)) + time_offset, 2),
                            "end": round(float(seg_dict.get("end", 0.0)) + time_offset, 2),
                            "text": text_cleaned,
                            "speaker": seg_dict.get("speaker", "SPEAKER_01"),
                            "words": words_list
                        })

                    # If no segments but text present
                    if not parsed_segs:
                        full_txt = getattr(resp, "text", "").strip()
                        if full_txt:
                            parsed_segs.append({
                                "start": round(time_offset, 2),
                                "end": round(time_offset + chunk_dur, 2),
                                "text": full_txt,
                                "speaker": "SPEAKER_01",
                                "words": []
                            })

                    # Persist chunk checkpoint for resumption
                    DatabaseService.save_chunk_checkpoint(
                        video_id=video_id,
                        chunk_index=idx,
                        time_offset=time_offset,
                        duration=chunk_dur,
                        segments_data=parsed_segs,
                        retries=retries,
                        latency=chunk_latency,
                        status="completed"
                    )

                    return (idx, time_offset, parsed_segs, lang, None)

                except Exception as e:
                    retries += 1
                    last_err = str(e)
                    err_lower = last_err.lower()

                    # Permanent errors (invalid API key, bad request) should abort immediately
                    if "invalid_api_key" in err_lower or "401" in err_lower or "unauthorized" in err_lower:
                        print(f"Vidara STT Fatal Error: {last_err}")
                        break

                    # Transient error (429 rate limit, 503, connection timeout) -> backoff with jitter
                    backoff = (2.0 ** attempt) * 1.5 + random.uniform(0.1, 0.6)
                    print(f"Vidara STT Retry {retries}/{max_retries} for chunk {idx} at {time_offset}s ({last_err}), waiting {backoff:.2f}s...")
                    time.sleep(backoff)

                finally:
                    # Cleanup temporary chunk file if not original audio
                    if chunk_file != audio_path and chunk_file.exists():
                        try:
                            chunk_file.unlink()
                        except Exception:
                            pass

            # If all retries failed, persist failed checkpoint
            DatabaseService.save_chunk_checkpoint(
                video_id=video_id,
                chunk_index=idx,
                time_offset=time_offset,
                duration=chunk_dur,
                segments_data=[],
                retries=retries,
                latency=time.time() - chunk_t0,
                status="failed",
                error=last_err
            )
            return (idx, time_offset, [], "en", last_err)

        # Execute transcription with bounded concurrency
        import concurrent.futures

        if chunks_to_process:
            if len(chunks_to_process) > 1 and max_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(transcribe_chunk_with_retry, chunks_to_process))
            else:
                results = [transcribe_chunk_with_retry(item) for item in chunks_to_process]

            for idx, time_offset, parsed_segs, lang, err in results:
                completed_chunk_segments[idx] = parsed_segs
                if lang:
                    detected_languages.append(lang)
                if progress_callback:
                    try:
                        progress_callback(len(completed_chunk_segments), total_chunks)
                    except Exception:
                        pass

        # Assemble and deduplicate segments across all chunks in chronological order
        raw_combined_segments = []
        for idx in sorted(completed_chunk_segments.keys()):
            raw_combined_segments.extend(completed_chunk_segments[idx])

        # Deduplicate overlapping segments across adjacent chunks
        deduped = self.deduplicate_overlapping_segments(raw_combined_segments)

        # Build schema objects
        final_segments: List[TranscriptSegment] = []
        for seg_idx, s in enumerate(deduped, start=1):
            words_objs = [
                WordTimestamp(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"]
                ) for w in s.get("words", [])
            ]
            final_segments.append(
                TranscriptSegment(
                    id=seg_idx,
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    speaker=s.get("speaker", "SPEAKER_01"),
                    words=words_objs
                )
            )

        # If completely empty
        if not final_segments:
            final_segments.append(
                TranscriptSegment(
                    id=1,
                    start=0.0,
                    end=round(duration, 2) or 1.0,
                    text="[No speech detected in audio track]",
                    speaker="SPEAKER_01",
                    words=[]
                )
            )

        lang = detected_languages[0] if detected_languages else "en"
        transcription_time = round(time.time() - start_overall_time, 3)

        enriched = EnrichedTranscript(
            video_id=video_id,
            filename=audio_path.name,
            duration_seconds=duration or (final_segments[-1].end if final_segments else 0.0),
            language=lang,
            segments=final_segments,
            transcription_time_seconds=transcription_time
        )

        # Apply speaker diarization and role estimation
        try:
            from backend.services.diarization_service import SpeakerDiarizationService
            diarizer = SpeakerDiarizationService()
            enriched = diarizer.diarize_transcript(enriched, audio_path=audio_path)
        except Exception as diar_err:
            print(f"Vidara Diarization notice: {diar_err}, continuing with basic labels.")

        return enriched

    @staticmethod
    def deduplicate_overlapping_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates overlapping transcript segments resulting from chunk boundary overlap.
        Preserves chronological ordering, avoids double-counted phrases, and ensures
        boundary sentences are fully represented.
        """
        if not segments:
            return []

        # Sort strictly by start time, then end time
        sorted_segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
        deduped: List[Dict[str, Any]] = []

        for seg in sorted_segs:
            text = seg["text"].strip()
            if not text:
                continue

            if not deduped:
                deduped.append(seg)
                continue

            prev = deduped[-1]
            prev_text = prev["text"].strip().lower()
            curr_text = text.lower()

            # Check if this segment completely duplicates previous segment text in overlap window
            if curr_text == prev_text and abs(seg["start"] - prev["start"]) < 5.0:
                # Merge words if current has words and previous does not
                if seg.get("words") and not prev.get("words"):
                    prev["words"] = seg["words"]
                continue

            # Check if current segment starts inside previous segment with near-identical beginning
            if seg["start"] < prev["end"]:
                overlap_dur = prev["end"] - seg["start"]
                # If high text similarity or containment
                if curr_text in prev_text or prev_text in curr_text:
                    if len(curr_text) > len(prev_text):
                        prev["text"] = seg["text"]
                        prev["end"] = max(prev["end"], seg["end"])
                        if seg.get("words"):
                            prev["words"] = seg["words"]
                    continue
                elif overlap_dur > 0.3:
                    # Clip boundaries monotonically so start time doesn't precede prev end
                    seg_start = round(max(seg["start"], prev["end"]), 2)
                    if seg["end"] > seg_start:
                        seg["start"] = seg_start

            deduped.append(seg)

        return deduped

    def align_selected_range(self, audio_path: Path, start_time: float, end_time: float) -> List[WordTimestamp]:
        """
        Selectively fetches word-level millisecond timestamps for a targeted clip boundary
        range [start_time, end_time], avoiding whole-video word alignment overhead.
        """
        client = self._get_client()
        dur = max(0.5, end_time - start_time)
        from backend.config import get_ffmpeg_executable
        import tempfile
        import subprocess

        ffmpeg_bin = get_ffmpeg_executable()
        words_out: List[WordTimestamp] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            slice_path = Path(tmpdir) / "slice.mp3"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(audio_path),
                "-t", f"{dur:.3f}",
                "-acodec", "libmp3lame",
                "-ar", "16000",
                "-ac", "1",
                "-b:a", "32k",
                str(slice_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and slice_path.exists():
                try:
                    with open(slice_path, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            file=(slice_path.name, f),
                            model=GROQ_WHISPER_PRECISE_MODEL or "whisper-large-v3",
                            response_format="verbose_json",
                            timestamp_granularities=["word"]
                        )
                    raw_words = getattr(resp, "words", []) or []
                    for w in raw_words:
                        w_dict = w if isinstance(w, dict) else w.__dict__
                        words_out.append(
                            WordTimestamp(
                                word=w_dict.get("word", ""),
                                start=round(float(w_dict.get("start", 0.0)) + start_time, 3),
                                end=round(float(w_dict.get("end", 0.0)) + start_time, 3)
                            )
                        )
                except Exception as e:
                    print(f"Vidara selective word alignment notice: {e}")

        return words_out
