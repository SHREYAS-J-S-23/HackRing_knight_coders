import json
from pathlib import Path
from typing import Optional
from groq import Groq
from backend.config import get_groq_stt_api_key, GROQ_WHISPER_MODEL
from backend.models.schemas import EnrichedTranscript, TranscriptSegment, WordTimestamp

class GroqSTTService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_groq_stt_api_key()
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def transcribe(self, audio_path: Path, video_id: str, duration: float = 0.0) -> EnrichedTranscript:
        """
        Transcribes the real audio using Groq Whisper large-v3 with word-level and segment-level timestamps.
        Supports 1hr, 2hr+ videos by automatically chunking audio into <= 10-minute pieces to bypass
        Groq's 25MB file size limit (HTTP 413 Request Entity Too Large).
        """
        from backend.services.audio_extractor import AudioExtractor

        key = self.api_key or get_groq_stt_api_key()
        if not key:
            raise ValueError(
                "GROQ_API_KEY / GROQ_STT_KEY is missing! Please provide your Groq API key in .env or via the Keys button."
            )

        client = Groq(api_key=key)

        # Split audio into <= 10-minute chunks if necessary
        chunks = AudioExtractor.split_audio(audio_path, chunk_duration=600.0)
        
        all_segments = []
        global_seg_idx = 1
        detected_language = "en"

        import concurrent.futures

        def transcribe_single_chunk(chunk_tuple):
            chunk_file, time_offset = chunk_tuple
            try:
                with open(chunk_file, "rb") as file_obj:
                    transcription = client.audio.transcriptions.create(
                        file=(chunk_file.name, file_obj),
                        model=GROQ_WHISPER_MODEL,
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"]
                    )
                return (time_offset, transcription)
            except Exception as e:
                print(f"Warning: Chunk transcription error at offset {time_offset}s: {e}")
                return (time_offset, None)
            finally:
                if chunk_file != audio_path and chunk_file.exists():
                    try:
                        chunk_file.unlink()
                    except Exception:
                        pass

        if len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
                chunk_results = list(executor.map(transcribe_single_chunk, chunks))
        else:
            chunk_results = [transcribe_single_chunk(chunks[0])]

        chunk_results.sort(key=lambda x: x[0])

        for time_offset, transcription in chunk_results:
            if not transcription:
                continue

            raw_segments = getattr(transcription, "segments", []) or []
            detected_language = getattr(transcription, "language", detected_language)

            for seg in raw_segments:
                seg_dict = seg if isinstance(seg, dict) else seg.__dict__
                words_list = []
                if "words" in seg_dict and seg_dict["words"]:
                    for w in seg_dict["words"]:
                        w_dict = w if isinstance(w, dict) else w.__dict__
                        w_start = round(float(w_dict.get("start", 0.0)) + time_offset, 3)
                        w_end = round(float(w_dict.get("end", 0.0)) + time_offset, 3)
                        words_list.append(
                            WordTimestamp(
                                word=w_dict.get("word", ""),
                                start=w_start,
                                end=w_end
                            )
                        )

                text_cleaned = seg_dict.get("text", "").strip()
                if not text_cleaned:
                    continue

                seg_start = round(float(seg_dict.get("start", 0.0)) + time_offset, 2)
                seg_end = round(float(seg_dict.get("end", 0.0)) + time_offset, 2)

                all_segments.append(
                    TranscriptSegment(
                        id=global_seg_idx,
                        start=seg_start,
                        end=seg_end,
                        text=text_cleaned,
                        speaker=seg_dict.get("speaker", "SPEAKER_01"),
                        words=words_list
                    )
                )
                global_seg_idx += 1

        segments = all_segments

        # If Whisper returned full text but no segments
        if not segments:
            full_text = getattr(transcription, "text", "").strip()
            if full_text:
                segments.append(
                    TranscriptSegment(
                        id=1,
                        start=0.0,
                        end=round(duration, 2) or 5.0,
                        text=full_text,
                        speaker="SPEAKER_01",
                        words=[]
                    )
                )
            else:
                # Actual silent video
                segments.append(
                    TranscriptSegment(
                        id=1,
                        start=0.0,
                        end=round(duration, 2) or 1.0,
                        text="[No speech detected in audio track]",
                        speaker="SPEAKER_01",
                        words=[]
                    )
                )

        detected_language = getattr(transcription, "language", "en")

        enriched = EnrichedTranscript(
            video_id=video_id,
            filename=audio_path.name,
            duration_seconds=duration or (segments[-1].end if segments else 0.0),
            language=detected_language,
            segments=segments
        )

        # Apply speaker diarization and role estimation
        try:
            from backend.services.diarization_service import SpeakerDiarizationService
            diarizer = SpeakerDiarizationService()
            enriched = diarizer.diarize_transcript(enriched, audio_path=audio_path)
        except Exception as diar_err:
            print(f"Vidara Diarization warning: {diar_err}, continuing with basic labels.")

        return enriched
