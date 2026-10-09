import os
import re
import math
import time
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from backend.config import get_ffmpeg_executable, DIARIZATION_MODE
from backend.models.schemas import EnrichedTranscript, TranscriptSegment, WordTimestamp

class SpeakerDiarizationService:
    """
    Robust multi-modal speaker diarization and role estimation service.
    Combines acoustic signal analysis (pitch, spectral energy, zero-crossing rate via FFmpeg/SciPy)
    with conversational dialogue linguistics (turn-taking, question-response dynamics, address terms).
    
    Modes:
    - FAST: Skip acoustic signal analysis; rely on conversational linguistic turn-taking cues (0 FFmpeg calls).
    - STANDARD: Lightweight turn dynamics and speaker-alternation heuristics.
    - PRECISE: Acoustic feature extraction using single-pass audio decode (1 FFmpeg pass total, no per-segment process spawns).
    """

    def __init__(self, num_speakers_hint: Optional[int] = None, mode: Optional[str] = None):
        self.num_speakers_hint = num_speakers_hint
        self.mode = (mode or DIARIZATION_MODE or "FAST").upper()

    def diarize_transcript(
        self,
        transcript: EnrichedTranscript,
        audio_path: Optional[Path] = None
    ) -> EnrichedTranscript:
        """
        Diarizes all segments in the transcript, assigning speaker IDs and estimating roles.
        Measures diarization time accurately.
        """
        start_time = time.time()
        segments = transcript.segments
        if not segments:
            return transcript

        # 1. Detect if distinct speaker labels already exist
        existing_speakers = set(s.speaker for s in segments if s.speaker and s.speaker not in ("SPEAKER_01", "unknown", ""))
        if len(existing_speakers) > 1:
            # Existing multi-speaker labels present; estimate roles and return
            roles = self._estimate_speaker_roles(segments)
            for s in segments:
                s.speaker_role = roles.get(s.speaker, "UNKNOWN")
                s.speaker_confidence = s.speaker_confidence or 0.90
            transcript.diarization_time_seconds = round(time.time() - start_time, 3)
            return transcript

        # 2. Extract acoustic turn signals if PRECISE mode is active and audio file is accessible
        acoustic_clusters = {}
        if self.mode == "PRECISE" and audio_path and Path(audio_path).exists():
            try:
                acoustic_clusters = self._extract_acoustic_clusters(audio_path, segments)
            except Exception as e:
                print(f"Vidara Diarization: Acoustic extraction notice ({e}), falling back to conversational turn model.")

        # 3. Conversational Linguistic Turn Analysis
        assigned_speakers = self._cluster_conversational_turns(segments, acoustic_clusters)

        # 4. Inferred Speaker Roles (Host vs Guest vs Participant)
        roles, role_confidences = self._estimate_roles_with_confidence(segments, assigned_speakers)

        # 5. Apply assigned speakers, roles, and confidence to segments and words
        for idx, s in enumerate(segments):
            spk_info = assigned_speakers.get(s.id, {"speaker": "SPEAKER_00", "confidence": 0.85})
            spk_id = spk_info["speaker"]
            spk_conf = spk_info["confidence"]

            s.speaker = spk_id
            s.speaker_confidence = round(spk_conf, 2)
            s.speaker_role = roles.get(spk_id, "UNKNOWN")

        transcript.diarization_time_seconds = round(time.time() - start_time, 3)
        return transcript

    # -------------------------------------------------------------------------
    # ACOUSTIC FEATURE EXTRACTION (SCIPY & SINGLE-PASS FFMPEG)
    # -------------------------------------------------------------------------
    def _extract_acoustic_clusters(
        self,
        audio_path: Path,
        segments: List[TranscriptSegment]
    ) -> Dict[int, str]:
        """
        Extracts sample audio features for segments and clusters them into distinct voices.
        OPTIMIZATION: Decodes audio to 16kHz mono WAV in a SINGLE pass, slicing in-memory
        instead of launching separate FFmpeg processes per segment.
        """
        import numpy as np
        from scipy.cluster.vq import kmeans2

        valid_segs = [s for s in segments if (s.end - s.start) >= 1.2 and len(s.text.split()) >= 3]
        if len(valid_segs) < 4:
            return {}

        sample_segs = valid_segs[:50]
        ffmpeg_bin = get_ffmpeg_executable()

        features = []
        seg_ids_used = []

        with tempfile.TemporaryDirectory() as tmpdir:
            pcm_wav = Path(tmpdir) / "full_16k_mono.wav"
            # Single-pass decode of audio to 16kHz mono WAV (max 1 FFmpeg process)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(audio_path),
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                str(pcm_wav)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and pcm_wav.exists() and pcm_wav.stat().st_size > 1000:
                try:
                    with open(pcm_wav, "rb") as f:
                        f.seek(44)  # Skip 44-byte WAV header
                        all_pcm = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32)

                    for s in sample_segs:
                        start_idx = int(s.start * 16000)
                        dur_samples = int(min(3.0, s.end - s.start) * 16000)
                        end_idx = min(len(all_pcm), start_idx + dur_samples)

                        if end_idx - start_idx > 1600:
                            pcm_slice = all_pcm[start_idx:end_idx]
                            # Feature 1: Zero-crossing rate
                            zcr = np.mean(np.abs(np.diff(np.sign(pcm_slice))))
                            # Feature 2: RMS energy
                            rms = np.sqrt(np.mean(pcm_slice ** 2))
                            # Feature 3: Autocorrelation peak (pitch proxy)
                            corr = np.correlate(pcm_slice[:2000], pcm_slice[:2000], mode="full")
                            corr = corr[len(corr)//2:]
                            peak_lag = np.argmax(corr[40:400]) + 40 if len(corr) >= 400 else 100
                            pitch_proxy = 16000.0 / max(1, peak_lag)

                            features.append([float(pitch_proxy), float(zcr * 1000), float(rms)])
                            seg_ids_used.append(s.id)
                except Exception as e:
                    print(f"Vidara Acoustic Clustering error: {e}")

        if len(features) < 6:
            return {}

        feats_np = np.array(features)
        feats_std = (feats_np - np.mean(feats_np, axis=0)) / (np.std(feats_np, axis=0) + 1e-6)

        k = self.num_speakers_hint or 2
        centroids, labels = kmeans2(feats_std, k=k, minit="points")

        cluster_map = {}
        for sid, label in zip(seg_ids_used, labels):
            cluster_map[sid] = f"SPEAKER_{label:02d}"

        return cluster_map

    # -------------------------------------------------------------------------
    # CONVERSATIONAL LINGUISTIC TURN-TAKING ANALYSIS
    # -------------------------------------------------------------------------
    def _cluster_conversational_turns(
        self,
        segments: List[TranscriptSegment],
        acoustic_clusters: Dict[int, str]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Assigns speaker IDs using dialogue turn cues, temporal pauses, vocatives,
        and acoustic cluster evidence.
        """
        assigned: Dict[int, Dict[str, Any]] = {}
        if not segments:
            return assigned

        # Conversational markers indicating dialogue responses / speaker change
        RESPONSE_MARKERS = [
            "yeah", "yes", "right", "exactly", "absolutely", "i agree", "well", "look",
            "no,", "no no", "actually", "so you're saying", "i mean", "to answer your question",
            "let me tell you", "that's a great question", "first of all"
        ]

        current_speaker = "SPEAKER_00"
        confidence = 0.85

        for i, s in enumerate(segments):
            text = s.text.strip().lower()
            prev_seg = segments[i - 1] if i > 0 else None
            time_gap = (s.start - prev_seg.end) if prev_seg else 0.0

            # 1. Acoustic ground truth check
            if s.id in acoustic_clusters:
                current_speaker = acoustic_clusters[s.id]
                confidence = 0.94
                assigned[s.id] = {"speaker": current_speaker, "confidence": confidence}
                continue

            # 2. Conversational shift indicators:
            is_turn_shift = False

            # Long pause between speech turns (> 0.8s) often signifies other speaker taking floor
            if time_gap >= 0.8:
                is_turn_shift = True

            # Preceding segment ended with question mark and current segment begins with response
            if prev_seg and prev_seg.text.strip().endswith(("?", "what?", "right?")):
                is_turn_shift = True

            # Segment begins with clear response marker or agreement
            if any(text.startswith(m) for m in RESPONSE_MARKERS):
                is_turn_shift = True

            # Vocatives (e.g. "Robert, what do you think?", "Sharan, exactly")
            if re.match(r'^[a-zA-Z]+,\s+(what|how|why|do you|can you|tell us)', text):
                is_turn_shift = True

            if is_turn_shift and prev_seg:
                # Alternate speaker
                prev_spk = assigned.get(prev_seg.id, {}).get("speaker", current_speaker)
                current_speaker = "SPEAKER_01" if prev_spk == "SPEAKER_00" else "SPEAKER_00"
                confidence = 0.88
            else:
                # Continue previous speaker turn
                if prev_seg and prev_seg.id in assigned:
                    current_speaker = assigned[prev_seg.id]["speaker"]
                    confidence = 0.91

            assigned[s.id] = {"speaker": current_speaker, "confidence": confidence}

        return assigned

    # -------------------------------------------------------------------------
    # ESTIMATE SPEAKER ROLES (HOST vs GUEST vs UNKNOWN)
    # -------------------------------------------------------------------------
    def _estimate_roles_with_confidence(
        self,
        segments: List[TranscriptSegment],
        assigned_speakers: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Estimates conversational roles (HOST, GUEST, UNKNOWN) by analyzing:
        - Question-asking frequency (Hosts ask leading questions)
        - Explanation and story duration (Guests speak longer narrative answers)
        - Introductions and episode wrap-ups
        
        Strict constraint: If the confidence is below 0.70, retain 'UNKNOWN' to avoid
        assuming the first speaker is always the host.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for s in segments:
            spk = assigned_speakers.get(s.id, {}).get("speaker", s.speaker or "SPEAKER_00")
            if spk not in stats:
                stats[spk] = {
                    "total_duration": 0.0,
                    "question_count": 0,
                    "explanation_words": 0,
                    "intro_cues": 0,
                    "segments": 0
                }
            dur = max(0.2, s.end - s.start)
            stats[spk]["total_duration"] += dur
            stats[spk]["segments"] += 1

            text = s.text.strip().lower()
            if text.endswith("?") or re.match(r'^(what|why|how|who|where|can you|tell me|do you)', text):
                stats[spk]["question_count"] += 1

            words = text.split()
            stats[spk]["explanation_words"] += len(words)

            if any(w in text for w in ["welcome to the show", "in today's episode", "our guest today", "thanks for coming on"]):
                stats[spk]["intro_cues"] += 1

        if len(stats) < 2:
            single_spk = list(stats.keys())[0] if stats else "SPEAKER_00"
            return {single_spk: "UNKNOWN"}, {single_spk: 0.50}

        speakers = list(stats.keys())
        spk_0, spk_1 = speakers[0], speakers[1]
        
        total_time = max(1.0, stats[spk_0]["total_duration"] + stats[spk_1]["total_duration"])
        
        def compute_host_affinity(spk: str) -> float:
            data = stats[spk]
            q_ratio = data["question_count"] / max(1, data["segments"])
            time_fraction = data["total_duration"] / total_time
            # Hosts ask questions, deliver intros, and speak less total narrative than the guest
            score = (q_ratio * 3.0) + (data["intro_cues"] * 4.0) + max(0.0, (0.5 - time_fraction) * 2.0)
            return score

        s0_score = compute_host_affinity(spk_0)
        s1_score = compute_host_affinity(spk_1)
        score_diff = abs(s0_score - s1_score)

        roles = {}
        confidences = {}

        if score_diff >= 0.8:
            host_spk = spk_0 if s0_score > s1_score else spk_1
            guest_spk = spk_1 if host_spk == spk_0 else spk_0
            conf = min(0.96, max(0.72, 0.70 + (score_diff * 0.1)))
            roles[host_spk] = "HOST"
            roles[guest_spk] = "GUEST"
            confidences[host_spk] = conf
            confidences[guest_spk] = conf
        else:
            roles[spk_0] = "UNKNOWN"
            roles[spk_1] = "UNKNOWN"
            confidences[spk_0] = 0.50
            confidences[spk_1] = 0.50

        return roles, confidences

    def _estimate_speaker_roles(self, segments: List[TranscriptSegment]) -> Dict[str, str]:
        """Convenience method for existing labeled segments."""
        assigned = {s.id: {"speaker": s.speaker, "confidence": 0.90} for s in segments}
        roles, _ = self._estimate_roles_with_confidence(segments, assigned)
        return roles
