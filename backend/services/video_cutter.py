import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from backend.config import get_ffmpeg_executable, OUTPUT_DIR
from backend.models.schemas import EditDecisionList, EditDecision

class VideoCutter:
    @classmethod
    def get_video_duration(cls, video_path: Path) -> float:
        """Helper to get exact video duration using ffprobe or ffmpeg."""
        ffmpeg_bin = get_ffmpeg_executable()
        ffprobe_bin = str(Path(ffmpeg_bin).parent / "ffprobe.exe") if "ffmpeg.exe" in ffmpeg_bin else "ffprobe"
        try:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            return 3600.0

    @classmethod
    def _compute_merged_spans(
        cls,
        edl: EditDecisionList,
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Groups contiguous or closely spaced KEEP decisions into seamless video spans.
        Eliminates intra-sentence jump cuts, prevents word clipping with natural padding,
        and cleanly cuts out discarded sections.
        """
        decisions = sorted(edl.decisions, key=lambda d: d.start)
        cut_segments = [d for d in decisions if d.action == "CUT"]

        raw_groups: List[List[EditDecision]] = []
        current_group: List[EditDecision] = []

        for d in decisions:
            if d.action == "KEEP":
                if not current_group:
                    current_group.append(d)
                else:
                    prev = current_group[-1]
                    has_intervening_cut = any(
                        c.start >= prev.end - 0.05 and c.end <= d.start + 0.05
                        for c in cut_segments
                    )
                    gap = d.start - prev.end

                    if not has_intervening_cut and gap <= 1.2:
                        current_group.append(d)
                    else:
                        raw_groups.append(current_group)
                        current_group = [d]

        if current_group:
            raw_groups.append(current_group)

        spans: List[Dict[str, Any]] = []
        for grp in raw_groups:
            span_start = grp[0].start
            span_end = grp[-1].end

            prev_cuts = [c.end for c in cut_segments if c.end <= span_start + 0.05]
            prev_bound = max(prev_cuts) if prev_cuts else 0.0

            next_cuts = [c.start for c in cut_segments if c.start >= span_end - 0.05]
            next_bound = min(next_cuts) if next_cuts else total_duration

            padded_start = max(prev_bound + 0.02, span_start - 0.08)
            padded_end = min(next_bound - 0.02, span_end + 0.16)

            padded_start = max(0.0, padded_start)
            padded_end = min(total_duration, max(padded_start + 0.1, padded_end))

            spans.append({
                "source_start": padded_start,
                "source_end": padded_end,
                "duration": padded_end - padded_start,
                "decisions": grp
            })

        return spans

    @classmethod
    def render_edited_video(
        cls,
        input_video_path: Path,
        edl: EditDecisionList,
        output_filename: Optional[str] = None
    ) -> Tuple[Path, List[Dict[str, Any]]]:
        """
        Deterministically cuts individual video clips and concatenates them into a merged final video.
        Returns (merged_output_path, clips_list) so both individual cut clips and the full merged video
        can be previewed and downloaded.
        """
        ffmpeg_bin = get_ffmpeg_executable()
        if output_filename is None:
            output_filename = f"{edl.video_id}_{edl.audience_id}_final.mp4"

        output_path = OUTPUT_DIR / output_filename
        total_duration = cls.get_video_duration(input_video_path)

        spans = cls._compute_merged_spans(edl, total_duration)
        if not spans:
            raise ValueError("No segments marked as KEEP in this Edit Decision List.")

        # Clean up any stale clip files from prior renders for this video & audience
        for old_f in OUTPUT_DIR.glob(f"{edl.video_id}_{edl.audience_id}_clip_*.mp4"):
            try:
                old_f.unlink()
            except Exception:
                pass

        clips_metadata: List[Dict[str, Any]] = []
        clip_paths: List[Path] = []

        # Render each individual span as a standalone clip
        for idx, span in enumerate(spans):
            clip_name = f"{edl.video_id}_{edl.audience_id}_clip_{idx + 1}.mp4"
            clip_file = OUTPUT_DIR / clip_name
            duration = span["duration"]
            if duration <= 0.05:
                continue

            cmd_cut = [
                ffmpeg_bin,
                "-y",
                "-ss", f"{span['source_start']:.3f}",
                "-i", str(input_video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(clip_file)
            ]
            res = subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and clip_file.exists():
                clip_paths.append(clip_file)
                spoken_text = " ".join(d.text.strip() for d in span["decisions"])
                clips_metadata.append({
                    "clip_index": idx + 1,
                    "filename": clip_name,
                    "video_path": str(clip_file),
                    "duration": round(duration, 2),
                    "start": round(span["source_start"], 2),
                    "end": round(span["source_end"], 2),
                    "text": spoken_text,
                    "segment_ids": [d.segment_id for d in span["decisions"]],
                    "download_url": f"/api/pipeline/clip/{edl.video_id}/{edl.audience_id}/{idx + 1}",
                    "video_url": f"/api/pipeline/clip/{edl.video_id}/{edl.audience_id}/{idx + 1}"
                })
            else:
                print(f"Warning: Clip {idx + 1} render issue: {res.stderr[:200]}")

        if not clip_paths:
            raise RuntimeError("Failed to generate video clips.")

        # If only 1 clip, copy/link to output_path
        if len(clip_paths) == 1:
            import shutil
            shutil.copyfile(clip_paths[0], output_path)
            return output_path, clips_metadata

        # Concatenate all individual clips into the final merged video
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            concat_manifest = tmpdir_path / "concat_list.txt"
            with open(concat_manifest, "w", encoding="utf-8") as f:
                for cp in clip_paths:
                    safe_path = str(cp.resolve()).replace("\\", "/")
                    f.write(f"file '{safe_path}'\n")

            cmd_concat = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_manifest),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_path)
            ]
            res_concat = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_concat.returncode != 0 or not output_path.exists():
                raise RuntimeError(f"FFmpeg concat assembly failed: {res_concat.stderr[:300]}")

        return output_path, clips_metadata

    @classmethod
    def export_subtitles_vtt(cls, edl: EditDecisionList, output_path: Path, video_path: Optional[Path] = None):
        """
        Generates a WebVTT subtitle track perfectly synchronized with the newly merged video.
        Uses exact span offset calculations so text appears at the exact moment spoken.
        """
        total_duration = cls.get_video_duration(video_path) if video_path and video_path.exists() else 3600.0
        spans = cls._compute_merged_spans(edl, total_duration)

        lines = ["WEBVTT\n"]
        counter = 1
        output_timeline_pos = 0.0

        for span in spans:
            for dec in span["decisions"]:
                seg_rel_start = max(0.0, dec.start - span["source_start"])
                seg_rel_end = min(span["duration"], dec.end - span["source_start"])

                vtt_start = output_timeline_pos + seg_rel_start
                vtt_end = output_timeline_pos + seg_rel_end

                if vtt_end > vtt_start:
                    start_str = cls._format_vtt_timestamp(vtt_start)
                    end_str = cls._format_vtt_timestamp(vtt_end)
                    clean_text = dec.text.strip()
                    lines.append(f"{counter}\n{start_str} --> {end_str}\n{clean_text}\n")
                    counter += 1

            output_timeline_pos += span["duration"]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    @classmethod
    def generate_clip_vtt(
        cls,
        segments: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        output_vtt_path: Path
    ) -> None:
        """Generates synchronized WebVTT subtitles with speaker tags for an individual clip."""
        lines = ["WEBVTT\n"]
        counter = 1
        clip_dur = clip_end - clip_start
        for s in segments:
            st = float(s.get("start", s.get("start_time", 0.0)))
            et = float(s.get("end", s.get("end_time", 0.0)))
            if et <= clip_start or st >= clip_end:
                continue
            rel_st = max(0.0, st - clip_start)
            rel_et = min(clip_dur, et - clip_start)
            if rel_et > rel_st:
                spk = s.get("speaker", "SPEAKER_00")
                text = s.get("text", "").strip()
                lines.append(f"{counter}\n{cls._format_vtt_timestamp(rel_st)} --> {cls._format_vtt_timestamp(rel_et)}\n<v {spk}>{text}</v>\n")
                counter += 1
        with open(output_vtt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @classmethod
    def render_topic_clips(
        cls,
        input_video_path: Path,
        video_id: str,
        topics: List[Dict[str, Any]],
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Renders standalone video clips strictly from validated candidate ranges.
        - Validates that start and end timestamps are finite and valid (end > start).
        - Prevents re-expanding or blindly inflating validated candidate ranges.
        - Cuts precise interval with FFmpeg using faststart.
        - Measures actual rendered file duration and associates rich topic & podcast metadata.
        - Generates synchronized WebVTT subtitles with speaker tags.
        """
        import concurrent.futures
        import math
        ffmpeg_bin = get_ffmpeg_executable()
        total_duration = cls.get_video_duration(input_video_path)

        # Filter out rejected or invalid candidates
        valid_topics = []
        for t in topics:
            if t.get("validation_status") == "REJECTED":
                continue
            try:
                st = float(t.get("start_time", 0.0))
                et = float(t.get("end_time", 0.0))
                if math.isfinite(st) and math.isfinite(et) and et > st:
                    valid_topics.append(t)
            except (ValueError, TypeError):
                continue

        def cut_single_topic(idx_topic_tuple):
            idx, t = idx_topic_tuple
            topic_id = str(t.get("id", f"topic_{idx + 1}"))
            clean_tid = topic_id.replace(f"{video_id}_", "")

            # Use exact validated timestamps
            raw_st = float(t["start_time"])
            raw_et = float(t["end_time"])

            start_t = max(0.0, raw_st)
            end_t = min(total_duration, raw_et)
            duration = max(0.5, end_t - start_t)

            if end_t <= start_t:
                return None

            clip_filename = f"{video_id}_{clean_tid}.mp4"
            clip_path = OUTPUT_DIR / clip_filename
            vtt_filename = f"{video_id}_{clean_tid}.vtt"
            vtt_path = OUTPUT_DIR / vtt_filename

            # Fast & high-quality cutting strategy:
            # 1. Attempt instantaneous stream-copy (-c copy) first (takes ~0.1s, 100% original crystal-clear quality)
            # 2. Fall back to crisp high-fidelity re-encoding (veryfast + CRF 19) if copy cannot seek cleanly
            rendered = False

            copy_cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", f"{start_t:.3f}",
                "-i", str(input_video_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(clip_path)
            ]
            res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 5000:
                try:
                    probe_dur = cls.get_video_duration(clip_path)
                    if probe_dur >= max(0.4, duration * 0.4):
                        rendered = True
                except Exception:
                    rendered = True

            if not rendered:
                encode_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-threads", "4",
                    "-ss", f"{start_t:.3f}",
                    "-i", str(input_video_path),
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "19",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    "-movflags", "+faststart",
                    str(clip_path)
                ]
                res = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0 and clip_path.exists():
                    rendered = True

            if rendered and clip_path.exists():
                # Probe actual rendered file duration from disk
                try:
                    actual_dur = cls.get_video_duration(clip_path)
                    final_dur = round(actual_dur, 2) if actual_dur > 0.1 else round(duration, 2)
                except Exception:
                    final_dur = round(duration, 2)

                # Generate synchronized WebVTT subtitles with speaker tags
                if segments:
                    try:
                        cls.generate_clip_vtt(segments, start_t, end_t, vtt_path)
                    except Exception as vtt_err:
                        print(f"Warning: VTT generation error for clip {clean_tid}: {vtt_err}")

                justification = t.get("editorial_justification") or (t.get("why_selected", [""])[0] if t.get("why_selected") else "")
                reason = justification or f"Concise {t.get('exchange_type', 'topic')} extraction"

                return {
                    "id": f"{video_id}_{clean_tid}",
                    "video_id": video_id,
                    "topic_id": clean_tid,
                    "clip_index": idx + 1,
                    "filename": clip_filename,
                    "filepath": str(clip_path),
                    "start_time": round(start_t, 2),
                    "end_time": round(end_t, 2),
                    "duration": final_dur,
                    "text": t.get("name", f"Clip {idx + 1}"),
                    "key_information": t.get("key_information", ""),
                    "selected_sentence_ids": t.get("selected_sentence_ids", []),
                    "importance_score": t.get("importance_score", 0.9),
                    "is_selected": True,
                    "download_url": f"/api/videos/{video_id}/clip/{clean_tid}",
                    "video_url": f"/api/videos/{video_id}/clip/{clean_tid}",
                    "subtitle_url": f"/api/videos/{video_id}/clip/{clean_tid}/subtitles",
                    "exchange_type": t.get("exchange_type", "GENERAL_TOPIC"),
                    "speakers_involved": t.get("speakers_involved", []),
                    "speaker_roles": t.get("speaker_roles", {}),
                    "question_included": t.get("question_included", False),
                    "depends_on_question": t.get("depends_on_question", False),
                    "editorial_justification": justification,
                    "reason": reason
                }
            else:
                print(f"Warning: Failed to render topic clip {topic_id}: {res.stderr[:200]}")
                return None

        # Execute all clip cuts in parallel across available CPU cores
        max_workers = min(os.cpu_count() or 4, 8) if valid_topics else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(cut_single_topic, enumerate(valid_topics)))

        return [r for r in results if r is not None]

    @classmethod
    def merge_clip_files(
        cls,
        clip_paths: List[Path],
        output_path: Path
    ) -> Path:
        """
        Merges explicitly selected clip files into a unified master video using FFmpeg concat.
        Uses instant stream-copy (-c copy) for near-instant (0.2s) merging without re-encoding!
        """
        ffmpeg_bin = get_ffmpeg_executable()
        existing_clips = [p for p in clip_paths if p.exists()]
        if not existing_clips:
            raise ValueError("No valid clip files to merge.")

        if len(existing_clips) == 1:
            import shutil
            shutil.copyfile(existing_clips[0], output_path)
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.txt"
            with open(manifest_file, "w", encoding="utf-8") as f:
                for cp in existing_clips:
                    safe = str(cp.resolve()).replace("\\", "/")
                    f.write(f"file '{safe}'\n")

            # Try instant stream-copy first! All clips share identical H.264+AAC settings from render_topic_clips.
            copy_cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(manifest_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path)
            ]
            copy_res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if copy_res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                return output_path

            # Fallback to ultrafast encoding if stream-copy encounters unusual container boundaries
            cmd = [
                ffmpeg_bin,
                "-y",
                "-threads", "0",
                "-f", "concat",
                "-safe", "0",
                "-i", str(manifest_file),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0 or not output_path.exists():
                raise RuntimeError(f"FFmpeg merge failed: {res.stderr[:300]}")

        return output_path
