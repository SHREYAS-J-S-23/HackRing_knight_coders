import asyncio
import subprocess
from pathlib import Path
from backend.config import get_ffmpeg_executable, AUDIO_DIR

class AudioExtractor:
    @staticmethod
    def extract_audio(video_path: Path, output_filename: str = None) -> Path:
        """
        Extracts 16kHz mono audio from video file using FFmpeg.
        Returns the path to the extracted .wav or .mp3 file.
        """
        ffmpeg_bin = get_ffmpeg_executable()
        video_path = Path(video_path).resolve()
        if output_filename is None:
            output_filename = f"{video_path.stem}.mp3"
        
        output_path = (AUDIO_DIR / output_filename).resolve()

        # Command to extract 16kHz audio for optimal Whisper processing
        # 32kbps keeps 1 hour of audio at ~14MB (well below Whisper limits)
        cmd = [
            ffmpeg_bin,
            "-y",                # Overwrite output if exists
            "-i", str(video_path),
            "-vn",               # Disable video recording
            "-acodec", "libmp3lame",
            "-ar", "16000",      # 16kHz sampling rate
            "-ac", "1",          # Mono channel
            "-b:a", "32k",       # 32kbps keeps 1 hour at ~14MB for fast transfer
            str(output_path)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            if "does not contain any stream" in result.stderr or "Output file does not contain any stream" in result.stderr:
                # Video has no audio track: synthesize silent audio so the pipeline handles it smoothly
                duration = AudioExtractor.get_video_duration(video_path) or 2.0
                cmd_silent = [
                    ffmpeg_bin,
                    "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=16000:cl=mono",
                    "-t", str(max(1.0, duration)),
                    "-acodec", "libmp3lame",
                    "-b:a", "32k",
                    str(output_path)
                ]
                res_silent = subprocess.run(cmd_silent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res_silent.returncode == 0 and output_path.exists():
                    return output_path
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        
        return output_path

    @staticmethod
    def split_audio(
        audio_path: Path,
        chunk_duration: float = 300.0,
        overlap: float = 2.0
    ) -> list:
        """
        Splits a long audio file into smaller chunks with boundary overlap
        to guarantee file sizes stay under Groq Whisper's 25MB limit and prevent
        clipped words at chunk boundaries.
        Returns a list of tuples: [(chunk_path, start_offset_seconds), ...]
        """
        ffmpeg_bin = get_ffmpeg_executable()
        total_duration = AudioExtractor.get_audio_duration(audio_path)
        
        # If already short and under 20MB, no need to split
        file_size_mb = audio_path.stat().st_size / (1024 * 1024) if audio_path.exists() else 0.0
        if total_duration <= chunk_duration and file_size_mb < 20.0:
            return [(audio_path, 0.0)]

        chunks = []
        chunk_idx = 0
        current_start = 0.0

        while current_start < total_duration:
            # Add overlap to duration_to_cut, bounded by total_duration
            duration_to_cut = min(chunk_duration + overlap, total_duration - current_start)
            chunk_file = audio_path.parent / f"{audio_path.stem}_chunk_{chunk_idx}.mp3"

            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", f"{current_start:.3f}",
                "-i", str(audio_path),
                "-t", f"{duration_to_cut:.3f}",
                "-acodec", "copy",
                str(chunk_file)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and chunk_file.exists():
                chunks.append((chunk_file, current_start))
            else:
                # If copy fails, re-encode chunk
                cmd_reencode = [
                    ffmpeg_bin,
                    "-y",
                    "-ss", f"{current_start:.3f}",
                    "-i", str(audio_path),
                    "-t", f"{duration_to_cut:.3f}",
                    "-acodec", "libmp3lame",
                    "-ar", "16000",
                    "-ac", "1",
                    "-b:a", "32k",
                    str(chunk_file)
                ]
                subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if chunk_file.exists():
                    chunks.append((chunk_file, current_start))

            current_start += chunk_duration
            chunk_idx += 1

        return chunks if chunks else [(audio_path, 0.0)]

    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        """Extracts exact audio duration in seconds via FFmpeg."""
        return AudioExtractor.get_video_duration(audio_path)

    @staticmethod
    def get_video_duration(video_path: Path) -> float:
        """Extracts exact video duration in seconds via FFmpeg/ffprobe."""
        ffmpeg_bin = get_ffmpeg_executable()
        video_path = Path(video_path).resolve()
        cmd = [
            ffmpeg_bin,
            "-i", str(video_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # FFmpeg outputs duration in stderr e.g.: "Duration: 00:01:23.45"
        for line in result.stderr.splitlines():
            if "Duration:" in line:
                try:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    parts = time_str.split(":")
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                except Exception:
                    pass
        return 0.0

    @staticmethod
    def convert_to_mp4(input_path: Path, output_path: Path) -> Path:
        """Transcodes/remuxes any media container (WebM, MKV, AVI) into standard MP4 with ultrafast encoding."""
        ffmpeg_bin = get_ffmpeg_executable()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-threads", "0",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg MP4 conversion failed: {res.stderr}")
        return output_path
