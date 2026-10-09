import os
import re
import uuid
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import yt_dlp

from backend.config import UPLOAD_DIR
from backend.services.audio_extractor import AudioExtractor
from backend.database import DatabaseService

class LinkDownloader:
    """
    Safely and deterministically downloads videos from:
    - Platforms: YouTube, Vimeo, Loom, Twitter/X, Dailymotion, Instagram, etc.
    - Direct HTTP/HTTPS video files: .mp4, .mov, .webm, .mkv, cloud storage URLs.
    Extracts 16kHz audio with FFmpeg and saves metadata to SQLite.
    """

    @staticmethod
    def is_valid_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        clean = url.strip()
        try:
            parsed = urllib.parse.urlparse(clean)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    @classmethod
    def download_video_from_url(cls, url: str, user_id: Optional[str] = None, audience_mode: str = "education") -> Dict[str, Any]:
        clean_url = url.strip()
        if not cls.is_valid_url(clean_url):
            raise ValueError("Invalid URL format. Please provide a valid http:// or https:// video link.")

        video_id = str(uuid.uuid4())[:8]
        output_template = str(UPLOAD_DIR / f"{video_id}.%(ext)s")
        target_mp4_path = UPLOAD_DIR / f"{video_id}.mp4"

        # Check if direct mp4/webm/mov/mkv link
        parsed_path = urllib.parse.urlparse(clean_url).path.lower()
        is_direct_stream = any(parsed_path.endswith(ext) for ext in [".mp4", ".mov", ".webm", ".mkv", ".m4v"])

        title = "Online Video"
        downloaded_path: Optional[Path] = None
        duration = 0.0

        # Try yt-dlp first for maximum platform compatibility and format selection
        if not is_direct_stream:
            from backend.config import get_ffmpeg_executable
            ffmpeg_path = get_ffmpeg_executable()

            ydl_opts = {
                'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/bestvideo[height<=720]+bestaudio/best[height<=720]/best',
                'outtmpl': output_template,
                'merge_output_format': 'mp4',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'retries': 3,
                'prefer_ffmpeg': True,
                'ffmpeg_location': ffmpeg_path,
                'concurrent_fragment_downloads': 5,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(clean_url, download=True)
                    if info:
                        raw_title = info.get('title')
                        if raw_title:
                            title = re.sub(r'[^\w\s\-\.]', '', raw_title)[:100].strip() or "Online Video"
                        duration = float(info.get('duration') or 0.0)

                        if target_mp4_path.exists():
                            downloaded_path = target_mp4_path
                        else:
                            matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
                            valid_matches = [m for m in matches if m.suffix.lower() in [".mp4", ".webm", ".mkv", ".mov", ".m4v"]]
                            if valid_matches:
                                downloaded_path = valid_matches[0]
            except Exception as ydl_err:
                print(f"Vidara yt-dlp primary format note: {ydl_err}. Retrying with fallback format 'best'...")
                try:
                    ydl_opts_fallback = dict(ydl_opts)
                    ydl_opts_fallback['format'] = 'best[ext=mp4]/best'
                    with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                        info = ydl.extract_info(clean_url, download=True)
                        if info:
                            raw_title = info.get('title')
                            if raw_title:
                                title = re.sub(r'[^\w\s\-\.]', '', raw_title)[:100].strip() or "Online Video"
                            duration = float(info.get('duration') or 0.0)
                            if target_mp4_path.exists():
                                downloaded_path = target_mp4_path
                            else:
                                matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
                                valid_matches = [m for m in matches if m.suffix.lower() in [".mp4", ".webm", ".mkv", ".mov", ".m4v"]]
                                if valid_matches:
                                    downloaded_path = valid_matches[0]
                except Exception as fallback_err:
                    print(f"Vidara yt-dlp fallback error: {fallback_err}")

        # If it's a direct video link (.mp4, .webm, etc.) and yt-dlp did not produce a file, use streaming download
        if (not downloaded_path or not downloaded_path.exists()) and is_direct_stream:
            downloaded_path = cls._stream_download_direct(clean_url, target_mp4_path)
            url_name = Path(urllib.parse.urlparse(clean_url).path).stem
            if url_name and len(url_name) > 2:
                title = url_name

        if not downloaded_path or not downloaded_path.exists():
            raise RuntimeError(
                "Could not download video from this URL. Please verify that the link is public, accessible, and not behind a private login."
            )

        # Standardize container and ensure H.264 video codec for 100% browser HTML5 playback
        final_video_path = target_mp4_path
        from backend.config import get_ffmpeg_executable
        import subprocess

        is_h264 = False
        if downloaded_path == target_mp4_path:
            try:
                probe_cmd = [get_ffmpeg_executable(), "-i", str(downloaded_path)]
                probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if "Video: h264" in probe_res.stderr or "Video: avc1" in probe_res.stderr:
                    is_h264 = True
            except Exception:
                pass

        if not is_h264:
            temp_trans = UPLOAD_DIR / f"{video_id}_temp_input{downloaded_path.suffix}"
            if downloaded_path.exists() and downloaded_path != temp_trans:
                downloaded_path.rename(temp_trans)
            # Try ultra-fast lossless stream copy remux into MP4 first (< 0.5s)
            remux_cmd = [
                get_ffmpeg_executable(), "-y", "-threads", "0",
                "-i", str(temp_trans),
                "-c", "copy",
                "-movflags", "+faststart",
                str(final_video_path)
            ]
            remux_res = subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # If fast copy remux succeeded and created valid MP4, skip slow CPU re-encoding
            if remux_res.returncode == 0 and final_video_path.exists() and final_video_path.stat().st_size > 1000:
                pass
            else:
                AudioExtractor.convert_to_mp4(temp_trans, final_video_path)
            try:
                temp_trans.unlink(missing_ok=True)
            except Exception:
                pass

        # Extract audio using FFmpeg (16kHz mono, 32kbps for fast Whisper inference)
        audio_path = AudioExtractor.extract_audio(final_video_path, f"{video_id}.mp3")

        # Extract true duration via FFmpeg if not provided by metadata
        if duration <= 0:
            duration = AudioExtractor.get_video_duration(final_video_path)

        filesize = final_video_path.stat().st_size

        # Save to database
        display_name = f"{title}.mp4" if not title.lower().endswith(".mp4") else title
        DatabaseService.save_video(
            video_id=video_id,
            filename=display_name,
            filepath=str(final_video_path),
            audio_path=str(audio_path),
            duration=duration,
            filesize=filesize,
            user_id=user_id,
            audience_mode=audience_mode
        )

        return {
            "video_id": video_id,
            "filename": display_name,
            "filepath": str(final_video_path),
            "audio_path": str(audio_path),
            "duration_seconds": duration,
            "filesize": filesize,
            "source_url": clean_url
        }

    @staticmethod
    def _stream_download_direct(url: str, dest_path: Path) -> Path:
        """Streams direct HTTP/HTTPS video in chunks with desktop User-Agent."""
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=1024 * 1024)
        return dest_path
