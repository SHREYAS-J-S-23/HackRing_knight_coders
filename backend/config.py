import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
load_dotenv(BASE_DIR / ".env", override=True)

# Dual AI Pipeline API Keys
_RUNTIME_VIDEO_ANALYSIS_KEY = os.getenv("VIDEO_ANALYSIS_API_KEY", "") or os.getenv("GROQ_STT_KEY", "") or os.getenv("GROQ_API_KEY", "")
_RUNTIME_REASONING_KEY = os.getenv("REASONING_API_KEY", "") or os.getenv("GROQ_LLM_KEY", "") or os.getenv("GROQ_API_KEY", "")
_RUNTIME_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

def get_video_analysis_api_key() -> str:
    """Returns dedicated video/audio analysis key (STT, timestamps, segmentation)."""
    global _RUNTIME_VIDEO_ANALYSIS_KEY
    load_dotenv(BASE_DIR / ".env", override=True)
    key = (
        os.getenv("VIDEO_ANALYSIS_API_KEY", "") or
        os.getenv("GROQ_STT_KEY", "") or
        os.getenv("GROQ_API_KEY", "")
    )
    if key:
        _RUNTIME_VIDEO_ANALYSIS_KEY = key
    return _RUNTIME_VIDEO_ANALYSIS_KEY or key

def get_reasoning_api_key() -> str:
    """Returns dedicated reasoning key (query understanding, topic ranking, EDL decisions)."""
    global _RUNTIME_REASONING_KEY
    load_dotenv(BASE_DIR / ".env", override=True)
    key = (
        os.getenv("REASONING_API_KEY", "") or
        os.getenv("GROQ_LLM_KEY", "") or
        os.getenv("GROQ_API_KEY", "")
    )
    if key:
        _RUNTIME_REASONING_KEY = key
    return _RUNTIME_REASONING_KEY or key

def get_video_analysis_model() -> str:
    return os.getenv("VIDEO_ANALYSIS_MODEL", "") or os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

def get_reasoning_model() -> str:
    return os.getenv("REASONING_MODEL", "") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Backwards compatible aliases
get_groq_stt_api_key = get_video_analysis_api_key
get_groq_llm_api_key = get_reasoning_api_key

def get_groq_api_key() -> str:
    global _RUNTIME_GROQ_KEY
    if _RUNTIME_GROQ_KEY:
        return _RUNTIME_GROQ_KEY
    load_dotenv(BASE_DIR / ".env", override=True)
    _RUNTIME_GROQ_KEY = os.getenv("GROQ_API_KEY", "") or get_video_analysis_api_key()
    return _RUNTIME_GROQ_KEY

def set_groq_api_key(key: str, stt_key: str = "", llm_key: str = ""):
    global _RUNTIME_GROQ_KEY, _RUNTIME_VIDEO_ANALYSIS_KEY, _RUNTIME_REASONING_KEY
    if key:
        _RUNTIME_GROQ_KEY = key.strip()
        os.environ["GROQ_API_KEY"] = _RUNTIME_GROQ_KEY
    if stt_key:
        _RUNTIME_VIDEO_ANALYSIS_KEY = stt_key.strip()
        os.environ["VIDEO_ANALYSIS_API_KEY"] = _RUNTIME_VIDEO_ANALYSIS_KEY
        os.environ["GROQ_STT_KEY"] = _RUNTIME_VIDEO_ANALYSIS_KEY
    if llm_key:
        _RUNTIME_REASONING_KEY = llm_key.strip()
        os.environ["REASONING_API_KEY"] = _RUNTIME_REASONING_KEY
        os.environ["GROQ_LLM_KEY"] = _RUNTIME_REASONING_KEY

    # Persist to .env file
    env_file = BASE_DIR / ".env"
    lines = []
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    key_dict = {
        "GROQ_API_KEY": _RUNTIME_GROQ_KEY,
        "VIDEO_ANALYSIS_API_KEY": _RUNTIME_VIDEO_ANALYSIS_KEY,
        "REASONING_API_KEY": _RUNTIME_REASONING_KEY,
        "GROQ_STT_KEY": _RUNTIME_VIDEO_ANALYSIS_KEY,
        "GROQ_LLM_KEY": _RUNTIME_REASONING_KEY
    }
    
    new_lines = []
    handled_keys = set()
    for line in lines:
        matched = False
        for k, v in key_dict.items():
            if line.startswith(f"{k}="):
                new_lines.append(f"{k}={v}\n")
                handled_keys.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in key_dict.items():
        if k not in handled_keys and v:
            new_lines.insert(0, f"{k}={v}\n")

    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Performance Optimization Settings
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "FAST").upper()  # FAST or PRECISE
GROQ_WHISPER_FAST_MODEL = os.getenv("GROQ_WHISPER_FAST_MODEL", "whisper-large-v3-turbo")
GROQ_WHISPER_PRECISE_MODEL = os.getenv("GROQ_WHISPER_PRECISE_MODEL", "whisper-large-v3")
TRANSCRIPTION_CHUNK_SECONDS = float(os.getenv("TRANSCRIPTION_CHUNK_SECONDS", "300.0"))
TRANSCRIPTION_OVERLAP_SECONDS = float(os.getenv("TRANSCRIPTION_OVERLAP_SECONDS", "2.0"))
TRANSCRIPTION_MAX_CONCURRENCY = int(os.getenv("TRANSCRIPTION_MAX_CONCURRENCY", "2"))
TRANSCRIPTION_MAX_RETRIES = int(os.getenv("TRANSCRIPTION_MAX_RETRIES", "3"))
TRANSCRIPTION_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "60.0"))

DIARIZATION_MODE = os.getenv("DIARIZATION_MODE", "FAST").upper()  # FAST, STANDARD, PRECISE
REASONING_TIMEOUT_SECONDS = float(os.getenv("REASONING_TIMEOUT_SECONDS", "25.0"))

AGNES_API_KEY = os.getenv("AGNES_API_KEY", "")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://platform.agnes-ai.com/v1")

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

# Storage directories
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
AUDIO_DIR = BASE_DIR / os.getenv("AUDIO_DIR", "storage/audio")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "storage/outputs")

for directory in [UPLOAD_DIR, AUDIO_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_ffmpeg_executable() -> str:
    """Finds system ffmpeg, Winget installed ffmpeg, or imageio-ffmpeg binary."""
    ffmpeg_sys = shutil.which("ffmpeg")
    if ffmpeg_sys:
        return ffmpeg_sys
    
    # Check WinGet installation directory
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        winget_path = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            ffmpeg_matches = list(winget_path.glob("**/ffmpeg.exe"))
            if ffmpeg_matches:
                return str(ffmpeg_matches[0])

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ====================================================
# SUPABASE CONFIGURATION & STORAGE BUCKETS
# ====================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = (os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_KEY", "")).strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip() or os.getenv("DATABASE_URL", "").strip()

# Storage bucket names (private buckets)
SUPABASE_STORAGE_BUCKET_VIDEOS = os.getenv("SUPABASE_STORAGE_BUCKET_VIDEOS", "original-videos")
SUPABASE_STORAGE_BUCKET_AUDIO = os.getenv("SUPABASE_STORAGE_BUCKET_AUDIO", "processed-audio")
SUPABASE_STORAGE_BUCKET_CLIPS = os.getenv("SUPABASE_STORAGE_BUCKET_CLIPS", "generated-clips")
SUPABASE_STORAGE_BUCKET_SUBTITLES = os.getenv("SUPABASE_STORAGE_BUCKET_SUBTITLES", "subtitles")
SUPABASE_STORAGE_BUCKET_EXPORTS = os.getenv("SUPABASE_STORAGE_BUCKET_EXPORTS", "final-exports")

def is_supabase_configured() -> bool:
    """Returns True if Supabase credentials are configured."""
    url = os.getenv("SUPABASE_URL", "") or SUPABASE_URL
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_KEY", "") or SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    return bool(url and key)

def get_supabase_key() -> str:
    """Returns the most privileged available key for server-side persistence operations."""
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or SUPABASE_SERVICE_ROLE_KEY or os.getenv("SUPABASE_ANON_KEY", "") or SUPABASE_ANON_KEY or os.getenv("SUPABASE_KEY", "")

