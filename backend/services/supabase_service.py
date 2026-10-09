import os
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from backend.config import (
    SUPABASE_URL,
    get_supabase_key,
    is_supabase_configured,
    SUPABASE_STORAGE_BUCKET_VIDEOS,
    SUPABASE_STORAGE_BUCKET_AUDIO,
    SUPABASE_STORAGE_BUCKET_CLIPS,
    SUPABASE_STORAGE_BUCKET_SUBTITLES,
    SUPABASE_STORAGE_BUCKET_EXPORTS,
)

_SUPABASE_CLIENT = None
_SUPABASE_ANON_CLIENT = None


def to_uuid(val: Any) -> str:
    """Deterministically maps any string or legacy ID into a valid UUID string."""
    if not val:
        return str(uuid.uuid4())
    val_str = str(val).strip()
    try:
        return str(uuid.UUID(val_str))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, val_str))


def get_supabase_anon_client():
    """Initializes and caches the Supabase client configured with the Anon public key for client auth flows."""
    global _SUPABASE_ANON_CLIENT
    if _SUPABASE_ANON_CLIENT is not None:
        return _SUPABASE_ANON_CLIENT

    if not is_supabase_configured():
        return None

    try:
        from supabase import create_client, ClientOptions
        url = os.getenv("SUPABASE_URL", "") or SUPABASE_URL
        anon_key = os.getenv("SUPABASE_ANON_KEY", "") or SUPABASE_ANON_KEY
        if not url or not anon_key:
            return None

        options = ClientOptions(postgrest_client_timeout=30, storage_client_timeout=60)
        _SUPABASE_ANON_CLIENT = create_client(url, anon_key, options=options)
        return _SUPABASE_ANON_CLIENT
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase anon client: {e}")
        return None


def get_supabase_client():
    """Initializes and caches the singleton Supabase client (service role for server persistence)."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    if not is_supabase_configured():
        return None

    try:
        from supabase import create_client, ClientOptions
        url = os.getenv("SUPABASE_URL", "") or SUPABASE_URL
        key = get_supabase_key()
        if not url or not key:
            return None

        # Configure timeout
        options = ClientOptions(postgrest_client_timeout=30, storage_client_timeout=60)
        _SUPABASE_CLIENT = create_client(url, key, options=options)
        return _SUPABASE_CLIENT
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
        return None


class SupabaseStorageService:
    """Handles private Supabase Storage operations with signed URLs."""

    @classmethod
    def ensure_buckets(cls) -> bool:
        """Ensures all required storage buckets exist in Supabase Storage."""
        client = get_supabase_client()
        if not client:
            return False

        required_buckets = [
            SUPABASE_STORAGE_BUCKET_VIDEOS,
            SUPABASE_STORAGE_BUCKET_AUDIO,
            SUPABASE_STORAGE_BUCKET_CLIPS,
            SUPABASE_STORAGE_BUCKET_SUBTITLES,
            SUPABASE_STORAGE_BUCKET_EXPORTS,
        ]

        try:
            existing = client.storage.list_buckets()
            existing_names = {b.name for b in existing} if existing else set()

            for b in required_buckets:
                if b not in existing_names:
                    try:
                        client.storage.create_bucket(b, options={"public": False})
                    except Exception as b_err:
                        # May already exist or restricted by role
                        pass
            return True
        except Exception as e:
            print(f"Storage bucket verification note: {e}")
            return False

    @classmethod
    def upload_file(
        cls,
        bucket_name: str,
        local_path: Union[str, Path],
        storage_dest_path: str,
        content_type: str = "video/mp4"
    ) -> Optional[str]:
        """Uploads a local media file to private Supabase Storage."""
        client = get_supabase_client()
        if not client:
            return None

        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Error: Local file {local_path} does not exist for upload.")
            return None

        # Supabase Storage standard tier limits individual uploads to 50MB.
        # Files larger than 50MB should be retained locally on disk to avoid hanging or 413 errors.
        file_size = local_path.stat().st_size
        if file_size > 50 * 1024 * 1024:
            print(f"Info: Skipping Supabase storage upload for {local_path.name} ({round(file_size / (1024*1024), 1)}MB > 50MB limit). Stored locally on disk.")
            return None

        try:
            with open(local_path, "rb") as f:
                file_bytes = f.read()

            clean_path = storage_dest_path.replace("\\", "/").lstrip("/")
            client.storage.from_(bucket_name).upload(
                path=clean_path,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            return clean_path
        except Exception as e:
            print(f"Failed to upload {local_path} to Supabase bucket '{bucket_name}': {e}")
            return None

    @classmethod
    def download_file(
        cls,
        bucket_name: str,
        storage_path: str,
        local_dest: Union[str, Path]
    ) -> Optional[Path]:
        """Downloads a private file from Supabase Storage to local disk."""
        client = get_supabase_client()
        if not client:
            return None

        local_dest = Path(local_dest)
        local_dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            clean_path = storage_path.replace("\\", "/").lstrip("/")
            res_bytes = client.storage.from_(bucket_name).download(clean_path)
            with open(local_dest, "wb") as f:
                f.write(res_bytes)
            return local_dest
        except Exception as e:
            print(f"Failed to download {storage_path} from Supabase bucket '{bucket_name}': {e}")
            return None

    @classmethod
    def create_signed_url(
        cls,
        bucket_name: str,
        storage_path: str,
        expires_in: int = 3600
    ) -> Optional[str]:
        """Generates a secure, short-lived signed URL for authenticated media playback."""
        client = get_supabase_client()
        if not client:
            return None

        try:
            clean_path = storage_path.replace("\\", "/").lstrip("/")
            res = client.storage.from_(bucket_name).create_signed_url(clean_path, expires_in)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signedUrl")
            return getattr(res, "signed_url", str(res))
        except Exception as e:
            print(f"Failed to create signed URL for {storage_path}: {e}")
            return None

    @classmethod
    def delete_file(cls, bucket_name: str, storage_path: str) -> bool:
        """Removes a private file from Supabase Storage."""
        client = get_supabase_client()
        if not client:
            return False
        try:
            clean_path = storage_path.replace("\\", "/").lstrip("/")
            client.storage.from_(bucket_name).remove([clean_path])
            return True
        except Exception as e:
            print(f"Failed to delete {storage_path} from Supabase bucket '{bucket_name}': {e}")
            return False


class SupabaseDbService:
    """Handles CRUD persistence operations directly against Supabase PostgreSQL."""

    @classmethod
    def save_video(
        cls,
        video_id: str,
        filename: str,
        filepath: str,
        audio_path: str = "",
        duration: float = 0.0,
        filesize: int = 0,
        user_id: Optional[str] = None,
        source_type: str = "upload",
        source_url: Optional[str] = None,
        storage_path: Optional[str] = None
    ) -> Dict[str, Any]:
        client = get_supabase_client()
        if not client:
            raise RuntimeError("Supabase client not configured")

        v_uuid = to_uuid(video_id)
        
        # Verify if user exists in auth/profiles, otherwise fallback to None for guest/legacy
        u_uuid = None
        if user_id:
            try:
                candidate_uuid = to_uuid(user_id)
                check = client.table("profiles").select("id").eq("id", candidate_uuid).execute()
                if check.data and len(check.data) > 0:
                    u_uuid = candidate_uuid
            except Exception:
                u_uuid = None

        # Build relative storage paths if not provided
        u_prefix = u_uuid or "anonymous"
        final_storage_path = storage_path or f"{u_prefix}/{v_uuid}/original.mp4"
        audio_storage_path = f"{u_prefix}/{v_uuid}/audio.mp3" if audio_path else None

        payload = {
            "id": v_uuid,
            "user_id": u_uuid,
            "title": filename,
            "source_type": source_type,
            "source_url": source_url,
            "storage_path": final_storage_path,
            "audio_storage_path": audio_storage_path,
            "duration_seconds": round(float(duration), 2),
            "filesize_bytes": int(filesize),
            "status": "uploaded",
            "updated_at": "now()"
        }

        res = client.table("videos").upsert(payload).execute()
        return res.data[0] if (res.data and len(res.data) > 0) else payload

    @classmethod
    def update_video_status(
        cls,
        video_id: str,
        status: str,
        core_thesis: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        client = get_supabase_client()
        if not client:
            return

        v_uuid = to_uuid(video_id)
        update_data = {
            "status": status,
            "updated_at": "now()"
        }
        if core_thesis is not None:
            update_data["core_thesis"] = core_thesis
        if error_message is not None:
            update_data["error_message"] = error_message

        client.table("videos").update(update_data).eq("id", v_uuid).execute()

    @classmethod
    def get_video(cls, video_id: str) -> Optional[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return None

        v_uuid = to_uuid(video_id)
        res = client.table("videos").select("*").eq("id", v_uuid).execute()
        if res.data and len(res.data) > 0:
            row = res.data[0]
            # Provide backward-compatible field names for existing routers
            row["filename"] = row.get("title", "")
            row["filepath"] = row.get("storage_path", "")
            row["duration"] = float(row.get("duration_seconds", 0.0) or 0.0)
            return row
        return None

    @classmethod
    def get_user_videos(cls, user_id: str) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return []

        u_uuid = to_uuid(user_id)
        res = client.table("videos").select("*").eq("user_id", u_uuid).order("created_at", desc=True).execute()
        results = []
        for row in (res.data or []):
            row["filename"] = row.get("title", "")
            row["filepath"] = row.get("storage_path", "")
            row["duration"] = float(row.get("duration_seconds", 0.0) or 0.0)
            results.append(row)
        return results

    @classmethod
    def save_segments(cls, video_id: str, segments: List[Dict[str, Any]]) -> None:
        client = get_supabase_client()
        if not client or not segments:
            return

        v_uuid = to_uuid(video_id)
        # Delete existing segments for this video before replacing
        try:
            client.table("transcript_segments").delete().eq("video_id", v_uuid).execute()
        except Exception:
            pass

        rows = []
        for idx, s in enumerate(segments):
            seg_index = int(s.get("segment_index", s.get("id", idx)))
            start_t = float(s.get("start", s.get("start_time", 0.0)))
            end_t = float(s.get("end", s.get("end_time", 0.0)))

            start_ms = int(round(start_t * 1000))
            end_ms = int(round(end_t * 1000))

            rows.append({
                "video_id": v_uuid,
                "segment_index": seg_index,
                "speaker_id": s.get("speaker", "SPEAKER_01"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": s.get("text", ""),
                "words_json": s.get("words", []),
                "embedding_json": s.get("embedding", []),
                "candidate_topics_json": s.get("candidate_topics", []),
                "category": s.get("category", "SUPPORTING")
            })

        # Insert in batches of 200 to avoid payload size constraints
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            client.table("transcript_segments").insert(rows[i:i + batch_size]).execute()

    @classmethod
    def get_segments(cls, video_id: str) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return []

        v_uuid = to_uuid(video_id)
        res = client.table("transcript_segments").select("*").eq("video_id", v_uuid).order("start_ms", desc=False).execute()
        results = []
        for r in (res.data or []):
            start_s = round(float(r["start_ms"]) / 1000.0, 3)
            end_s = round(float(r["end_ms"]) / 1000.0, 3)
            results.append({
                "id": r["segment_index"],
                "segment_index": r["segment_index"],
                "video_id": video_id,
                "start": start_s,
                "end": end_s,
                "start_time": start_s,
                "end_time": end_s,
                "text": r["text"],
                "speaker": r.get("speaker_id", "SPEAKER_01"),
                "words": r.get("words_json") or [],
                "embedding": r.get("embedding_json") or [],
                "candidate_topics": r.get("candidate_topics_json") or [],
                "category": r.get("category", "SUPPORTING")
            })
        return results

    @classmethod
    def save_topics(cls, video_id: str, topics: List[Dict[str, Any]]) -> None:
        client = get_supabase_client()
        if not client or not topics:
            return

        v_uuid = to_uuid(video_id)
        try:
            client.table("topics").delete().eq("video_id", v_uuid).execute()
        except Exception:
            pass

        rows = []
        for t in topics:
            raw_id = t.get("id", "")
            t_uuid = to_uuid(f"{v_uuid}_{raw_id}")
            start_t = float(t.get("start_time", 0.0))
            end_t = float(t.get("end_time", 0.0))

            metadata = {
                "legacy_id": raw_id,
                "subtopics": t.get("subtopics", []),
                "why_selected": t.get("why_selected", []),
                "segment_ids": t.get("segment_ids", []),
                "speakers_involved": t.get("speakers_involved", []),
                "speaker_roles": t.get("speaker_roles", {}),
                "question_included": bool(t.get("question_included", False)),
                "depends_on_question": bool(t.get("depends_on_question", False)),
                "exchange_type": t.get("exchange_type", "GENERAL_TOPIC"),
                "editorial_justification": t.get("editorial_justification", "")
            }

            rows.append({
                "id": t_uuid,
                "video_id": v_uuid,
                "title": t.get("name", t.get("title", "")),
                "description": t.get("description", ""),
                "importance_score": float(t.get("importance_score", 0.0)),
                "start_ms": int(round(start_t * 1000)),
                "end_ms": int(round(end_t * 1000)),
                "metadata": metadata,
                "status": t.get("status", "discovered")
            })

        client.table("topics").insert(rows).execute()

    @classmethod
    def get_topics(cls, video_id: str) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return []

        v_uuid = to_uuid(video_id)
        res = client.table("topics").select("*").eq("video_id", v_uuid).order("importance_score", desc=True).execute()
        results = []
        for r in (res.data or []):
            meta = r.get("metadata") or {}
            start_s = round(float(r["start_ms"]) / 1000.0, 2)
            end_s = round(float(r["end_ms"]) / 1000.0, 2)
            results.append({
                "id": meta.get("legacy_id") or r["id"],
                "video_id": video_id,
                "name": r["title"],
                "title": r["title"],
                "description": r.get("description", ""),
                "start_time": start_s,
                "end_time": end_s,
                "importance_score": float(r.get("importance_score", 0.0) or 0.0),
                "subtopics": meta.get("subtopics", []),
                "why_selected": meta.get("why_selected", []),
                "segment_ids": meta.get("segment_ids", []),
                "speakers_involved": meta.get("speakers_involved", []),
                "speaker_roles": meta.get("speaker_roles", {}),
                "question_included": meta.get("question_included", False),
                "depends_on_question": meta.get("depends_on_question", False),
                "exchange_type": meta.get("exchange_type", "GENERAL_TOPIC"),
                "editorial_justification": meta.get("editorial_justification", ""),
                "status": r.get("status", "discovered")
            })
        return results

    @classmethod
    def save_clips(cls, video_id: str, clips: List[Dict[str, Any]]) -> None:
        client = get_supabase_client()
        if not client or not clips:
            return

        v_uuid = to_uuid(video_id)
        try:
            client.table("clips").delete().eq("video_id", v_uuid).execute()
        except Exception:
            pass

        rows = []
        for c in clips:
            raw_id = c.get("id", "")
            c_uuid = to_uuid(f"{v_uuid}_{raw_id}")
            start_t = float(c.get("start_time", 0.0))
            end_t = float(c.get("end_time", 0.0))
            dur = float(c.get("duration", max(0.1, end_t - start_t)))

            meta = {
                "legacy_id": raw_id,
                "topic_id": c.get("topic_id"),
                "clip_index": c.get("clip_index", 1),
                "filename": c.get("filename", ""),
                "filepath": c.get("filepath", ""),
                "text": c.get("text", ""),
                "speakers_involved": c.get("speakers_involved", []),
                "speaker_roles": c.get("speaker_roles", {}),
                "exchange_type": c.get("exchange_type", "GENERAL_TOPIC"),
                "question_included": bool(c.get("question_included", False)),
                "depends_on_question": bool(c.get("depends_on_question", False)),
                "editorial_justification": c.get("editorial_justification", ""),
                "reason": c.get("reason", "")
            }

            rows.append({
                "id": c_uuid,
                "video_id": v_uuid,
                "topic_id": to_uuid(f"{v_uuid}_{c.get('topic_id')}") if c.get("topic_id") else None,
                "title": c.get("text", c.get("filename", f"Clip {c.get('clip_index', 1)}")),
                "start_ms": int(round(start_t * 1000)),
                "end_ms": int(round(end_t * 1000)),
                "duration_seconds": round(dur, 2),
                "storage_path": c.get("storage_path") or c.get("filepath", ""),
                "subtitle_storage_path": c.get("subtitle_storage_path"),
                "quality_score": float(c.get("importance_score", 0.9)),
                "status": "ready",
                "metadata": meta,
                "updated_at": "now()"
            })

        client.table("clips").insert(rows).execute()

    @classmethod
    def get_clips(cls, video_id: str) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return []

        v_uuid = to_uuid(video_id)
        res = client.table("clips").select("*").eq("video_id", v_uuid).order("created_at", desc=False).execute()
        results = []
        for r in (res.data or []):
            meta = r.get("metadata") or {}
            start_s = round(float(r["start_ms"]) / 1000.0, 2)
            end_s = round(float(r["end_ms"]) / 1000.0, 2)
            dur = float(r.get("duration_seconds", 0.0) or (end_s - start_s))
            results.append({
                "id": meta.get("legacy_id") or r["id"],
                "video_id": video_id,
                "topic_id": meta.get("topic_id"),
                "clip_index": meta.get("clip_index", 1),
                "filename": meta.get("filename", ""),
                "filepath": meta.get("filepath", r.get("storage_path", "")),
                "storage_path": r.get("storage_path"),
                "subtitle_storage_path": r.get("subtitle_storage_path"),
                "start_time": start_s,
                "end_time": end_s,
                "duration": dur,
                "text": r.get("title", meta.get("text", "")),
                "importance_score": float(r.get("quality_score", 0.9) or 0.9),
                "is_selected": True,
                "download_url": f"/api/videos/{video_id}/clip/{meta.get('topic_id')}",
                "video_url": f"/api/videos/{video_id}/clip/{meta.get('topic_id')}",
                "subtitle_url": f"/api/videos/{video_id}/clip/{meta.get('topic_id')}/subtitles",
                "speakers_involved": meta.get("speakers_involved", []),
                "speaker_roles": meta.get("speaker_roles", {}),
                "exchange_type": meta.get("exchange_type", "GENERAL_TOPIC"),
                "editorial_justification": meta.get("editorial_justification", ""),
                "reason": meta.get("reason", "")
            })
        return results

    @classmethod
    def save_clip_selection(
        cls,
        user_id: str,
        video_id: str,
        clip_id: str,
        position: int,
        selected: bool = True
    ) -> Dict[str, Any]:
        """Persists user clip selection and ordering in clip_selections table."""
        client = get_supabase_client()
        if not client:
            return {}

        u_uuid = to_uuid(user_id)
        v_uuid = to_uuid(video_id)
        c_uuid = to_uuid(f"{v_uuid}_{clip_id}")

        payload = {
            "user_id": u_uuid,
            "video_id": v_uuid,
            "clip_id": c_uuid,
            "position": position,
            "selected": selected,
            "updated_at": "now()"
        }

        res = client.table("clip_selections").upsert(
            payload,
            on_conflict="user_id,video_id,clip_id"
        ).execute()
        return res.data[0] if (res.data and len(res.data) > 0) else payload

    @classmethod
    def get_clip_selections(cls, user_id: str, video_id: str) -> List[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return []

        u_uuid = to_uuid(user_id)
        v_uuid = to_uuid(video_id)
        res = client.table("clip_selections").select("*").eq("user_id", u_uuid).eq("video_id", v_uuid).order("position", desc=False).execute()
        return res.data or []

    @classmethod
    def set_job(
        cls,
        job_id: str,
        video_id: str,
        job_type: str,
        status: str,
        progress: int,
        stage: str,
        message: str,
        result: Any = None,
        error: str = None,
        user_id: Optional[str] = None
    ) -> None:
        client = get_supabase_client()
        if not client:
            return

        j_uuid = to_uuid(job_id)
        v_uuid = to_uuid(video_id)
        u_uuid = to_uuid(user_id) if user_id else None

        res_data = result if isinstance(result, dict) else ({"data": result} if result is not None else {})
        res_data["job_type"] = job_type
        res_data["progress"] = progress
        res_data["message"] = message

        payload = {
            "id": j_uuid,
            "video_id": v_uuid,
            "user_id": u_uuid,
            "stage": stage,
            "status": status,
            "error_message": error,
            "result_json": res_data,
            "updated_at": "now()"
        }

        if status == "running":
            payload["started_at"] = "now()"
        elif status in ("completed", "failed"):
            payload["completed_at"] = "now()"

        client.table("processing_jobs").upsert(payload).execute()

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        client = get_supabase_client()
        if not client:
            return None

        j_uuid = to_uuid(job_id)
        res = client.table("processing_jobs").select("*").eq("id", j_uuid).execute()
        if not res.data or len(res.data) == 0:
            return None

        r = res.data[0]
        res_json = r.get("result_json") or {}
        return {
            "id": job_id,
            "job_id": job_id,
            "video_id": str(r.get("video_id")),
            "user_id": str(r.get("user_id")) if r.get("user_id") else None,
            "stage": r.get("stage"),
            "current_stage": r.get("stage"),
            "status": r.get("status"),
            "progress": res_json.get("progress", 0),
            "message": res_json.get("message", ""),
            "result": res_json,
            "error": r.get("error_message")
        }

    @classmethod
    def claim_video_ownership(cls, video_id: str, user_id: str) -> bool:
        """Assigns an unowned legacy video record to the authenticated user."""
        client = get_supabase_client()
        if not client:
            return False
        v_uuid = to_uuid(video_id)
        u_uuid = to_uuid(user_id)
        try:
            client.table("videos").update({
                "user_id": u_uuid,
                "updated_at": "now()"
            }).eq("id", v_uuid).is_("user_id", "null").execute()
            return True
        except Exception as e:
            print(f"Supabase claim_video_ownership error: {e}")
            return False

    @classmethod
    def delete_video(cls, video_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes video and cascades to all child records in Supabase."""
        client = get_supabase_client()
        if not client:
            return False
        v_uuid = to_uuid(video_id)
        try:
            query = client.table("videos").delete().eq("id", v_uuid)
            if user_id:
                u_uuid = to_uuid(user_id)
                query = query.eq("user_id", u_uuid)
            query.execute()
            return True
        except Exception as e:
            print(f"Supabase delete_video error: {e}")
            return False
