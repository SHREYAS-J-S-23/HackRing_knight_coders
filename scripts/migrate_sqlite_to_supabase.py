#!/usr/bin/env python3
"""
VIDARA AI: SQLite to Supabase PostgreSQL & Storage Migration Utility
====================================================================
Safely extracts existing videos, transcripts, topics, clips, and processing
jobs from the local SQLite database (storage/vidara.db) and migrates them to
Supabase PostgreSQL and private Supabase Storage buckets.
"""

import sys
import os
import argparse
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    BASE_DIR,
    is_supabase_configured,
    SUPABASE_URL,
    SUPABASE_STORAGE_BUCKET_VIDEOS,
    SUPABASE_STORAGE_BUCKET_AUDIO,
    SUPABASE_STORAGE_BUCKET_CLIPS,
    SUPABASE_STORAGE_BUCKET_SUBTITLES,
    SUPABASE_STORAGE_BUCKET_EXPORTS,
)
from backend.services.supabase_service import (
    get_supabase_client,
    SupabaseStorageService,
    to_uuid
)

SQLITE_DB_PATH = BASE_DIR / "storage" / "vidara.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate Vidara AI from SQLite to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and simulate migration without writing to Supabase")
    parser.add_argument("--upload-media", action="store_true", help="Upload large media files to Supabase Storage (may take several minutes)")
    return parser.parse_args()


def get_sqlite_conn() -> sqlite3.Connection:
    if not SQLITE_DB_PATH.exists():
        alt_path = BASE_DIR / "vidara.db"
        if alt_path.exists():
            conn = sqlite3.connect(f"file:{alt_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        raise FileNotFoundError(f"SQLite database not found at {SQLITE_DB_PATH}")

    conn = sqlite3.connect(f"file:{SQLITE_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_migration(dry_run: bool = False, upload_media: bool = False):
    print("=" * 70, flush=True)
    print("  VIDARA AI — SQLite to Supabase PostgreSQL Migration", flush=True)
    print("=" * 70, flush=True)
    print(f"Mode: {'DRY RUN (Simulation)' if dry_run else 'LIVE MIGRATION'}", flush=True)
    print(f"Source Database: {SQLITE_DB_PATH}", flush=True)
    print(f"Target Supabase: {SUPABASE_URL or 'NOT CONFIGURED'}", flush=True)
    print(f"Media Uploads:   {'ENABLED' if upload_media else 'METADATA ONLY (use --upload-media for video binaries)'}\n", flush=True)

    if not is_supabase_configured() and not dry_run:
        print("[ERROR] Supabase credentials are not configured in .env!", flush=True)
        print("Please set the following environment variables in .env:", flush=True)
        print("  - SUPABASE_URL=https://your-project.supabase.co", flush=True)
        print("  - SUPABASE_SERVICE_ROLE_KEY=your_service_role_key", flush=True)
        print("\nOr run with --dry-run to inspect existing SQLite records.\n", flush=True)
        sys.exit(1)

    client = None
    if not dry_run:
        client = get_supabase_client()
        if not client:
            print("[ERROR] Failed to connect to Supabase. Check your SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.", flush=True)
            sys.exit(1)
        # Ensure buckets exist
        SupabaseStorageService.ensure_buckets()

    try:
        conn = get_sqlite_conn()
    except Exception as e:
        print(f"[ERROR] Could not open SQLite database: {e}", flush=True)
        sys.exit(1)

    cursor = conn.cursor()

    stats = {
        "videos_found": 0,
        "videos_migrated": 0,
        "segments_found": 0,
        "segments_migrated": 0,
        "topics_found": 0,
        "topics_migrated": 0,
        "clips_found": 0,
        "clips_migrated": 0,
        "files_uploaded": 0,
        "errors": []
    }

    # 1. Migrate Videos
    print("[1/5] Migrating Videos...", flush=True)
    try:
        video_rows = cursor.execute("SELECT * FROM videos").fetchall()
    except sqlite3.OperationalError:
        video_rows = []

    stats["videos_found"] = len(video_rows)
    print(f"  Found {len(video_rows)} videos in SQLite.", flush=True)

    migrated_video_ids = set()
    for idx, v in enumerate(video_rows, 1):
        vid_dict = dict(v)
        legacy_id = vid_dict["id"]
        v_uuid = to_uuid(legacy_id)
        user_id = vid_dict.get("user_id")
        
        # Verify if user exists in auth/profiles, otherwise fallback to None for guest/legacy
        u_uuid = None
        if user_id and client:
            try:
                candidate_uuid = to_uuid(user_id)
                check = client.table("profiles").select("id").eq("id", candidate_uuid).execute()
                if check.data and len(check.data) > 0:
                    u_uuid = candidate_uuid
            except Exception:
                u_uuid = None

        u_prefix = u_uuid or "anonymous"

        video_path_local = Path(vid_dict.get("filepath", ""))
        audio_path_local = Path(vid_dict.get("audio_path", ""))

        storage_path = f"{u_prefix}/{v_uuid}/original{video_path_local.suffix or '.mp4'}"
        audio_storage_path = f"{u_prefix}/{v_uuid}/audio.mp3" if vid_dict.get("audio_path") else None

        # Storage upload if requested and file actually exists
        if not dry_run and upload_media:
            if video_path_local.exists():
                print(f"    Uploading video {video_path_local.name} ({round(video_path_local.stat().st_size / 1024 / 1024, 1)} MB)...", flush=True)
                upload_res = SupabaseStorageService.upload_file(
                    SUPABASE_STORAGE_BUCKET_VIDEOS,
                    video_path_local,
                    storage_path,
                    "video/mp4"
                )
                if upload_res:
                    stats["files_uploaded"] += 1
            if audio_path_local.exists():
                upload_res = SupabaseStorageService.upload_file(
                    SUPABASE_STORAGE_BUCKET_AUDIO,
                    audio_path_local,
                    audio_storage_path,
                    "audio/mpeg"
                )
                if upload_res:
                    stats["files_uploaded"] += 1

        payload = {
            "id": v_uuid,
            "user_id": u_uuid,
            "title": vid_dict.get("filename", f"Video {legacy_id}"),
            "source_type": "upload",
            "storage_path": storage_path,
            "audio_storage_path": audio_storage_path,
            "duration_seconds": round(float(vid_dict.get("duration", 0.0) or 0.0), 2),
            "filesize_bytes": int(vid_dict.get("filesize", 0) or 0),
            "status": vid_dict.get("status", "uploaded"),
            "core_thesis": vid_dict.get("core_thesis"),
            "language": vid_dict.get("language", "en")
        }

        if not dry_run:
            try:
                client.table("videos").upsert(payload).execute()
                stats["videos_migrated"] += 1
                migrated_video_ids.add(v_uuid)
            except Exception as e:
                stats["errors"].append(f"Failed to migrate video {legacy_id}: {e}")
        else:
            stats["videos_migrated"] += 1
            migrated_video_ids.add(v_uuid)

    print(f"  --> {stats['videos_migrated']}/{stats['videos_found']} videos successfully upserted.", flush=True)

    # 2. Migrate Transcript Segments
    print("\n[2/5] Migrating Transcript Segments...", flush=True)
    try:
        segment_rows = cursor.execute("SELECT * FROM transcript_segments ORDER BY video_id, start_time ASC").fetchall()
    except sqlite3.OperationalError:
        segment_rows = []

    stats["segments_found"] = len(segment_rows)
    print(f"  Found {len(segment_rows)} transcript segments. Batching...", flush=True)

    batch_segments = []
    for s in segment_rows:
        s_dict = dict(s)
        v_uuid = to_uuid(s_dict["video_id"])
        if v_uuid not in migrated_video_ids:
            continue
        start_ms = int(round(float(s_dict.get("start_time", 0.0)) * 1000))
        end_ms = int(round(float(s_dict.get("end_time", 0.0)) * 1000))

        words = []
        if s_dict.get("words_json"):
            try:
                words = json.loads(s_dict["words_json"])
            except Exception:
                pass

        emb = []
        if s_dict.get("embedding_json"):
            try:
                emb = json.loads(s_dict["embedding_json"])
            except Exception:
                pass

        cand_topics = []
        if s_dict.get("candidate_topics_json"):
            try:
                cand_topics = json.loads(s_dict["candidate_topics_json"])
            except Exception:
                pass

        seg_record = {
            "video_id": v_uuid,
            "segment_index": int(s_dict.get("segment_index", s_dict.get("id", 0))),
            "speaker_id": s_dict.get("speaker", "SPEAKER_01"),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": s_dict.get("text", ""),
            "words_json": words,
            "embedding_json": emb,
            "candidate_topics_json": cand_topics,
            "category": s_dict.get("category", "SUPPORTING")
        }
        batch_segments.append(seg_record)

    if not dry_run and batch_segments:
        total_batches = (len(batch_segments) + 199) // 200
        for b_idx, i in enumerate(range(0, len(batch_segments), 200), 1):
            chunk = batch_segments[i:i + 200]
            try:
                client.table("transcript_segments").upsert(chunk, on_conflict="video_id,segment_index").execute()
                stats["segments_migrated"] += len(chunk)
                if b_idx % 10 == 0 or b_idx == total_batches:
                    print(f"    Progress: {stats['segments_migrated']}/{len(batch_segments)} segments upserted ({round(stats['segments_migrated']/len(batch_segments)*100)}%)...", flush=True)
            except Exception as e:
                stats["errors"].append(f"Failed to upsert segment chunk: {e}")
    else:
        stats["segments_migrated"] = len(batch_segments)

    print(f"  --> {stats['segments_migrated']}/{stats['segments_found']} segments successfully upserted.", flush=True)

    # 3. Migrate Topics
    print("\n[3/5] Migrating Topics...", flush=True)
    try:
        topic_rows = cursor.execute("SELECT * FROM topics").fetchall()
    except sqlite3.OperationalError:
        topic_rows = []

    stats["topics_found"] = len(topic_rows)
    print(f"  Found {len(topic_rows)} topics.", flush=True)

    migrated_topic_ids = set()
    topic_records = []
    for t in topic_rows:
        t_dict = dict(t)
        raw_id = t_dict["id"]
        v_uuid = to_uuid(t_dict["video_id"])
        if v_uuid not in migrated_video_ids:
            continue
        t_uuid = to_uuid(f"{v_uuid}_{raw_id}")

        start_ms = int(round(float(t_dict.get("start_time", 0.0)) * 1000))
        end_ms = int(round(float(t_dict.get("end_time", 0.0)) * 1000))

        metadata = {
            "legacy_id": raw_id,
            "subtopics": json.loads(t_dict.get("subtopics_json", "[]") or "[]"),
            "why_selected": json.loads(t_dict.get("why_selected_json", "[]") or "[]"),
            "segment_ids": json.loads(t_dict.get("segment_ids_json", "[]") or "[]"),
            "speakers_involved": json.loads(t_dict.get("speakers_involved_json", "[]") or "[]"),
            "speaker_roles": json.loads(t_dict.get("speaker_roles_json", "{}") or "{}"),
            "question_included": bool(t_dict.get("question_included", 0)),
            "depends_on_question": bool(t_dict.get("depends_on_question", 0)),
            "exchange_type": t_dict.get("exchange_type", "GENERAL_TOPIC"),
            "editorial_justification": t_dict.get("editorial_justification", "")
        }

        topic_record = {
            "id": t_uuid,
            "video_id": v_uuid,
            "title": t_dict.get("name", "Untitled Topic"),
            "description": t_dict.get("description", ""),
            "importance_score": float(t_dict.get("importance_score", 0.0) or 0.0),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "metadata": metadata,
            "status": t_dict.get("status", "discovered")
        }
        topic_records.append(topic_record)
        migrated_topic_ids.add(t_uuid)

    if not dry_run and topic_records:
        for i in range(0, len(topic_records), 100):
            chunk = topic_records[i:i + 100]
            try:
                client.table("topics").upsert(chunk).execute()
                stats["topics_migrated"] += len(chunk)
            except Exception as e:
                stats["errors"].append(f"Failed to upsert topics batch: {e}")
    else:
        stats["topics_migrated"] = len(topic_records)

    print(f"  --> {stats['topics_migrated']}/{stats['topics_found']} topics successfully upserted.", flush=True)

    # 4. Migrate Clips
    print("\n[4/5] Migrating Clips...", flush=True)
    try:
        clip_rows = cursor.execute("SELECT * FROM clips").fetchall()
    except sqlite3.OperationalError:
        clip_rows = []

    stats["clips_found"] = len(clip_rows)
    print(f"  Found {len(clip_rows)} clips.", flush=True)

    clip_records = []
    for c in clip_rows:
        c_dict = dict(c)
        raw_id = c_dict["id"]
        v_uuid = to_uuid(c_dict["video_id"])
        if v_uuid not in migrated_video_ids:
            continue
        c_uuid = to_uuid(f"{v_uuid}_{raw_id}")
        raw_topic_id = c_dict.get("topic_id")
        candidate_t_uuid = to_uuid(f"{v_uuid}_{raw_topic_id}") if raw_topic_id else None
        t_uuid = candidate_t_uuid if (candidate_t_uuid in migrated_topic_ids) else None

        start_ms = int(round(float(c_dict.get("start_time", 0.0)) * 1000))
        end_ms = int(round(float(c_dict.get("end_time", 0.0)) * 1000))
        dur = float(c_dict.get("duration", (end_ms - start_ms) / 1000.0))

        clip_path_local = Path(c_dict.get("filepath", ""))
        storage_path = f"{v_uuid}/clips/{clip_path_local.name or (c_uuid + '.mp4')}"

        if not dry_run and upload_media and clip_path_local.exists():
            upload_res = SupabaseStorageService.upload_file(
                SUPABASE_STORAGE_BUCKET_CLIPS,
                clip_path_local,
                storage_path,
                "video/mp4"
            )
            if upload_res:
                stats["files_uploaded"] += 1

        clip_record = {
            "id": c_uuid,
            "video_id": v_uuid,
            "topic_id": t_uuid,
            "title": c_dict.get("text") or c_dict.get("filename") or f"Clip {raw_id}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_seconds": round(dur, 2),
            "storage_path": storage_path,
            "status": "ready",
            "metadata": {
                "legacy_id": raw_id,
                "filename": c_dict.get("filename", ""),
                "exchange_type": c_dict.get("exchange_type", "GENERAL_TOPIC"),
                "reason": c_dict.get("reason", "")
            }
        }
        clip_records.append(clip_record)

    if not dry_run and clip_records:
        for i in range(0, len(clip_records), 100):
            chunk = clip_records[i:i + 100]
            try:
                client.table("clips").upsert(chunk).execute()
                stats["clips_migrated"] += len(chunk)
            except Exception as e:
                stats["errors"].append(f"Failed to upsert clips batch: {e}")
    else:
        stats["clips_migrated"] = len(clip_records)

    print(f"  --> {stats['clips_migrated']}/{stats['clips_found']} clips successfully upserted.", flush=True)

    conn.close()

    # 5. Summary
    print("\n" + "=" * 70, flush=True)
    print("  MIGRATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Videos:              {stats['videos_migrated']}/{stats['videos_found']} migrated", flush=True)
    print(f"  Transcript Segments: {stats['segments_migrated']}/{stats['segments_found']} migrated", flush=True)
    print(f"  Topics:              {stats['topics_migrated']}/{stats['topics_found']} migrated", flush=True)
    print(f"  Clips:               {stats['clips_migrated']}/{stats['clips_found']} migrated", flush=True)
    print(f"  Files Uploaded:      {stats['files_uploaded']}", flush=True)
    print(f"  Errors Encountered:  {len(stats['errors'])}", flush=True)
    if stats["errors"]:
        print("\nErrors:", flush=True)
        for err in stats["errors"][:10]:
            print(f"  - {err}", flush=True)
    print("=" * 70, flush=True)
    print("Migration finished successfully.\n", flush=True)


if __name__ == "__main__":
    args = parse_args()
    run_migration(dry_run=args.dry_run, upload_media=args.upload_media)
