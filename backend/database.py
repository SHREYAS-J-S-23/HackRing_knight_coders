import re
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import BASE_DIR, is_supabase_configured

DB_PATH = BASE_DIR / "storage" / "vidara.db"

def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT UNIQUE,
        email TEXT UNIQUE,
        avatar_url TEXT,
        auth_provider TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_login_at REAL NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS otp_sessions (
        id TEXT PRIMARY KEY,
        phone TEXT NOT NULL,
        otp_code TEXT NOT NULL,
        expires_at REAL NOT NULL,
        is_used INTEGER DEFAULT 0
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS saved_clips (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        video_id TEXT NOT NULL,
        clip_id TEXT NOT NULL,
        title TEXT NOT NULL,
        video_title TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        duration REAL NOT NULL,
        download_url TEXT NOT NULL,
        video_url TEXT NOT NULL,
        filepath TEXT,
        created_at REAL NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        audio_path TEXT,
        duration REAL DEFAULT 0.0,
        filesize INTEGER DEFAULT 0,
        status TEXT DEFAULT 'uploaded',
        core_thesis TEXT,
        language TEXT DEFAULT 'en',
        created_at REAL
    );
    """)

    # Ensure user_id column exists if table was created previously
    try:
        cursor.execute("ALTER TABLE videos ADD COLUMN user_id TEXT;")
    except Exception:
        pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcript_segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        segment_index INTEGER NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        text TEXT NOT NULL,
        speaker TEXT DEFAULT 'SPEAKER_01',
        words_json TEXT,
        embedding_json TEXT,
        candidate_topics_json TEXT,
        category TEXT DEFAULT 'SUPPORTING',
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        importance_score REAL DEFAULT 0.0,
        confidence REAL DEFAULT 0.0,
        coverage_score REAL DEFAULT 0.0,
        depth_score REAL DEFAULT 0.0,
        centrality_score REAL DEFAULT 0.0,
        subtopics_json TEXT,
        why_selected_json TEXT,
        segment_ids_json TEXT,
        status TEXT DEFAULT 'discovered',
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topic_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        source_topic TEXT NOT NULL,
        target_topic TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clips (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        topic_id TEXT,
        clip_index INTEGER NOT NULL,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        duration REAL NOT NULL,
        text TEXT,
        is_selected INTEGER DEFAULT 1,
        download_url TEXT,
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    # Ensure podcast metadata columns exist in topics and clips
    for col, col_type in [
        ("exchange_type", "TEXT DEFAULT 'GENERAL_TOPIC'"),
        ("speakers_involved_json", "TEXT DEFAULT '[]'"),
        ("speaker_roles_json", "TEXT DEFAULT '{}'"),
        ("question_included", "INTEGER DEFAULT 0"),
        ("depends_on_question", "INTEGER DEFAULT 0"),
        ("editorial_justification", "TEXT DEFAULT ''"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE topics ADD COLUMN {col} {col_type};")
        except Exception:
            pass

    for col, col_type in [
        ("exchange_type", "TEXT DEFAULT 'GENERAL_TOPIC'"),
        ("speakers_involved_json", "TEXT DEFAULT '[]'"),
        ("speaker_roles_json", "TEXT DEFAULT '{}'"),
        ("question_included", "INTEGER DEFAULT 0"),
        ("depends_on_question", "INTEGER DEFAULT 0"),
        ("editorial_justification", "TEXT DEFAULT ''"),
        ("subtitle_url", "TEXT"),
        ("reason", "TEXT"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE clips ADD COLUMN {col} {col_type};")
        except Exception:
            pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_jobs (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        user_id TEXT,
        job_type TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        current_stage TEXT,
        message TEXT,
        result_json TEXT,
        error TEXT,
        created_at REAL,
        updated_at REAL
    );
    """)

    try:
        cursor.execute("ALTER TABLE analysis_jobs ADD COLUMN user_id TEXT;")
    except Exception:
        pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcript_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        time_offset REAL NOT NULL,
        duration REAL DEFAULT 0.0,
        segments_json TEXT NOT NULL,
        status TEXT DEFAULT 'completed',
        retries INTEGER DEFAULT 0,
        latency REAL DEFAULT 0.0,
        error TEXT,
        created_at REAL,
        UNIQUE(video_id, chunk_index)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_queries (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        query_text TEXT NOT NULL,
        concepts_json TEXT,
        matched_topics_json TEXT,
        created_at REAL,
        FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    # Performance optimization: Database Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_segments_vid ON transcript_segments(video_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_segments_vid_time ON transcript_segments(video_id, start_time);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_vid ON topics(video_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clips_vid ON clips(video_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_vid ON analysis_jobs(video_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vid ON transcript_chunks(video_id);")

    conn.commit()
    conn.close()

# Initialize DB at import
init_db()

class DatabaseService:
    @staticmethod
    def save_video(video_id: str, filename: str, filepath: str, audio_path: str = "", duration: float = 0.0, filesize: int = 0, user_id: Optional[str] = None) -> None:
        conn = get_db_connection()
        conn.execute("""
            INSERT OR REPLACE INTO videos (id, user_id, filename, filepath, audio_path, duration, filesize, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (video_id, user_id, filename, filepath, audio_path, duration, filesize, time.time()))
        conn.commit()
        conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.save_video(
                    video_id=video_id,
                    filename=filename,
                    filepath=filepath,
                    audio_path=audio_path,
                    duration=duration,
                    filesize=filesize,
                    user_id=user_id
                )
            except Exception as e:
                print(f"Warning: Supabase save_video error: {e}")

    @staticmethod
    def update_video_status(video_id: str, status: str, core_thesis: str = None) -> None:
        conn = get_db_connection()
        if core_thesis:
            conn.execute("UPDATE videos SET status = ?, core_thesis = ? WHERE id = ?", (status, core_thesis, video_id))
        else:
            conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
        conn.commit()
        conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.update_video_status(video_id=video_id, status=status, core_thesis=core_thesis)
            except Exception as e:
                print(f"Warning: Supabase update_video_status error: {e}")

    @staticmethod
    def get_video(video_id: str) -> Optional[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_vid = SupabaseDbService.get_video(video_id)
                if supa_vid:
                    return supa_vid
            except Exception as e:
                print(f"Warning: Supabase get_video error: {e}")

        conn = get_db_connection()
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def save_segments(video_id: str, segments: List[Dict[str, Any]]) -> None:
        """Batch-persists transcript segments with batch embedding and transaction."""
        from backend.services.ai_providers import EmbeddingProvider

        if not segments:
            return

        # Precompute any missing embeddings in bulk
        missing_texts = [s.get("text", "") for s in segments if not s.get("embedding") and s.get("text")]
        batch_embeddings = EmbeddingProvider.embed_batch(missing_texts) if missing_texts else []
        emb_iter = iter(batch_embeddings)

        rows = []
        for s in segments:
            text = s.get("text", "")
            emb = s.get("embedding")
            if not emb and text:
                emb = next(emb_iter, [])

            rows.append((
                video_id,
                s.get("segment_index", s.get("id", 0)),
                s.get("start", s.get("start_time", 0.0)),
                s.get("end", s.get("end_time", 0.0)),
                text,
                s.get("speaker", "SPEAKER_01"),
                json.dumps(s.get("words", [])),
                json.dumps(emb or []),
                json.dumps(s.get("candidate_topics", [])),
                s.get("category", "SUPPORTING")
            ))

        conn = get_db_connection()
        try:
            with conn:
                conn.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
                conn.executemany("""
                    INSERT INTO transcript_segments 
                    (video_id, segment_index, start_time, end_time, text, speaker, words_json, embedding_json, candidate_topics_json, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
        finally:
            conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.save_segments(video_id=video_id, segments=segments)
            except Exception as e:
                print(f"Warning: Supabase save_segments error: {e}")

    @staticmethod
    def get_segments(video_id: str) -> List[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_segs = SupabaseDbService.get_segments(video_id)
                if supa_segs:
                    return supa_segs
            except Exception as e:
                print(f"Warning: Supabase get_segments error: {e}")

        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time ASC", (video_id,)).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["start"] = d.get("start_time", 0.0)
            d["end"] = d.get("end_time", 0.0)
            d["words"] = json.loads(d["words_json"]) if d["words_json"] else []
            d["embedding"] = json.loads(d["embedding_json"]) if d["embedding_json"] else []
            d["candidate_topics"] = json.loads(d["candidate_topics_json"]) if d["candidate_topics_json"] else []
            results.append(d)
        return results

    @staticmethod
    def search_segments(video_id: str, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Vector cosine similarity + semantic keyword hybrid retrieval over stored segments.
        Does not call LLM; operates entirely on indexed embeddings.
        """
        from backend.services.ai_providers import EmbeddingProvider

        all_segs = DatabaseService.get_segments(video_id)
        if not all_segs:
            return []

        query_vec = EmbeddingProvider.embed_text(query)
        q_terms = set(re.findall(r"\w+", query.lower()))

        scored_segs = []
        for s in all_segs:
            emb = s.get("embedding", [])
            cos_sim = EmbeddingProvider.cosine_similarity(query_vec, emb) if emb else 0.0
            s_text = s.get("text", "").lower()
            keyword_hits = sum(1 for term in q_terms if len(term) > 2 and term in s_text)
            
            # Hybrid relevance score: cosine similarity + keyword hit bonus
            hybrid_score = round(cos_sim * 0.7 + min(0.3, keyword_hits * 0.1), 3)
            # Require at least one non-trivial keyword hit or solid cosine similarity (>= 0.45)
            if (keyword_hits > 0 and hybrid_score > 0.15) or hybrid_score >= 0.45:
                s_copy = dict(s)
                s_copy["score"] = hybrid_score
                scored_segs.append(s_copy)

        scored_segs.sort(key=lambda x: x["score"], reverse=True)
        return scored_segs[:top_k]

    @staticmethod
    def save_topics(video_id: str, topics: List[Dict[str, Any]]) -> None:
        """Batch-persists discovered topics using executemany inside a transaction."""
        if not topics:
            return

        rows = []
        for t in topics:
            tid = t["id"]
            if not tid.startswith(video_id):
                tid = f"{video_id}_{tid}"
            rows.append((
                tid,
                video_id,
                t["name"],
                t.get("description", ""),
                t["start_time"],
                t["end_time"],
                t.get("importance_score", 0.0),
                t.get("confidence", 0.0),
                t.get("coverage_score", 0.0),
                t.get("depth_score", 0.0),
                t.get("centrality_score", 0.0),
                json.dumps(t.get("subtopics", [])),
                json.dumps(t.get("why_selected", [])),
                json.dumps(t.get("segment_ids", [])),
                t.get("status", "discovered"),
                t.get("exchange_type", "GENERAL_TOPIC"),
                json.dumps(t.get("speakers_involved", [])),
                json.dumps(t.get("speaker_roles", {})),
                1 if t.get("question_included") else 0,
                1 if t.get("depends_on_question") else 0,
                t.get("editorial_justification", "")
            ))

        conn = get_db_connection()
        try:
            with conn:
                conn.execute("DELETE FROM topics WHERE video_id = ?", (video_id,))
                conn.executemany("""
                    INSERT OR REPLACE INTO topics 
                    (id, video_id, name, description, start_time, end_time, importance_score, confidence,
                     coverage_score, depth_score, centrality_score, subtopics_json, why_selected_json, segment_ids_json, status,
                     exchange_type, speakers_involved_json, speaker_roles_json, question_included, depends_on_question, editorial_justification)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
        finally:
            conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.save_topics(video_id=video_id, topics=topics)
            except Exception as e:
                print(f"Warning: Supabase save_topics error: {e}")

    # -------------------------------------------------------------
    # CHUNK CHECKPOINT RESUMPTION
    # -------------------------------------------------------------
    @staticmethod
    def save_chunk_checkpoint(
        video_id: str,
        chunk_index: int,
        time_offset: float,
        duration: float,
        segments_data: List[Dict[str, Any]],
        retries: int = 0,
        latency: float = 0.0,
        status: str = "completed",
        error: str = ""
    ) -> None:
        """Persists successful or failed chunk state for resumption without retranscribing."""
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO transcript_chunks
                    (video_id, chunk_index, time_offset, duration, segments_json, status, retries, latency, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    chunk_index,
                    time_offset,
                    duration,
                    json.dumps(segments_data),
                    status,
                    retries,
                    round(latency, 3),
                    error,
                    time.time()
                ))
        finally:
            conn.close()

    @staticmethod
    def get_chunk_checkpoints(video_id: str) -> List[Dict[str, Any]]:
        """Retrieves previously completed chunk checkpoints for this video."""
        conn = get_db_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM transcript_chunks WHERE video_id = ? ORDER BY chunk_index ASC",
                (video_id,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["segments"] = json.loads(d["segments_json"]) if d["segments_json"] else []
                results.append(d)
            return results
        finally:
            conn.close()

    @staticmethod
    def clear_chunk_checkpoints(video_id: str) -> None:
        """Clears chunk checkpoints when re-running clean from scratch."""
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("DELETE FROM transcript_chunks WHERE video_id = ?", (video_id,))
        finally:
            conn.close()

    @staticmethod
    def get_topics(video_id: str) -> List[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_topics = SupabaseDbService.get_topics(video_id)
                if supa_topics:
                    return supa_topics
            except Exception as e:
                print(f"Warning: Supabase get_topics error: {e}")

        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM topics WHERE video_id = ? ORDER BY importance_score DESC", (video_id,)).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["subtopics"] = json.loads(d["subtopics_json"]) if d.get("subtopics_json") else []
            d["why_selected"] = json.loads(d["why_selected_json"]) if d.get("why_selected_json") else []
            d["segment_ids"] = json.loads(d["segment_ids_json"]) if d.get("segment_ids_json") else []
            d["speakers_involved"] = json.loads(d["speakers_involved_json"]) if d.get("speakers_involved_json") else []
            d["speaker_roles"] = json.loads(d["speaker_roles_json"]) if d.get("speaker_roles_json") else {}
            d["question_included"] = bool(d.get("question_included", 0))
            d["depends_on_question"] = bool(d.get("depends_on_question", 0))
            d["exchange_type"] = d.get("exchange_type", "GENERAL_TOPIC")
            d["editorial_justification"] = d.get("editorial_justification", "")
            results.append(d)
        return results

    @staticmethod
    def save_topic_relations(video_id: str, relations: List[Dict[str, Any]]) -> None:
        conn = get_db_connection()
        conn.execute("DELETE FROM topic_relations WHERE video_id = ?", (video_id,))
        for r in relations:
            conn.execute("""
                INSERT INTO topic_relations (video_id, source_topic, target_topic, relation_type, weight)
                VALUES (?, ?, ?, ?, ?)
            """, (video_id, r["source"], r["target"], r.get("relation_type", "related"), r.get("weight", 1.0)))
        conn.commit()
        conn.close()

    @staticmethod
    def save_clips(video_id: str, clips: List[Dict[str, Any]]) -> None:
        conn = get_db_connection()
        conn.execute("DELETE FROM clips WHERE video_id = ?", (video_id,))
        for c in clips:
            conn.execute("""
                INSERT OR REPLACE INTO clips 
                (id, video_id, topic_id, clip_index, filename, filepath, start_time, end_time, duration, text, is_selected, download_url,
                 exchange_type, speakers_involved_json, speaker_roles_json, question_included, depends_on_question, editorial_justification, subtitle_url, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c["id"],
                video_id,
                c.get("topic_id"),
                c["clip_index"],
                c["filename"],
                c["filepath"],
                c["start_time"],
                c["end_time"],
                c["duration"],
                c.get("text", ""),
                1 if c.get("is_selected", True) else 0,
                c.get("download_url", ""),
                c.get("exchange_type", "GENERAL_TOPIC"),
                json.dumps(c.get("speakers_involved", [])),
                json.dumps(c.get("speaker_roles", {})),
                1 if c.get("question_included") else 0,
                1 if c.get("depends_on_question") else 0,
                c.get("editorial_justification", ""),
                c.get("subtitle_url", ""),
                c.get("reason", "")
            ))
        conn.commit()
        conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.save_clips(video_id=video_id, clips=clips)
            except Exception as e:
                print(f"Warning: Supabase save_clips error: {e}")

    @staticmethod
    def get_clips(video_id: str) -> List[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_clips = SupabaseDbService.get_clips(video_id)
                if supa_clips:
                    return supa_clips
            except Exception as e:
                print(f"Warning: Supabase get_clips error: {e}")

        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM clips WHERE video_id = ? ORDER BY clip_index ASC", (video_id,)).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["speakers_involved"] = json.loads(d["speakers_involved_json"]) if d.get("speakers_involved_json") else []
            d["speaker_roles"] = json.loads(d["speaker_roles_json"]) if d.get("speaker_roles_json") else {}
            d["question_included"] = bool(d.get("question_included", 0))
            d["depends_on_question"] = bool(d.get("depends_on_question", 0))
            d["exchange_type"] = d.get("exchange_type", "GENERAL_TOPIC")
            d["editorial_justification"] = d.get("editorial_justification", "")
            d["reason"] = d.get("reason", "")
            results.append(d)
        return results

    @staticmethod
    def set_job(
        job_id: str,
        video_id: str,
        job_type: str,
        status: str,
        progress: int,
        stage: str,
        message: str,
        result: Any = None,
        error: str = None,
        chunks_completed: int = 0,
        chunks_total: int = 0,
        elapsed_seconds: float = 0.0,
        retries: int = 0,
        user_id: Optional[str] = None
    ):
        conn = get_db_connection()
        now = time.time()

        # If user_id is not passed, attempt to inherit from the parent video record
        resolved_user_id = user_id
        if not resolved_user_id:
            try:
                v_row = conn.execute("SELECT user_id FROM videos WHERE id = ?", (video_id,)).fetchone()
                if v_row and v_row["user_id"]:
                    resolved_user_id = v_row["user_id"]
            except Exception:
                pass

        res_data = result if isinstance(result, dict) else ({"data": result} if result is not None else {})
        res_data["chunks_completed"] = chunks_completed
        res_data["chunks_total"] = chunks_total
        res_data["elapsed_seconds"] = round(elapsed_seconds, 1)
        res_data["retries"] = retries
        res_data["stage"] = stage
        res_json = json.dumps(res_data)

        conn.execute("""
            INSERT OR REPLACE INTO analysis_jobs (id, video_id, user_id, job_type, status, progress, current_stage, message, result_json, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM analysis_jobs WHERE id = ?), ?), ?)
        """, (job_id, video_id, resolved_user_id, job_type, status, progress, stage, message, res_json, error, job_id, now, now))
        conn.commit()
        conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.set_job(
                    job_id=job_id,
                    video_id=video_id,
                    job_type=job_type,
                    status=status,
                    progress=progress,
                    stage=stage,
                    message=message,
                    result=res_data,
                    error=error,
                    user_id=resolved_user_id
                )
            except Exception as e:
                print(f"Warning: Supabase set_job error: {e}")

    @staticmethod
    def get_job(job_id: str) -> Optional[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_job = SupabaseDbService.get_job(job_id)
                if supa_job:
                    return supa_job
            except Exception as e:
                print(f"Warning: Supabase get_job error: {e}")

        conn = get_db_connection()
        row = conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        d["stage"] = d.get("current_stage", "")
        d["chunks_completed"] = 0
        d["chunks_total"] = 0
        d["elapsed_seconds"] = 0.0
        d["retries"] = 0
        if d["result_json"]:
            try:
                res = json.loads(d["result_json"])
                d["result"] = res
                if isinstance(res, dict):
                    d["chunks_completed"] = res.get("chunks_completed", 0)
                    d["chunks_total"] = res.get("chunks_total", 0)
                    d["elapsed_seconds"] = res.get("elapsed_seconds", 0.0)
                    d["retries"] = res.get("retries", 0)
            except Exception:
                d["result"] = d["result_json"]
        return d

    # -------------------------------------------------------------
    # USER & AUTHENTICATION METHODS
    # -------------------------------------------------------------
    @staticmethod
    def upsert_user(user_id: str, name: str, phone: Optional[str] = None, email: Optional[str] = None, avatar_url: Optional[str] = None, auth_provider: str = "phone") -> Dict[str, Any]:
        conn = get_db_connection()
        now = time.time()
        conn.execute("""
            INSERT INTO users (id, name, phone, email, avatar_url, auth_provider, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                phone = COALESCE(excluded.phone, users.phone),
                email = COALESCE(excluded.email, users.email),
                avatar_url = COALESCE(excluded.avatar_url, users.avatar_url),
                last_login_at = excluded.last_login_at
        """, (user_id, name, phone, email, avatar_url, auth_provider, now, now))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else {}

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_user_by_phone(phone: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone.strip(),)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def save_otp(phone: str, otp_code: str, expires_in_seconds: int = 300) -> str:
        import uuid
        conn = get_db_connection()
        otp_id = str(uuid.uuid4())[:8]
        expires_at = time.time() + expires_in_seconds
        conn.execute("UPDATE otp_sessions SET is_used = 1 WHERE phone = ?", (phone.strip(),))
        conn.execute("""
            INSERT INTO otp_sessions (id, phone, otp_code, expires_at, is_used)
            VALUES (?, ?, ?, ?, 0)
        """, (otp_id, phone.strip(), otp_code.strip(), expires_at))
        conn.commit()
        conn.close()
        return otp_id

    @staticmethod
    def verify_otp(phone: str, otp_code: str) -> bool:
        conn = get_db_connection()
        now = time.time()
        row = conn.execute("""
            SELECT * FROM otp_sessions 
            WHERE phone = ? AND otp_code = ? AND is_used = 0 AND expires_at >= ?
            ORDER BY expires_at DESC LIMIT 1
        """, (phone.strip(), otp_code.strip(), now)).fetchone()
        if row:
            conn.execute("UPDATE otp_sessions SET is_used = 1 WHERE id = ?", (row["id"],))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False

    # -------------------------------------------------------------
    # PERMANENT SAVED CLIPS & LIBRARY METHODS
    # -------------------------------------------------------------
    @staticmethod
    def save_user_clip(
        user_id: str,
        video_id: str,
        clip_id: str,
        title: str,
        video_title: str,
        start_time: float,
        end_time: float,
        duration: float,
        download_url: str,
        video_url: str,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        import uuid
        conn = get_db_connection()
        now = time.time()
        saved_id = f"saved_{uuid.uuid4().hex[:12]}"

        # Avoid duplicate saves for the same clip
        existing = conn.execute("""
            SELECT * FROM saved_clips WHERE user_id = ? AND video_id = ? AND clip_id = ?
        """, (user_id, video_id, clip_id)).fetchone()

        if existing:
            conn.close()
            return dict(existing)

        conn.execute("""
            INSERT INTO saved_clips (
                id, user_id, video_id, clip_id, title, video_title,
                start_time, end_time, duration, download_url, video_url, filepath, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            saved_id, user_id, video_id, clip_id, title, video_title,
            start_time, end_time, duration, download_url, video_url, filepath, now
        ))
        conn.commit()
        row = conn.execute("SELECT * FROM saved_clips WHERE id = ?", (saved_id,)).fetchone()
        conn.close()
        return dict(row) if row else {}

    @staticmethod
    def get_user_saved_clips(user_id: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        rows = conn.execute("""
            SELECT * FROM saved_clips WHERE user_id = ? ORDER BY created_at DESC
        """, (user_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def delete_user_saved_clip(user_id: str, clip_id: str) -> bool:
        conn = get_db_connection()
        res = conn.execute("""
            DELETE FROM saved_clips 
            WHERE user_id = ? AND (id = ? OR clip_id = ?)
        """, (user_id, clip_id, clip_id))
        deleted = res.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    @staticmethod
    def is_clip_saved(user_id: str, clip_id: str) -> bool:
        conn = get_db_connection()
        row = conn.execute("""
            SELECT 1 FROM saved_clips WHERE user_id = ? AND (id = ? OR clip_id = ?) LIMIT 1
        """, (user_id, clip_id, clip_id)).fetchone()
        conn.close()
        return bool(row)

    @staticmethod
    def get_user_videos(user_id: str) -> List[Dict[str, Any]]:
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                supa_vids = SupabaseDbService.get_user_videos(user_id)
                if supa_vids:
                    return supa_vids
            except Exception as e:
                print(f"Warning: Supabase get_user_videos error: {e}")

        conn = get_db_connection()
        rows = conn.execute("""
            SELECT * FROM videos WHERE user_id = ? ORDER BY created_at DESC
        """, (user_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def save_clip_selection(user_id: str, video_id: str, clip_id: str, position: int, selected: bool = True) -> Dict[str, Any]:
        """Persists user clip selection and playlist ordering."""
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                return SupabaseDbService.save_clip_selection(user_id, video_id, clip_id, position, selected)
            except Exception as e:
                print(f"Warning: Supabase save_clip_selection error: {e}")
        return {"user_id": user_id, "video_id": video_id, "clip_id": clip_id, "position": position, "selected": selected}

    @staticmethod
    def get_clip_selections(user_id: str, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves user clip selections and playlist ordering."""
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                return SupabaseDbService.get_clip_selections(user_id, video_id)
            except Exception as e:
                print(f"Warning: Supabase get_clip_selections error: {e}")
        return []

    @staticmethod
    def claim_video_ownership(video_id: str, user_id: str) -> bool:
        """Associates an unowned legacy video record with the authenticated user."""
        conn = get_db_connection()
        conn.execute("UPDATE videos SET user_id = ? WHERE id = ? AND (user_id IS NULL OR user_id = '')", (user_id, video_id))
        conn.execute("UPDATE analysis_jobs SET user_id = ? WHERE video_id = ? AND (user_id IS NULL OR user_id = '')", (user_id, video_id))
        conn.commit()
        conn.close()

        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.claim_video_ownership(video_id, user_id)
            except Exception as e:
                print(f"Warning: Supabase claim_video_ownership error: {e}")
        return True

    @staticmethod
    def delete_video(video_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes video and all associated local records and files."""
        conn = get_db_connection()
        # Find local file paths before deleting records
        vid_row = conn.execute("SELECT filepath, audio_path FROM videos WHERE id = ?", (video_id,)).fetchone()
        local_vpath = vid_row["filepath"] if vid_row and vid_row["filepath"] else None
        local_apath = vid_row["audio_path"] if vid_row and vid_row["audio_path"] else None

        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        conn.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM topics WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM clips WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM analysis_jobs WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM transcript_chunks WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM user_queries WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM saved_clips WHERE video_id = ?", (video_id,))
        conn.commit()
        conn.close()

        # Delete local media files if present
        try:
            if local_vpath and Path(local_vpath).exists():
                Path(local_vpath).unlink(missing_ok=True)
            if local_apath and Path(local_apath).exists():
                Path(local_apath).unlink(missing_ok=True)
        except Exception as f_err:
            print(f"Warning deleting local files for {video_id}: {f_err}")

        # Supabase deletion
        if is_supabase_configured():
            try:
                from backend.services.supabase_service import SupabaseDbService
                SupabaseDbService.delete_video(video_id=video_id, user_id=user_id)
            except Exception as e:
                print(f"Warning: Supabase delete_video error: {e}")
        return True

