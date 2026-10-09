-- ==============================================================================
-- VIDARA AI: SUPABASE POSTGRESQL NORMALIZED SCHEMA & MIGRATIONS
-- ==============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ------------------------------------------------------------------------------
-- A. PROFILES (Application-specific user profile referencing auth.users)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Service role full access on profiles"
    ON public.profiles FOR ALL
    TO service_role
    USING (true);

-- ------------------------------------------------------------------------------
-- B. VIDEOS (Uploaded or imported videos)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'upload', -- 'upload' or 'url'
    source_url TEXT,
    storage_path TEXT NOT NULL DEFAULT '',       -- Path in private bucket 'original-videos'
    audio_storage_path TEXT,                    -- Path in private bucket 'processed-audio'
    duration_seconds NUMERIC DEFAULT 0.0,
    filesize_bytes BIGINT DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'uploaded',    -- 'uploaded', 'processing', 'analyzed', 'completed', 'failed'
    error_message TEXT,
    core_thesis TEXT,
    language TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own videos"
    ON public.videos FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert own videos"
    ON public.videos FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update own videos"
    ON public.videos FOR UPDATE
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can delete own videos"
    ON public.videos FOR DELETE
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Service role full access on videos"
    ON public.videos FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_videos_user_id ON public.videos(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_created_at ON public.videos(created_at DESC);

-- ------------------------------------------------------------------------------
-- C. TRANSCRIPT SEGMENTS (Word & segment timestamps with metadata)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.transcript_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    speaker_id TEXT DEFAULT 'SPEAKER_01',
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    text TEXT NOT NULL,
    words_json JSONB,
    embedding_json JSONB,
    candidate_topics_json JSONB,
    category TEXT DEFAULT 'SUPPORTING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_video_segment UNIQUE (video_id, segment_index)
);

ALTER TABLE public.transcript_segments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view segments of their videos"
    ON public.transcript_segments FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = transcript_segments.video_id
            AND (v.user_id = auth.uid() OR v.user_id IS NULL)
        )
    );

CREATE POLICY "Service role full access on transcript_segments"
    ON public.transcript_segments FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_segments_video_id ON public.transcript_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_start_ms ON public.transcript_segments(video_id, start_ms);

-- ------------------------------------------------------------------------------
-- D. TOPICS (Discovered topics, importance scores, evidence & metadata)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    importance_score NUMERIC DEFAULT 0.0,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb, -- subtopics, why_selected, segment_ids, speaker_roles, editorial_justification
    status TEXT DEFAULT 'discovered',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view topics of their videos"
    ON public.topics FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = topics.video_id
            AND (v.user_id = auth.uid() OR v.user_id IS NULL)
        )
    );

CREATE POLICY "Service role full access on topics"
    ON public.topics FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_topics_video_id ON public.topics(video_id);
CREATE INDEX IF NOT EXISTS idx_topics_importance ON public.topics(video_id, importance_score DESC);

-- ------------------------------------------------------------------------------
-- E. CLIPS (Generated clip metadata, validated timestamps, storage paths)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES public.topics(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    duration_seconds NUMERIC DEFAULT 0.0,
    storage_path TEXT,          -- Path in 'generated-clips' bucket: {user_id}/{video_id}/clips/{clip_id}.mp4
    subtitle_storage_path TEXT, -- Path in 'subtitles' bucket: {user_id}/{video_id}/subtitles/{clip_id}.vtt
    quality_score NUMERIC,
    status TEXT NOT NULL DEFAULT 'ready',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.clips ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view clips of their videos"
    ON public.clips FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = clips.video_id
            AND (v.user_id = auth.uid() OR v.user_id IS NULL)
        )
    );

CREATE POLICY "Service role full access on clips"
    ON public.clips FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_clips_video_id ON public.clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_topic_id ON public.clips(topic_id);

-- ------------------------------------------------------------------------------
-- F. CLIP SELECTIONS (User-specific selections, ordering, and playlist state)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.clip_selections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    video_id UUID NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
    clip_id UUID NOT NULL REFERENCES public.clips(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    selected BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT uq_user_video_clip UNIQUE (user_id, video_id, clip_id)
);

ALTER TABLE public.clip_selections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own clip selections"
    ON public.clip_selections FOR ALL
    USING (auth.uid() = user_id);

CREATE POLICY "Service role full access on clip_selections"
    ON public.clip_selections FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_selections_user_video ON public.clip_selections(user_id, video_id, position ASC);

-- ------------------------------------------------------------------------------
-- G. PROCESSING JOBS (Long-running asynchronous operations and stage tracking)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES public.videos(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    stage TEXT NOT NULL DEFAULT 'Media ingestion',
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    result_json JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

ALTER TABLE public.processing_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own jobs"
    ON public.processing_jobs FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Service role full access on processing_jobs"
    ON public.processing_jobs FOR ALL
    TO service_role
    USING (true);

CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON public.processing_jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON public.processing_jobs(status);

-- ------------------------------------------------------------------------------
-- H. STORAGE BUCKETS PROVISIONING & POLICIES
-- ------------------------------------------------------------------------------
-- Insert private storage buckets if they do not exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES 
    ('original-videos', 'original-videos', false, 1073741824, ARRAY['video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska']),
    ('processed-audio', 'processed-audio', false, 268435456, ARRAY['audio/mpeg', 'audio/mp3', 'audio/wav']),
    ('generated-clips', 'generated-clips', false, 1073741824, ARRAY['video/mp4']),
    ('subtitles', 'subtitles', false, 10485760, ARRAY['text/vtt', 'text/plain']),
    ('final-exports', 'final-exports', false, 2147483648, ARRAY['video/mp4'])
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: allow authenticated users to read their own folders, service_role full bypass
CREATE POLICY "Allow authenticated read on own files"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (bucket_id IN ('original-videos', 'processed-audio', 'generated-clips', 'subtitles', 'final-exports') 
           AND (storage.foldername(name))[1] = auth.uid()::text);

CREATE POLICY "Allow authenticated upload to own folders"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (bucket_id IN ('original-videos', 'processed-audio', 'generated-clips', 'subtitles', 'final-exports') 
                AND (storage.foldername(name))[1] = auth.uid()::text);
