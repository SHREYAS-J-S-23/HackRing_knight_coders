-- ==============================================================================
-- VIDARA AI: MIGRATION 003 — STRICT USER DATA OWNERSHIP & ROW LEVEL SECURITY
-- ==============================================================================
-- This migration guarantees complete multi-tenant user isolation:
-- 1. All videos, transcript segments, topics, clips, selections, and jobs
--    are strictly bound to the authenticated owner (auth.uid()).
-- 2. Child records (segments, topics, clips) inherit ownership via parent video.
-- 3. Storage objects are strictly isolated into user-owned directories.
-- 4. Service role retains full bypass access for background workers and maintenance.
-- ==============================================================================

-- 1. Ensure columns exist on profiles
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS username TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS email TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now());

-- 2. Ensure user_id indexes exist for fast ownership queries
CREATE INDEX IF NOT EXISTS idx_videos_user_id ON public.videos(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON public.processing_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_clip_selections_user_id ON public.clip_selections(user_id);

-- ------------------------------------------------------------------------------
-- A. TIGHTEN RLS ON VIDEOS TABLE
-- ------------------------------------------------------------------------------
ALTER TABLE public.videos ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own videos" ON public.videos;
CREATE POLICY "Users can view own videos"
    ON public.videos FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own videos" ON public.videos;
CREATE POLICY "Users can insert own videos"
    ON public.videos FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own videos" ON public.videos;
CREATE POLICY "Users can update own videos"
    ON public.videos FOR UPDATE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own videos" ON public.videos;
CREATE POLICY "Users can delete own videos"
    ON public.videos FOR DELETE
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Service role full access on videos" ON public.videos;
CREATE POLICY "Service role full access on videos"
    ON public.videos FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ------------------------------------------------------------------------------
-- B. TIGHTEN RLS ON TRANSCRIPT SEGMENTS (Inherited through parent video)
-- ------------------------------------------------------------------------------
ALTER TABLE public.transcript_segments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view segments of their videos" ON public.transcript_segments;
CREATE POLICY "Users can view segments of their videos"
    ON public.transcript_segments FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = transcript_segments.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert segments of their videos" ON public.transcript_segments;
CREATE POLICY "Users can insert segments of their videos"
    ON public.transcript_segments FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = transcript_segments.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can delete segments of their videos" ON public.transcript_segments;
CREATE POLICY "Users can delete segments of their videos"
    ON public.transcript_segments FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = transcript_segments.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Service role full access on transcript_segments" ON public.transcript_segments;
CREATE POLICY "Service role full access on transcript_segments"
    ON public.transcript_segments FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ------------------------------------------------------------------------------
-- C. TIGHTEN RLS ON TOPICS (Inherited through parent video)
-- ------------------------------------------------------------------------------
ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view topics of their videos" ON public.topics;
CREATE POLICY "Users can view topics of their videos"
    ON public.topics FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = topics.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert topics of their videos" ON public.topics;
CREATE POLICY "Users can insert topics of their videos"
    ON public.topics FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = topics.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can delete topics of their videos" ON public.topics;
CREATE POLICY "Users can delete topics of their videos"
    ON public.topics FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = topics.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Service role full access on topics" ON public.topics;
CREATE POLICY "Service role full access on topics"
    ON public.topics FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ------------------------------------------------------------------------------
-- D. TIGHTEN RLS ON CLIPS (Inherited through parent video)
-- ------------------------------------------------------------------------------
ALTER TABLE public.clips ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view clips of their videos" ON public.clips;
CREATE POLICY "Users can view clips of their videos"
    ON public.clips FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = clips.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can insert clips of their videos" ON public.clips;
CREATE POLICY "Users can insert clips of their videos"
    ON public.clips FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = clips.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can update clips of their videos" ON public.clips;
CREATE POLICY "Users can update clips of their videos"
    ON public.clips FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = clips.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Users can delete clips of their videos" ON public.clips;
CREATE POLICY "Users can delete clips of their videos"
    ON public.clips FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = clips.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Service role full access on clips" ON public.clips;
CREATE POLICY "Service role full access on clips"
    ON public.clips FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ------------------------------------------------------------------------------
-- E. TIGHTEN RLS ON PROCESSING JOBS
-- ------------------------------------------------------------------------------
ALTER TABLE public.processing_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own jobs" ON public.processing_jobs;
CREATE POLICY "Users can view own jobs"
    ON public.processing_jobs FOR SELECT
    USING (
        auth.uid() = user_id
        OR EXISTS (
            SELECT 1 FROM public.videos v
            WHERE v.id = processing_jobs.video_id
            AND v.user_id = auth.uid()
        )
    );

DROP POLICY IF EXISTS "Service role full access on processing_jobs" ON public.processing_jobs;
CREATE POLICY "Service role full access on processing_jobs"
    ON public.processing_jobs FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ------------------------------------------------------------------------------
-- F. STORAGE RLS: Strict path-based ownership isolation
-- ------------------------------------------------------------------------------
-- Allows authenticated users to select/read files strictly in their user folder ({user_id}/*)
DROP POLICY IF EXISTS "Allow authenticated read on own files" ON storage.objects;
CREATE POLICY "Allow authenticated read on own files"
    ON storage.objects FOR SELECT
    TO authenticated
    USING (
        bucket_id IN ('original-videos', 'processed-audio', 'generated-clips', 'subtitles', 'final-exports') 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- Allows authenticated users to upload files strictly into their user folder ({user_id}/*)
DROP POLICY IF EXISTS "Allow authenticated upload to own folders" ON storage.objects;
CREATE POLICY "Allow authenticated upload to own folders"
    ON storage.objects FOR INSERT
    TO authenticated
    WITH CHECK (
        bucket_id IN ('original-videos', 'processed-audio', 'generated-clips', 'subtitles', 'final-exports') 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- Allows authenticated users to delete files strictly in their user folder
DROP POLICY IF EXISTS "Allow authenticated delete on own files" ON storage.objects;
CREATE POLICY "Allow authenticated delete on own files"
    ON storage.objects FOR DELETE
    TO authenticated
    USING (
        bucket_id IN ('original-videos', 'processed-audio', 'generated-clips', 'subtitles', 'final-exports') 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );
