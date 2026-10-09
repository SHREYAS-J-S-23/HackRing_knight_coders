-- Migration 004: Audience Mode and Clip Notes with Persistent Caching

-- 1. Add audience_mode column to videos table
ALTER TABLE videos ADD COLUMN IF NOT EXISTS audience_mode TEXT DEFAULT 'education';

-- 2. Create clip_notes table for audience-aware notes persistence
CREATE TABLE IF NOT EXISTS clip_notes (
    id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    user_id TEXT,
    audience_mode TEXT NOT NULL DEFAULT 'education',
    notes_json JSONB NOT NULL,
    model_metadata JSONB,
    created_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
    updated_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
    UNIQUE(clip_id, audience_mode)
);

-- 3. Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_clip_notes_lookup ON clip_notes(clip_id, audience_mode);
CREATE INDEX IF NOT EXISTS idx_clip_notes_vid ON clip_notes(video_id);
CREATE INDEX IF NOT EXISTS idx_clip_notes_user ON clip_notes(user_id);
