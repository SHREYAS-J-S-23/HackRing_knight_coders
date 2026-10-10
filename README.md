# 🧠 VIDARA — AI Video & Podcast Intelligence Platform
> *"Understand Every Moment."*  
> **Autonomous Long-Form Video Comprehension, Speaker Diarization, Podcast Intelligence & Semantic Extraction Engine**

[![HACKERING 2.0](https://img.shields.io/badge/HACKERING%202.0-Round%202-blueviolet.svg?style=flat)]()
[![Voice AI Track](https://img.shields.io/badge/Track-Voice%20AI-orange.svg?style=flat)]()
[![Team](https://img.shields.io/badge/Team-Knight%20Coders-blue.svg?style=flat)]()
[![Repository](https://img.shields.io/badge/Repo-HR2--OI--6C86C871-purple.svg?style=flat)]()
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq LPU](https://img.shields.io/badge/AI%20Hardware-Groq%20LPU-f55036.svg?style=flat)](https://groq.com/)
[![Whisper Large-v3](https://img.shields.io/badge/STT-Whisper%20Large--v3-blue.svg?style=flat)](https://github.com/openai/whisper)
[![Supabase](https://img.shields.io/badge/Database-Supabase%20%2F%20PostgreSQL-3ECF8E.svg?style=flat&logo=supabase)](https://supabase.com/)
[![FFmpeg](https://img.shields.io/badge/Video%20Assembly-FFmpeg-green.svg?style=flat&logo=ffmpeg)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/Tests-34%20Core%20Passing-brightgreen.svg?style=flat)]()

---

## 🌟 Overview

**Vidara AI** is a state-of-the-art video and podcast intelligence platform built for deep long-form video comprehension (1–2hr+ podcasts, keynotes, university lectures, and founder interviews). It eliminates the core frustrations of traditional video clippers: clips that are either arbitrarily cut, bloated with irrelevant banter, or missing the framing question that gives a speaker's answer its meaning.

By continuously analyzing conversational turn dynamics, acoustic characteristics, topical graphs, and editorial boundaries, Vidara extracts the **shortest complete and meaningful clips** that preserve full editorial clarity without syllable clipping.

---

## ⚡ Core Capabilities & Innovations

### 1. 🎙️ Acoustic & Conversational Speaker Diarization
- **Multi-Modal Diarization**: Extracts acoustic features (energy profile, zero-crossing rate, autocorrelation pitch using FFmpeg and SciPy) and aligns them with conversational turn markers.
- **Observed Identity vs. Inferred Roles**: Labels speakers deterministically (`SPEAKER_00`, `SPEAKER_01`) and infers roles (`HOST`, `GUEST`, or neutral `UNKNOWN`) using speech patterns without assuming the first speaker is always the host.
- **Word-Level Speaker Mapping**: Every transcribed word and utterance contains timestamps, speaker IDs, and confidence levels.

### 2. 🧩 Conversation-Aware Semantic Representation
Vidara parses transcripts into structured conversational units:
- `QUESTION` (host inquiry, premise framing)
- `ANSWER` (guest response, core explanation)
- `FOLLOW_UP` (clarifications that alter or qualify previous claims)
- `CLARIFICATION` & `CONCLUSION` (takeaways, lessons)
- `DISAGREEMENT` (meaningful debate, counter-arguments)
- `EXAMPLE_STORY` (revelations, personal anecdotes)
- `TRANSITION` & `DIGRESSION` (banter, filler, introductions)

Answers are automatically linked to preceding questions via `depends_on_question`, distinguishing indexical answers from `is_standalone_insight`.

### 3. 🎬 5 High-Value Podcast Moment Archetypes
Vidara extracts 5 complete podcast exchange archetypes:
1. **`QUESTION_AND_ANSWER`**: Host inquiry paired with guest answer, with introductory rambles trimmed.
2. **`STANDALONE_INSIGHT`**: Self-contained lessons extracted without forcing unrelated questions.
3. **`QA_WITH_FOLLOWUP`**: Inquiry, answer, and essential follow-up question/qualification.
4. **`MEANINGFUL_DISAGREEMENT`**: Debates preserving opposing perspectives.
5. **`REVELATION_OR_STORY`**: Narrative turning points or personal stories.

### 4. 🎯 Soft Duration Targets & Natural Boundary Guard
- **Short insight**: 15–30 seconds
- **Question & concise answer**: 20–60 seconds
- **Detailed exchange**: 45–90 seconds
- **Natural Boundary Guard**: Preserves natural speech boundaries with `+80ms` lead-in and `+160ms` tail-out padding to prevent syllable or word clipping.

### 5. 🛡️ 10-Point Editorial Validation Engine
Integrated with [`MeaningValidator`](backend/services/meaning_validator.py) and a multi-model fallback chain:
1. What is the central insight?
2. Why is this moment worth watching?
3. Is the question necessary to understand the answer?
4. Does the clip preserve the complete point?
5. Is any included dialogue irrelevant or repetitive?
6. Does the ending occur after the useful exchange concludes?
7. Does the beginning depend on missing context?
8. Are essential qualifications or counterarguments preserved?
9. Can the clip be shortened without losing meaning?
10. Does the moment justify a standalone clip?

*Fallback Resilience*: If the primary LLM is unavailable or rate-limited, the system falls back through secondary models (`GROQ_FALLBACK_MODEL`, `qwen/qwen3.8-27b`) and finally to a deterministic 10-point rule-based audit without crashing the pipeline.

### 6. 👥 Audience Persona Adaptations & Tone Targeting
Tailor clip selection, editorial framing, and summaries according to specific audience archetypes:
- **Education / Students**: Focuses on core concepts, revision notes, conceptual explanations, and practice quiz questions.
- **Professional / Executives**: Emphasizes strategic takeaways, technical architecture breakdowns, and implementation insights.
- **Content Creators**: Generates high-retention hooks, viral titles, social captions, and storytelling breakdowns.

### 7. 📝 AI Clip Notes & Smart Summaries with Text-to-Speech (TTS)
- Automated generation of structured executive summaries, key bullet points, and actionable takeaways for every extracted moment.
- Built-in audio narration (TTS) allowing users to listen to clip summaries on-the-go.
- Instant mode-specific note switching cached per clip.

### 8. 🔤 Synchronized WebVTT Subtitles with Speaker Attribution
Generates WebVTT subtitles containing `<v SPEAKER_XX>Dialogue text</v>` speaker styling tags, synchronized with the video player and available for instant download and burning.

### 9. 🔒 Supabase PostgreSQL & Cloud Data Isolation (RLS)
- Supports dual persistence: local SQLite for offline/development and Supabase PostgreSQL with Storage buckets for cloud production.
- Strict multi-tenant data isolation with Row-Level Security (RLS) ensuring each user's videos, notes, and clips remain private and secure.
- Dedicated migrations for profiles, user ownership, and audience notes.

### 10. 🚀 High-Performance Architecture
- **Parallel Chunking**: Chunks audio into <= 10-minute segments to bypass upload limits and enable concurrent Whisper transcription on Groq LPUs.
- **Sub-Second Semantic Retrieval**: Local vector index allows instant natural language and voice-based question answering across hours of content.
- **Selective Merge Engine**: Stitches selected highlight clips into a polished continuous reel without re-encoding delays.

---

## 🏗️ Pipeline Architecture

```
Raw Long-Form Video / URL (1-2hr+)
       │
       ▼
[AudioExtractor] ──► 16kHz Mono Audio Chunks (<= 10 mins bypasses Groq 25MB limit)
       │
       ▼
[GroqSTTService] ──► Word-Level Timestamps (Whisper Large-v3 via Groq LPU)
       │
       ▼
[SpeakerDiarizationService] ──► Acoustic Features + Turn Dynamics + Role Scoring
       │
       ▼
[PodcastIntelligenceEngine] ──► Conversational Units & 5 Exchange Archetypes
       │
       ▼
[MeaningValidator] ──► 10-Point Editorial Audit & Multi-Model Fallback Chain
       │
       ▼
[AudienceNotesService] ──► Persona Adaptation (Education / Professional / Creator)
       │                   └── Text-to-Speech (TTS) Narration Generation
       ▼
[VideoCutter] ──► Deterministic FFmpeg Boundary Trimming (+80ms/-160ms)
       │          └── WebVTT Subtitle Generation (<v SPEAKER_XX> tags)
       ▼
[Vidara Video Studio] ──► Studio Player, Topic Cards, Selective Merge & Library
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.10 or higher
- **FFmpeg**: System binary installed and added to `PATH`
- **Groq API Key**: [console.groq.com](https://console.groq.com)
- **Supabase Account** *(Optional)*: [supabase.com](https://supabase.com) for cloud PostgreSQL and storage

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/hackering-2-0/HR2-OI-6C86C871.git
cd HR2-OI-6C86C871
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Set your API keys in `.env`:
```env
# Dedicated Video Analysis Key (Whisper STT, transcription)
VIDEO_ANALYSIS_API_KEY=gsk_your_groq_key_here
VIDEO_ANALYSIS_MODEL=whisper-large-v3

# Dedicated Reasoning Key (Editorial reasoning, topic discovery)
REASONING_API_KEY=gsk_your_groq_key_here
REASONING_MODEL=qwen/qwen3.8-27b
REASONING_FALLBACK_MODEL=llama-3.3-70b-versatile

# Master Key (Used if dedicated keys above are not set)
GROQ_API_KEY=gsk_your_groq_key_here

# Optional: Supabase Cloud Database & Storage
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### 3. Launch the Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Automated Test Suite

Vidara includes comprehensive tests verifying core systems, intelligent boundary trimming, podcast conversational archetypes, audience targeting, and cloud persistence:

```bash
# Run the 34 core platform tests
python -m unittest test_vidara.py test_intelligent_trimming.py test_podcast_intelligence.py
```

### Test Coverage Highlights:
- `test_valuable_host_question_and_answer`: Verifies host inquiry and guest response binding.
- `test_insightful_answer_stands_alone`: Standalone principles extracted without unrelated banter.
- `test_long_answer_trimmed_to_core_section`: Preamble and post-chatter trimmed to core explanation.
- `test_followup_that_changes_meaning_is_retained`: Crucial qualification questions preserved.
- `test_answer_requiring_preceding_question`: Question prepended when answer starts indexically.
- `test_irrelevant_anecdote_penalized_and_trimmed`: Unrelated anecdotes pruned.
- `test_multi_speaker_and_dialogue_alternation`: Multi-speaker conversations handled accurately.
- `test_repeated_qa_deduplication`: Suppresses duplicate or highly overlapping exchanges.
- `test_topic_revisited_separated_in_time`: Handles recurring topics separated chronologically.
- `test_missing_speaker_labels_fallback`: Graceful neutral fallback when diarization is absent.
- `test_editorial_validator_resilient_fallback`: Resilient fallback during LLM provider rate limits.
- `test_subtitle_vtt_generation`: Accurate WebVTT cutting with `<v SPEAKER_XX>` tags.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/videos/upload` | Streams video file, extracts audio, registers video in DB |
| `POST` | `/api/videos/ingest-url` | Ingests public video URL (YouTube, direct MP4) with optional audience mode |
| `POST` | `/api/videos/{id}/analyze` | Chunks audio, transcribes with Whisper Large-v3, and builds semantic index |
| `POST` | `/api/videos/{id}/discover-topics` | Autonomous podcast moment discovery & boundary ranking |
| `POST` | `/api/videos/{id}/query` | Natural language / voice semantic retrieval |
| `POST` | `/api/videos/{id}/generate-clips` | Renders validated standalone topic clips with subtitles |
| `POST` | `/api/videos/{id}/merge` | Deterministically stitches selected clips into a master cut |
| `GET` | `/api/videos/{id}/topics` | Retrieves discovered topics with podcast metadata |
| `GET` | `/api/videos/{id}/clips` | Retrieves generated clips with durations and justifications |
| `GET` | `/api/videos/{id}/clip/{topic_id}` | Streams an individual topic clip |
| `GET` | `/api/videos/{id}/clip/{topic_id}/subtitles` | Serves synchronized WebVTT subtitles with speaker tags |
| `GET` | `/api/videos/{id}/stream` | Streams the full source video |
| `GET` | `/api/clips/{id}/notes` | Fetches cached or generated structured clip notes for an audience persona |
| `POST` | `/api/clips/{id}/notes` | Generates audience-specific notes (Education, Professional, Content Creator) |
| `GET` | `/api/clips/{id}/notes/tts` | Synthesizes and streams audio narration (TTS) of clip notes |
| `GET` | `/api/library/clips` | Retrieves saved clips for the user profile |
| `POST` | `/api/library/clips/save` | Permanently saves a clip to user dashboard |
| `POST` | `/api/auth/register` | Registers a new user account with Supabase / local auth |
| `POST` | `/api/auth/login` | Authenticates user and returns JWT token |

---

## 📁 Repository Structure

```
HR2-OI-6C86C871/
├── backend/
│   ├── config.py                     # Environment, Groq dual-key & Supabase configuration
│   ├── database.py                   # SQLite schema, dual persistence & migrations
│   ├── main.py                       # FastAPI entrypoint, router mounts & static serving
│   ├── migrations/                   # SQL migration scripts for Supabase / PostgreSQL
│   │   ├── 001_create_supabase_schema.sql
│   │   ├── 002_user_auth_and_profiles.sql
│   │   ├── 003_secure_user_ownership_rls.sql
│   │   └── 004_add_audience_mode_and_clip_notes.sql
│   ├── models/
│   │   └── schemas.py                # Pydantic schemas (ConversationalTurn, PodcastExchangeCandidate)
│   ├── routers/
│   │   ├── auth.py                   # Authentication & user profile router
│   │   ├── audience.py               # Audience presets and mode selection router
│   │   ├── library.py                # Permanent user clip library & dashboard
│   │   ├── pipeline.py               # Ingestion and processing pipeline router
│   │   └── videos.py                 # Core video analysis, topic discovery & clip endpoints
│   └── services/
│       ├── ai_providers.py           # Provider abstractions & stable vector embeddings
│       ├── audio_extractor.py        # Audio extraction & <= 10min chunking
│       ├── auth_service.py           # Token validation & user session management
│       ├── diarization_service.py    # Multi-modal acoustic & conversational speaker diarization
│       ├── groq_stt.py               # Groq Whisper Large-v3 STT with chunk concurrency
│       ├── meaning_validator.py      # 10-point editorial validation & multi-model fallback
│       ├── notes_service.py          # Audience-aware clip notes generation & TTS audio
│       ├── podcast_intelligence.py   # Turn segmentation, 5 exchange archetypes & trimming
│       ├── supabase_service.py       # Supabase client, storage buckets & RLS helpers
│       ├── topic_intelligence.py     # Hybrid topic discovery, graph scoring & moment selection
│       └── video_cutter.py           # FFmpeg boundary cutting & WebVTT subtitle generator
├── frontend/
│   ├── css/
│   │   └── style.css                 # Glassmorphic dark design system & podcast badges
│   ├── js/
│   │   └── app.js                    # Web Speech API, studio player & playlist controller
│   └── index.html                    # Responsive Vidara interface & modal controls
├── test_vidara.py                    # Core platform integration test suite
├── test_intelligent_trimming.py      # Boundary trimming & meaning preservation tests
├── test_podcast_intelligence.py      # 12 podcast conversational edge case tests
├── test_audience_notes_integration.py # Audience mode & clip notes tests
├── test_auth_integration.py          # Auth & multi-tenant isolation tests
├── test_supabase_migration.py        # Supabase schema & storage tests
├── Dockerfile                        # Production Docker container build
├── render.yaml                       # Cloud deployment configuration for Render
├── requirements.txt                  # Python dependencies
├── .env.example                      # Configuration template
├── .gitignore                        # Git ignore rules (secrets, media, caches excluded)
└── README.md                         # Project documentation
```

---

## 🐳 Deployment

### Docker
```bash
docker build -t vidara-ai .
docker run -p 8000:8000 --env-file .env vidara-ai
```

### Cloud Deployment (Render)
A `render.yaml` configuration is included for zero-downtime deployment:
1. Connect this GitHub repository to Render.
2. Configure environment variables (`GROQ_API_KEY`, `VIDEO_ANALYSIS_API_KEY`, `REASONING_API_KEY`).
3. Deploy directly as a web service.

---

## 👥 Authors & Team

**Team Knight Coders**  
*HACKERING 2.0 (Round 2) — Voice AI Track*  
*Project ID: HR2-OI-6C86C871*

Built with ❤️ using FastAPI, Groq LPUs, Whisper Large-v3, Supabase, and FFmpeg.
