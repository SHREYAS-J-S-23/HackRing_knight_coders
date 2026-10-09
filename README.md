# 🧠 VIDARA — AI Video & Podcast Intelligence Platform
> *"Understand Every Moment."*  
> **Autonomous Long-Form Video Comprehension, Speaker Diarization, Podcast Intelligence & Semantic Extraction Engine**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq LPU](https://img.shields.io/badge/AI%20Hardware-Groq%20LPU-f55036.svg?style=flat)](https://groq.com/)
[![Whisper Large-v3](https://img.shields.io/badge/STT-Whisper%20Large--v3-blue.svg?style=flat)](https://github.com/openai/whisper)
[![FFmpeg](https://img.shields.io/badge/Video%20Assembly-FFmpeg-green.svg?style=flat&logo=ffmpeg)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/Tests-34%20Passing-brightgreen.svg?style=flat)]()

---

## 🌟 Overview

**Vidara AI** is a state-of-the-art video and podcast intelligence platform built for long-form video comprehension (1–2hr+ podcasts, keynotes, university lectures, and founder interviews). It solves the fundamental problem of traditional video clipping tools: extracting clips that are either too long, filled with rambling banter, or missing the critical question that gives a guest's answer its meaning.

Vidara continuously analyzes conversational turn dynamics, speaker acoustics, and topical graphs to deterministically extract the **shortest complete and meaningful clip** preserving full editorial context without syllable clipping.

---

## ⚡ Key Capabilities

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

### 4. 🎯 Soft Duration Targets & Boundary Trimming
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

### 6. 📝 Synchronized WebVTT Subtitles with Speaker Attribution
Generates WebVTT subtitles containing `<v SPEAKER_XX>Dialogue text</v>` speaker styling tags, synchronized with the video player and available for instant download.

---

## 🏗️ Pipeline Architecture

```
Raw Long-Form Video / URL (1-2hr+)
       │
       ▼
[AudioExtractor] ──► 16kHz Mono Audio Chunks (<= 10 mins bypasses Groq 25MB limit)
       │
       ▼
[GroqSTTService] ──► Word-Level Timestamps (Whisper Large-v3)
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

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/SHREYAS-J-S-23/HackRing_knight_coders.git
cd HackRing_knight_coders
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

Vidara includes 34 comprehensive tests verifying core systems, intelligent boundary trimming, and podcast edge cases:

```bash
# Run the entire test suite
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
| `POST` | `/api/videos/ingest-url` | Ingests public video URL (YouTube, Vimeo, direct MP4) |
| `POST` | `/api/videos/{id}/analyze` | Chunks audio, transcribes with Whisper, and builds index |
| `POST` | `/api/videos/{id}/discover-topics` | Autonomous podcast moment discovery & boundary ranking |
| `POST` | `/api/videos/{id}/query` | Natural language / voice semantic retrieval |
| `POST` | `/api/videos/{id}/generate-clips` | Renders validated standalone topic clips with subtitles |
| `POST` | `/api/videos/{id}/merge` | Deterministically stitches selected clips into a master cut |
| `GET` | `/api/videos/{id}/topics` | Retrieves discovered topics with podcast metadata |
| `GET` | `/api/videos/{id}/clips` | Retrieves generated clips with durations and justifications |
| `GET` | `/api/videos/{id}/clip/{topic_id}` | Streams an individual topic clip |
| `GET` | `/api/videos/{id}/clip/{topic_id}/subtitles` | Serves synchronized WebVTT subtitles with speaker tags |
| `GET` | `/api/videos/{id}/stream` | Streams the full source video |
| `GET` | `/api/library/clips` | Retrieves saved clips for the user profile |
| `POST` | `/api/library/clips/save` | Permanently saves a clip to user dashboard |

---

## 📁 Repository Structure

```
HackRing_knight_coders/
├── backend/
│   ├── config.py                     # Dual-key environment & model configuration
│   ├── database.py                   # SQLite schema, migrations & persistence layer
│   ├── main.py                       # FastAPI entrypoint, router mounts & static serving
│   ├── models/
│   │   └── schemas.py                # Pydantic schemas (ConversationalTurn, PodcastExchangeCandidate)
│   ├── routers/
│   │   ├── auth.py                   # Phone OTP & Google sign-in authentication
│   │   ├── library.py                # Permanent user clip library & dashboard
│   │   ├── videos.py                 # Core video analysis, topic discovery & clip endpoints
│   │   ├── pipeline.py               # Preserved legacy pipeline router
│   │   └── audience.py               # Preserved audience presets router
│   └── services/
│       ├── audio_extractor.py        # Audio extraction & <= 10min chunking
│       ├── groq_stt.py               # Groq Whisper Large-v3 STT with chunk concurrency
│       ├── diarization_service.py    # Multi-modal acoustic & conversational speaker diarization
│       ├── podcast_intelligence.py   # Turn segmentation, 5 exchange archetypes & trimming
│       ├── topic_intelligence.py     # Hybrid topic discovery, graph scoring & moment selection
│       ├── meaning_validator.py      # 10-point editorial validation & multi-model fallback
│       ├── ai_providers.py           # Provider abstractions & stable vector embeddings
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
├── requirements.txt                  # Python dependencies
├── .env.example                      # Configuration template
├── .gitignore                        # Git ignore rules (secrets, media, caches excluded)
└── README.md                         # Project documentation
```

---

## 👥 Authors & Team

**Knight Coders** — HackRing Hackathon  
- Built with ❤️ using FastAPI, Groq LPUs, Whisper, and FFmpeg.
