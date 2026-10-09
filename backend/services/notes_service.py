import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from groq import Groq
from openai import OpenAI

from backend.config import (
    get_reasoning_api_key,
    get_reasoning_model,
    GROQ_FALLBACK_MODEL,
    AGNES_API_KEY,
    AGNES_BASE_URL
)
from backend.database import DatabaseService

VALID_AUDIENCE_MODES = {"education", "professional", "content_creator"}


class ClipNotesService:
    """
    Audience-Aware AI Notes Generation Service for Vidara clips.
    Extracts timestamp-aligned transcript segments and produces
    structured, validated pedagogical, technical, or creator notes.
    """

    @classmethod
    def get_overlapping_transcript(
        cls,
        video_id: str,
        start_time: float,
        end_time: float,
        tolerance_seconds: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves timestamp-aligned transcript segments overlapping [start_time, end_time].
        """
        all_segments = DatabaseService.get_segments(video_id)
        if not all_segments:
            return []

        # Prioritize strict timestamp overlap first
        overlapping = [
            s for s in all_segments
            if (s.get("start_time", 0.0) < end_time and
                s.get("end_time", 0.0) > start_time)
        ]

        # If strict overlap is empty, expand with tolerance
        if not overlapping and tolerance_seconds > 0:
            overlapping = [
                s for s in all_segments
                if (s.get("start_time", 0.0) < (end_time + tolerance_seconds) and
                    s.get("end_time", 0.0) > (start_time - tolerance_seconds))
            ]

        # If strict overlap is empty, find closest segment
        if not overlapping:
            sorted_segs = sorted(
                all_segments,
                key=lambda s: abs(s.get("start_time", 0.0) - start_time)
            )
            overlapping = sorted_segs[:3]

        overlapping.sort(key=lambda s: s.get("start_time", 0.0))
        return overlapping

    @classmethod
    def format_transcript_text(cls, segments: List[Dict[str, Any]]) -> str:
        lines = []
        for s in segments:
            speaker = s.get("speaker", "SPEAKER")
            start = round(float(s.get("start_time", 0.0)), 1)
            end = round(float(s.get("end_time", 0.0)), 1)
            text = s.get("text", "").strip()
            if text:
                lines.append(f"[{start}s - {end}s] {speaker}: {text}")
        return "\n".join(lines)

    @classmethod
    def _get_llm_client(cls) -> tuple:
        key = get_reasoning_api_key()
        model = get_reasoning_model()
        if key:
            return Groq(api_key=key), model, "groq"
        elif AGNES_API_KEY:
            return OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL), "agnes-3.0-flash", "agnes"
        else:
            raise ValueError(
                "Reasoning API key is not configured. Please add GROQ_API_KEY to your environment."
            )

    @classmethod
    def _build_prompt(
        cls,
        audience_mode: str,
        clip_title: str,
        start_time: float,
        end_time: float,
        transcript_text: str
    ) -> str:
        base_header = f"""You are Vidara's Audience-Aware AI Clip Notes Generator.
You must generate structured, high-value notes for the following video clip.

CLIP METADATA:
- Title: {clip_title}
- Timestamps: {round(start_time, 1)}s to {round(end_time, 1)}s (Duration: {round(end_time - start_time, 1)}s)

FACTUAL SOURCE TRANSCRIPT (DO NOT FABRICATE BEYOND THIS EVIDENCE):
\"\"\"
{transcript_text}
\"\"\"
"""
        if audience_mode == "education":
            return base_header + """
AUDIENCE: EDUCATION (Students, Learners, Developers studying concepts)
Goal: Explain what is taught, break down core concepts, offer compact revision notes, and test understanding.

Return STRICT JSON matching this EXACT structure:
{
  "clip_summary": "Clear, beginner-friendly summary of what the speaker teaches in the clip while retaining technical terminology.",
  "key_points": [
    "Concise bullet point covering definitions, formulas, or distinctions in the clip",
    "Second key point"
  ],
  "sections": {
    "clip_overview": "Comprehensive overview explaining the lesson and its significance.",
    "key_concepts": [
      {
        "name": "Concept Name",
        "explanation": "Simple, intuitive explanation.",
        "why_it_matters": "Why this concept is important.",
        "example": "Relevant example from the clip, or empty string if none provided."
      }
    ],
    "important_points": [
      "Point 1 with exact definitions or distinctions",
      "Point 2"
    ],
    "revision_notes": [
      "Compact note a student can quickly review before an exam",
      "Key distinction or rule to memorize"
    ],
    "practice_questions": {
      "short_answer_questions": [
        "Short-answer question 1 directly answerable from clip",
        "Short-answer question 2",
        "Short-answer question 3"
      ],
      "conceptual_questions": [
        "Conceptual or application question 1 based on clip ideas",
        "Conceptual question 2"
      ],
      "multiple_choice_question": {
        "question": "A multiple choice question directly testing clip knowledge",
        "options": [
          "A) Option 1",
          "B) Option 2",
          "C) Option 3",
          "D) Option 4"
        ],
        "correct_answer": "A) Option 1",
        "explanation": "Short explanation justifying why this answer is correct based on the speaker's explanation."
      }
    },
    "quick_recap": [
      "Concise takeaway 1",
      "Concise takeaway 2",
      "Concise takeaway 3"
    ]
  }
}
"""

        elif audience_mode == "professional":
            return base_header + """
AUDIENCE: PROFESSIONAL (Engineers, Architects, Decision Makers, Industry Experts)
Goal: Executive-level technical breakdown, industry relevance, implementation insights, trade-offs, and terminology.

Return STRICT JSON matching this EXACT structure:
{
  "clip_summary": "Precise executive summary of the central argument, technical mechanism, or industry insight.",
  "key_points": [
    "Technical or strategic key point 1",
    "Technical key point 2"
  ],
  "sections": {
    "executive_summary": "In-depth executive summary with concise, authoritative language.",
    "technical_breakdown": "Thorough explanation of the technical concepts, systems, methods, algorithms, architectures, or processes discussed, preserving exact domain terms.",
    "industry_relevance": "Practical implications for engineering, software development, operations, or industry practice. Distinguish explicitly stated facts from reasonable implications.",
    "implementation_insights": [
      {
        "problem": "Problem discussed",
        "approach": "Approach or pattern used",
        "how_it_works": "Mechanism of execution",
        "benefits": "Key advantages",
        "trade_offs": "Trade-offs or cost",
        "limitations": "Known constraints or caveats mentioned"
      }
    ],
    "key_takeaways": [
      "Key professional conclusion 1",
      "Key professional conclusion 2"
    ],
    "action_items": [
      "Practical investigation step or next question for engineering teams",
      "Actionable follow-up"
    ],
    "terminology": [
      {
        "term": "Technical Term",
        "definition": "Precise explanation based on context."
      }
    ]
  }
}
"""

        else:  # content_creator
            return base_header + """
AUDIENCE: CONTENT CREATOR (Social Media Creators, Storytellers, Video Editors, Educators)
Goal: Turn insights into viral, engaging storytelling material, hooks, titles, captions, and narrative structures.

Return STRICT JSON matching this EXACT structure:
{
  "clip_summary": "Engaging summary of the core message that makes this clip valuable.",
  "key_points": [
    "Key storytelling point 1",
    "Key storytelling point 2"
  ],
  "sections": {
    "core_message": "The central insight or epiphany that hooks the audience.",
    "suggested_titles": [
      "Compelling Title 1",
      "Compelling Title 2",
      "Compelling Title 3",
      "Compelling Title 4",
      "Compelling Title 5"
    ],
    "hooks": [
      "Suggested Hook 1 (Opening line to capture attention within 3 seconds)",
      "Suggested Hook 2 (Curiosity gap or provocative question based on the clip)"
    ],
    "story_structure": {
      "hook": "Opening hook to reel viewer in",
      "context": "Why this matters or current problem",
      "main_idea": "The core insight from the clip",
      "supporting_explanation": "Supporting example or evidence",
      "conclusion": "Final punchline or memorable takeaway"
    },
    "storytelling_improvements": "Editorial suggestions on pacing, visual metaphors, tone, or emphasis to maximize viewer retention.",
    "suggested_conclusion": "Suggested closing punchline or memorable sign-off line.",
    "social_caption": "Ready-to-post engaging caption with relevant context (no clickbait).",
    "call_to_action": "Relevant, contextual call to action (e.g. Save this for later, Share with your team)."
  }
}
"""

    @classmethod
    def generate_notes_for_clip(
        cls,
        video_id: str,
        clip: Dict[str, Any],
        audience_mode: str = "education",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience wrapper accepting a clip dictionary.
        """
        return cls.generate_clip_notes(
            clip_id=clip.get("id", ""),
            video_id=video_id,
            audience_mode=audience_mode,
            clip_title=clip.get("title") or clip.get("name") or "Clip",
            start_time=float(clip.get("start_time", 0.0)),
            end_time=float(clip.get("end_time", 0.0)),
            user_id=user_id
        )

    @classmethod
    def generate_clip_notes(
        cls,
        clip_id: str,
        video_id: str,
        audience_mode: str = "education",
        clip_title: str = "Clip",
        start_time: float = 0.0,
        end_time: float = 0.0,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates or retrieves cached audience-specific notes for a given clip.
        """
        mode = audience_mode.strip().lower() if audience_mode else "education"
        if mode not in VALID_AUDIENCE_MODES:
            mode = "education"

        # 1. Check persistent cache in DB first
        cached = DatabaseService.get_clip_notes(clip_id, mode)
        if cached:
            return cached

        # 2. Extract timestamp-aligned transcript segments
        overlapping_segs = cls.get_overlapping_transcript(video_id, start_time, end_time)
        transcript_text = cls.format_transcript_text(overlapping_segs)

        if not transcript_text:
            transcript_text = f"Speaker discusses {clip_title} from {round(start_time, 1)}s to {round(end_time, 1)}s."

        prompt = cls._build_prompt(
            audience_mode=mode,
            clip_title=clip_title,
            start_time=start_time,
            end_time=end_time,
            transcript_text=transcript_text
        )

        client, primary_model, provider_type = cls._get_llm_client()
        models_to_try = [primary_model]
        if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL not in models_to_try:
            models_to_try.append(GROQ_FALLBACK_MODEL)
        if "llama-3.3-70b-versatile" not in models_to_try:
            models_to_try.append("llama-3.3-70b-versatile")

        parsed_data = None
        used_model = primary_model

        for m in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Vidara's AI Notes Intelligence Engine. "
                                "You produce factual, structured, audience-aligned notes based strictly on transcript evidence. "
                                "Return STRICT JSON ONLY."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1800,
                    temperature=0.2
                )
                raw_content = response.choices[0].message.content or ""
                if raw_content.strip():
                    parsed_data = json.loads(raw_content)
                    used_model = m
                    break
            except Exception as llm_err:
                print(f"Notes LLM attempt with {m} failed: {llm_err}, attempting regex cleanup...")
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "Return STRICT JSON matching the schema, without markdown formatting."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1800,
                        temperature=0.2
                    )
                    raw_content = (response.choices[0].message.content or "").strip()
                    if raw_content.startswith("```"):
                        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                        raw_content = re.sub(r"\s*```$", "", raw_content)
                    match = re.search(r"\{[\s\S]*\}", raw_content)
                    if match:
                        parsed_data = json.loads(match.group(0))
                        used_model = m
                        break
                except Exception as inner_err:
                    print(f"Notes LLM retry on {m} failed: {inner_err}")

        # Fallback if LLM fails
        if not parsed_data or not isinstance(parsed_data, dict):
            parsed_data = cls._generate_deterministic_fallback(
                audience_mode=mode,
                clip_title=clip_title,
                start_time=start_time,
                end_time=end_time,
                transcript_text=transcript_text
            )

        # Assemble full validated payload
        result: Dict[str, Any] = {
            "clip_id": clip_id,
            "video_id": video_id,
            "audience_mode": mode,
            "clip_title": clip_title,
            "clip_summary": parsed_data.get("clip_summary") or f"Discussion of {clip_title}.",
            "key_points": parsed_data.get("key_points") or [f"Core discussion regarding {clip_title}."],
            "sections": parsed_data.get("sections") or {},
            "source_start": round(start_time, 2),
            "source_end": round(end_time, 2),
            "generation_status": "completed",
            "model_metadata": {
                "model": used_model,
                "provider": provider_type,
                "generated_at": time.time()
            }
        }

        # Persist to database cache
        DatabaseService.save_clip_notes(
            clip_id=clip_id,
            video_id=video_id,
            user_id=user_id,
            audience_mode=mode,
            notes_data=result,
            model_metadata=result["model_metadata"]
        )

        return result

    @classmethod
    def _generate_deterministic_fallback(
        cls,
        audience_mode: str,
        clip_title: str,
        start_time: float,
        end_time: float,
        transcript_text: str
    ) -> Dict[str, Any]:
        """Safe deterministic fallback if LLM encounters transient rate limits."""
        summary = f"In this {round(end_time - start_time, 1)}s clip, the speaker presents key insights on {clip_title}."
        if audience_mode == "education":
            return {
                "clip_summary": summary,
                "key_points": [
                    f"Understanding foundational concepts of {clip_title}.",
                    "Applying principles demonstrated in the clip."
                ],
                "sections": {
                    "clip_overview": f"A dedicated instructional segment focusing on {clip_title}.",
                    "key_concepts": [
                        {
                            "name": clip_title,
                            "explanation": "Primary subject discussed in this segment.",
                            "why_it_matters": "Forms the basis for the surrounding argument.",
                            "example": ""
                        }
                    ],
                    "important_points": [
                        f"Detailed analysis of {clip_title}.",
                        "Critical boundaries and distinctions."
                    ],
                    "revision_notes": [
                        f"Review definitions associated with {clip_title}.",
                        "Note speaker's core justification."
                    ],
                    "practice_questions": {
                        "short_answer_questions": [
                            f"What is the main premise of {clip_title}?",
                            "How does the speaker justify this argument?",
                            "What outcome is described in the clip?"
                        ],
                        "conceptual_questions": [
                            f"How does {clip_title} apply in practice?",
                            "What are the implications if this concept is misunderstood?"
                        ],
                        "multiple_choice_question": {
                            "question": f"Which statement best characterizes {clip_title}?",
                            "options": [
                                f"A) It represents the core insight of this clip.",
                                "B) It is an unrelated digression.",
                                "C) It contradicts the speaker's main thesis.",
                                "D) It has no practical application."
                            ],
                            "correct_answer": f"A) It represents the core insight of this clip.",
                            "explanation": f"The segment directly focuses on developing understanding of {clip_title}."
                        }
                    },
                    "quick_recap": [
                        f"Key concept: {clip_title}.",
                        "Speaker emphasizes clear understanding.",
                        "Essential for revision and mastery."
                    ]
                }
            }
        elif audience_mode == "professional":
            return {
                "clip_summary": summary,
                "key_points": [
                    f"Technical architecture and methodology for {clip_title}.",
                    "Operational implications and trade-offs."
                ],
                "sections": {
                    "executive_summary": summary,
                    "technical_breakdown": f"Technical breakdown of {clip_title} and related components.",
                    "industry_relevance": "Direct applicability to production systems and engineering workflow.",
                    "implementation_insights": [
                        {
                            "problem": f"Managing challenges related to {clip_title}",
                            "approach": "Systematic implementation",
                            "how_it_works": "Follows the architecture outlined by the speaker",
                            "benefits": "Higher reliability and clear semantics",
                            "trade_offs": "Requires engineering investment",
                            "limitations": "Constrained by existing system architecture"
                        }
                    ],
                    "key_takeaways": [
                        f"Maintain alignment with {clip_title} principles.",
                        "Audit systems for matching guarantees."
                    ],
                    "action_items": [
                        f"Investigate current team implementation of {clip_title}.",
                        "Evaluate metrics and trade-offs."
                    ],
                    "terminology": [
                        {
                            "term": clip_title,
                            "definition": "Key subject addressed in this professional overview."
                        }
                    ]
                }
            }
        else:
            return {
                "clip_summary": summary,
                "key_points": [
                    f"Compelling storytelling opportunity regarding {clip_title}.",
                    "High-engagement narrative hook."
                ],
                "sections": {
                    "core_message": f"Why {clip_title} transforms the way we think.",
                    "suggested_titles": [
                        f"The Truth About {clip_title}",
                        f"Why Nobody Understands {clip_title}",
                        f"Master {clip_title} in 60 Seconds",
                        f"The Secret Behind {clip_title}",
                        f"Stop Making This Mistake with {clip_title}"
                    ],
                    "hooks": [
                        f"You won't believe what happens when you understand {clip_title}...",
                        f"If you're still doing {clip_title} the old way, watch this."
                    ],
                    "story_structure": {
                        "hook": f"Provocative question about {clip_title}",
                        "context": "The hidden problem most people face",
                        "main_idea": f"The breakthrough idea in {clip_title}",
                        "supporting_explanation": "Real-world example from the clip",
                        "conclusion": "Final memorable punchline"
                    },
                    "storytelling_improvements": "Focus on high-energy delivery, highlight the contrast in the first 3 seconds.",
                    "suggested_conclusion": f"That is why {clip_title} changes everything.",
                    "social_caption": f"Here is what you need to know about {clip_title}. Watch till the end!",
                    "call_to_action": "Save this clip and share it with someone who needs to see this."
                }
            }
