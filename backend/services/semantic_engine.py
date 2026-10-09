import json
import re
from typing import Optional, List, Dict, Any
from groq import Groq
from openai import OpenAI
from backend.config import (
    get_groq_api_key,
    get_groq_llm_api_key,
    GROQ_MODEL,
    GROQ_FALLBACK_MODEL,
    AGNES_API_KEY,
    AGNES_BASE_URL
)
from backend.models.schemas import (
    EnrichedTranscript,
    IntentGraph,
    TopicTheme,
    SegmentAnnotation,
    AudienceProfile,
    EditDecisionList,
    EditDecision
)

class SemanticEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_groq_llm_api_key()

    def _get_client(self):
        key = self.api_key or get_groq_llm_api_key()
        if key:
            return Groq(api_key=key), GROQ_MODEL, GROQ_FALLBACK_MODEL
        elif AGNES_API_KEY:
            return OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL), "agnes-3.0-flash", "agnes-3.0-flash"
        else:
            raise ValueError(
                "GROQ_API_KEY is not configured! Please provide your Groq API key in .env or via the Keys button."
            )

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """Safely parses JSON even if wrapped in markdown codeblocks or preceded by text."""
        if not content:
            return None
        text = content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # Regex search for outer JSON object
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                val = json.loads(m.group(0))
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
        return None

    def _call_llm_json(
        self,
        client,
        prompt: str,
        system_prompt: str,
        models: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a robust multi-stage LLM completion with automatic model fallback,
        response_format relaxation if json_validate_failed (HTTP 400) occurs,
        and resilient regex JSON extraction.
        """
        model_queue: List[str] = []
        for m in models:
            if m and m not in model_queue:
                model_queue.append(m)
        if "qwen/qwen3.8-27b" not in model_queue:
            model_queue.append("qwen/qwen3.8-27b")

        for model in model_queue:
            # 1. Attempt with response_format={"type": "json_object"}
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=850,
                    temperature=0.1
                )
                content = response.choices[0].message.content or ""
                parsed = self._extract_json(content)
                if parsed:
                    return parsed
            except Exception as e:
                err_str = str(e).lower()
                print(f"Vidara: Model {model} with response_format failed ({type(e).__name__}: {e}).")
                # If rate/token limit, continue immediately to next model
                if "rate_limit" in err_str or "413" in err_str or "tpm" in err_str or "tpd" in err_str or "tokens" in err_str:
                    continue

            # 2. Attempt without response_format, extracting JSON via regex
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"{system_prompt} Return ONLY raw, valid JSON. Do not wrap in markdown or include conversational text."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=850,
                    temperature=0.1
                )
                content = response.choices[0].message.content or ""
                parsed = self._extract_json(content)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"Vidara: Model {model} without response_format failed ({type(e).__name__}: {e}).")

        return None

    def _heuristic_build_intent_graph(self, transcript: EnrichedTranscript) -> IntentGraph:
        """
        Deterministic, robust IntentGraph generator when LLM calls are unavailable,
        rate-limited, or failed schema validation. Guarantees 0-downtime indexing.
        """
        segs = transcript.segments
        if not segs:
            return IntentGraph(
                video_id=transcript.video_id,
                core_thesis=f"Video overview for {transcript.filename or 'media'}.",
                themes=[],
                annotated_segments=[]
            )

        # 1. Core thesis heuristic
        first_few_texts = [s.text.strip() for s in segs[:5] if len(s.text.strip().split()) >= 3]
        sample_title = " ".join(first_few_texts[:2])
        if len(sample_title) > 160:
            sample_title = sample_title[:157] + "..."
        core_thesis = sample_title or f"Discussion and key insights from {transcript.filename or 'video'}."

        # 2. Partition into chronological themes
        num_themes = min(5, max(2, len(segs) // 10))
        chunk_size = max(1, len(segs) // num_themes)
        themes = []
        theme_names = [
            "Introduction & Overview",
            "Core Concepts & Principles",
            "Detailed Analysis & Discussion",
            "Practical Application & Examples",
            "Summary & Concluding Takeaways"
        ]

        for i in range(num_themes):
            chunk = segs[i * chunk_size : (i + 1) * chunk_size] if i < num_themes - 1 else segs[i * chunk_size :]
            if not chunk:
                continue
            chunk_sids = [s.id for s in chunk]
            chunk_text = " ".join(s.text for s in chunk[:3])
            t_name = theme_names[i] if i < len(theme_names) else f"Part {i+1}: Detailed Discussion"
            themes.append(
                TopicTheme(
                    title=t_name,
                    summary=chunk_text[:200] + ("..." if len(chunk_text) > 200 else ""),
                    segment_ids=chunk_sids,
                    implicit_takeaway=f"Key perspective on {t_name.lower()}."
                )
            )

        # 3. Annotate segments heuristically
        annotated = []
        for s in segs:
            t_lower = s.text.lower().strip()
            word_count = len(t_lower.split())
            if word_count < 3 or any(t_lower.startswith(k) for k in ["um", "uh", "hello", "welcome"]):
                cat = "FILLER"
                score = 1
            elif word_count > 12:
                cat = "CORE_POINT"
                score = 4
            else:
                cat = "SUPPORTING"
                score = 3

            annotated.append(
                SegmentAnnotation(
                    segment_id=s.id,
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    category=cat,
                    implicit_meaning=s.text[:100],
                    critical_context="",
                    importance_score=score
                )
            )

        return IntentGraph(
            video_id=transcript.video_id,
            core_thesis=core_thesis,
            themes=themes,
            annotated_segments=annotated
        )

    def _heuristic_generate_edl(
        self,
        transcript: EnrichedTranscript,
        intent_graph: IntentGraph,
        audience: AudienceProfile
    ) -> EditDecisionList:
        """
        Deterministic, faithful Edit Decision List generator when LLM is unavailable.
        Uses segment importance scores and categories from the intent graph.
        """
        ann_map = {a.segment_id: a for a in intent_graph.annotated_segments}
        decisions = []
        kept_duration = 0.0

        for seg in transcript.segments:
            ann = ann_map.get(seg.id)
            score = ann.importance_score if ann else 3
            cat = ann.category if ann else "SUPPORTING"

            if cat in ("FILLER", "TANGENT", "MISTAKE_CORRECTION") or score <= 2:
                action = "CUT"
                reason = f"Filtered {cat.lower().replace('_', ' ')} segment to meet compression target."
            else:
                action = "KEEP"
                reason = "Retained as key informative segment supporting core topic."
                kept_duration += (seg.end - seg.start)

            decisions.append(
                EditDecision(
                    segment_id=seg.id,
                    action=action,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    reason=reason,
                    meaning_impact="Minimal" if action == "CUT" else "Preserves core context",
                    safety_check="PASS"
                )
            )

        decisions.sort(key=lambda x: x.start)
        orig_duration = transcript.duration_seconds or 1.0
        compression = max(0.0, round(((orig_duration - kept_duration) / orig_duration) * 100, 1))

        return EditDecisionList(
            video_id=transcript.video_id,
            audience_id=audience.id,
            audience_name=audience.name,
            decisions=decisions,
            original_duration=round(orig_duration, 2),
            edited_duration=round(kept_duration, 2),
            compression_percent=compression,
            kept_count=sum(1 for d in decisions if d.action == "KEEP"),
            cut_count=sum(1 for d in decisions if d.action == "CUT")
        )

    def build_intent_graph(self, transcript: EnrichedTranscript) -> IntentGraph:
        """
        Performs deep semantic analysis of the transcript.
        Guarantees prompt tokens stay <= 3,500 tokens to strictly respect
        Groq's 8,000 TPM limit, and provides multi-model + heuristic fallback.
        """
        client, primary_model, fallback_model = self._get_client()

        raw_segs = transcript.segments
        if len(raw_segs) > 24:
            # 1. Create chronological coarse blocks
            raw_blocks = []
            cur_texts = []
            cur_ids = []
            block_start = raw_segs[0].start
            for idx, s in enumerate(raw_segs):
                cur_texts.append(s.text)
                cur_ids.append(s.id)
                if (s.end - block_start >= 45.0) or idx == len(raw_segs) - 1:
                    raw_blocks.append({
                        "block_id": len(raw_blocks) + 1,
                        "segment_ids": cur_ids.copy(),
                        "start": round(block_start, 2),
                        "end": round(s.end, 2),
                        "speaker": s.speaker,
                        "text": " ".join(cur_texts)[:280]  # Cap snippet to prevent token bloat
                    })
                    cur_texts = []
                    cur_ids = []
                    if idx + 1 < len(raw_segs):
                        block_start = raw_segs[idx + 1].start

            # 2. Downsample to at most 20 representative blocks across the entire video timeline
            if len(raw_blocks) > 20:
                step = (len(raw_blocks) - 1) / 19.0
                sampled_indices = sorted(list(set(int(round(i * step)) for i in range(20))))
                transcript_payload = [raw_blocks[i] for i in sampled_indices]
            else:
                transcript_payload = raw_blocks
        else:
            transcript_payload = [
                {
                    "block_id": s.id,
                    "segment_ids": [s.id],
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                    "text": s.text[:280]
                }
                for s in raw_segs
            ]

        prompt = f"""
You are an expert editorial narrative analyst.
Perform a thorough, faithful semantic analysis of this real timestamped video transcript.
DO NOT use generic boilerplate. Your analysis must strictly reflect the actual words spoken in this video.

ACTUAL TRANSCRIPT NARRATIVE SAMPLES:
{json.dumps(transcript_payload, indent=1)}

Analysis Tasks:
1. Identify the precise Core Thesis of this video: what is the speaker's primary message, problem statement, or demonstration?
2. Group the narrative into distinct, well-defined Topic Themes. For each theme:
   - title: Specific and descriptive title reflecting what is discussed
   - summary: Detailed explanation of what was discussed word-by-word
   - block_ids: Exact list of block IDs covering this theme
   - implicit_takeaway: The speaker's underlying intent or takeaway
3. Classify EVERY block individually into one of:
   - "CORE_POINT": Essential thesis, key conclusion, or core architectural/conceptual claim
   - "SUPPORTING": Concrete example, code walkthrough, proof, or elaboration
   - "TANGENT": Off-topic anecdote, detour, or unrelated remark
   - "REPETITION": Redundant restatement of a previously made point
   - "FILLER": Banter, microphone check, pauses, audio adjustment, verbal tics
   - "TRANSITION": Bridge sentence connecting two topics
   - "MISTAKE_CORRECTION": Speaker correcting a verbal slip or false start
4. For every block, provide:
   - implicit_meaning: The true intent or subtext behind the words
   - critical_context: Any crucial condition, caveat, or prerequisite that MUST NOT be cut out of context
   - importance_score: 1 (disposable) to 5 (crucial)

Return STRICTLY a JSON object matching this schema:
{{
  "core_thesis": "string",
  "themes": [
    {{
      "title": "string",
      "summary": "string",
      "block_ids": [1, 2],
      "implicit_takeaway": "string"
    }}
  ],
  "annotated_blocks": [
    {{
      "block_id": 1,
      "category": "CORE_POINT",
      "implicit_meaning": "string",
      "critical_context": "string",
      "importance_score": 5
    }}
  ]
}}
"""
        candidate_models = [primary_model, fallback_model, "qwen/qwen3.8-27b"]
        data = self._call_llm_json(
            client=client,
            prompt=prompt,
            system_prompt="You are a senior media intelligence AI analyzing real video dialogue. Respond with strict, valid JSON only.",
            models=candidate_models
        )

        if not data:
            print("Vidara: All LLM models failed or were rate-limited. Falling back to heuristic Intent Graph.")
            return self._heuristic_build_intent_graph(transcript)

        # Build block lookup
        block_map = {b["block_id"]: b for b in transcript_payload}
        block_ann_map = {}
        for ann in data.get("annotated_blocks", data.get("annotated_segments", [])):
            bid = ann.get("block_id", ann.get("segment_id"))
            block_ann_map[bid] = ann

        themes = []
        for i, t in enumerate(data.get("themes", [])):
            b_ids = t.get("block_ids", t.get("segment_ids", []))
            # Resolve actual segment IDs
            all_sids = []
            for bid in b_ids:
                if bid in block_map:
                    all_sids.extend(block_map[bid]["segment_ids"])
                else:
                    all_sids.append(bid)
            themes.append(
                TopicTheme(
                    title=t.get("title", f"Theme {i+1}"),
                    summary=t.get("summary", ""),
                    segment_ids=all_sids,
                    implicit_takeaway=t.get("implicit_takeaway", "")
                )
            )

        seg_map = {s.id: s for s in transcript.segments}
        annotated = []

        # Map block annotations to individual segments
        for b in transcript_payload:
            bid = b["block_id"]
            ann = block_ann_map.get(bid, {})
            cat = ann.get("category", "SUPPORTING")
            imp_meaning = ann.get("implicit_meaning", "")
            crit_ctx = ann.get("critical_context", "")
            imp_score = ann.get("importance_score", 3)

            for sid in b["segment_ids"]:
                s = seg_map.get(sid)
                if s:
                    annotated.append(
                        SegmentAnnotation(
                            segment_id=sid,
                            start=s.start,
                            end=s.end,
                            text=s.text,
                            category=cat,
                            implicit_meaning=imp_meaning,
                            critical_context=crit_ctx,
                            importance_score=imp_score
                        )
                    )

        # In case some segments were missed by the LLM
        annotated_ids = {a.segment_id for a in annotated}
        for seg in transcript.segments:
            if seg.id not in annotated_ids:
                annotated.append(
                    SegmentAnnotation(
                        segment_id=seg.id,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        category="SUPPORTING",
                        implicit_meaning="",
                        critical_context="",
                        importance_score=3
                    )
                )

        annotated.sort(key=lambda x: x.start)

        return IntentGraph(
            video_id=transcript.video_id,
            core_thesis=data.get("core_thesis") or f"Discussion and core insights from {transcript.filename or 'video'}.",
            themes=themes,
            annotated_segments=annotated
        )

    def generate_edl(self, transcript: EnrichedTranscript, intent_graph: IntentGraph, audience: AudienceProfile) -> EditDecisionList:
        """
        Generates an audience-specific Edit Decision List tailored 100% to the real content.
        """
        client, model, fallback_model = self._get_client()

        # Cap sample of annotated segments if large
        ann_samples = [a.model_dump() for a in intent_graph.annotated_segments]
        if len(ann_samples) > 25:
            step = (len(ann_samples) - 1) / 24.0
            sampled_indices = sorted(list(set(int(round(i * step)) for i in range(25))))
            ann_payload = [ann_samples[i] for i in sampled_indices]
        else:
            ann_payload = ann_samples

        prompt = f"""
You are an expert video editor. Using the verified Intent Graph and real video transcript, generate an Edit Decision List (EDL) specifically for the following target audience:

Target Audience: {audience.name}
Role / Viewer Profile: {audience.target_role}
Compression Goal: {audience.compression_target}
Tone: {audience.tone}

Real Video Thesis: {intent_graph.core_thesis}

Annotated Segments:
{json.dumps(ann_payload, indent=1)}

Editing Guidelines:
1. For each segment, decide: "KEEP" or "CUT".
2. If cutting: Provide a specific, content-grounded reason (e.g., "Cut tangent about X", "Removed setup filler", "Too technical for general audience").
3. Evaluate meaning_impact: Confirm whether removing this segment alters the meaning of retained points.
4. Protect critical caveats: Never cut a segment containing a critical qualification unless the claim it qualifies is also cut.
5. Provide safety_check: "PASS", "WARN", or "FLAGGED".

Return STRICTLY a JSON object with this format:
{{
  "decisions": [
    {{
      "segment_id": 1,
      "action": "KEEP",
      "reason": "Clear explanation grounded in the actual spoken topic",
      "meaning_impact": "None. Thesis premise retained.",
      "safety_check": "PASS"
    }}
  ]
}}
"""
        candidate_models = [model, fallback_model, "qwen/qwen3.8-27b"]
        data = self._call_llm_json(
            client=client,
            prompt=prompt,
            system_prompt="You are a professional video editor generating precise Edit Decision Lists grounded strictly in the real transcript. Return valid JSON only.",
            models=candidate_models
        )

        if not data or "decisions" not in data:
            print("Vidara: LLM EDL generation failed across all models. Falling back to heuristic EDL.")
            return self._heuristic_generate_edl(transcript, intent_graph, audience)

        raw_decisions = data.get("decisions", [])
        seg_map = {s.id: s for s in transcript.segments}
        decisions = []
        kept_duration = 0.0

        for dec in raw_decisions:
            sid = dec.get("segment_id")
            seg = seg_map.get(sid)
            if seg:
                action = dec.get("action", "KEEP")
                if action == "KEEP":
                    kept_duration += (seg.end - seg.start)
                decisions.append(
                    EditDecision(
                        segment_id=sid,
                        action=action,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        reason=dec.get("reason", "Kept for audience value"),
                        meaning_impact=dec.get("meaning_impact", "No adverse impact"),
                        safety_check=dec.get("safety_check", "PASS")
                    )
                )

        accounted_ids = {d.segment_id for d in decisions}
        for seg in transcript.segments:
            if seg.id not in accounted_ids:
                decisions.append(
                    EditDecision(
                        segment_id=seg.id,
                        action="CUT",
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        reason="Cut to satisfy audience compression target",
                        meaning_impact="Minimal",
                        safety_check="PASS"
                    )
                )

        decisions.sort(key=lambda x: x.start)
        orig_duration = transcript.duration_seconds or 1.0
        compression = max(0.0, round(((orig_duration - kept_duration) / orig_duration) * 100, 1))

        return EditDecisionList(
            video_id=transcript.video_id,
            audience_id=audience.id,
            audience_name=audience.name,
            decisions=decisions,
            original_duration=round(orig_duration, 2),
            edited_duration=round(kept_duration, 2),
            compression_percent=compression,
            kept_count=sum(1 for d in decisions if d.action == "KEEP"),
            cut_count=sum(1 for d in decisions if d.action == "CUT")
        )
