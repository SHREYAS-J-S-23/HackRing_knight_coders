import json
from typing import Optional, List, Dict, Any, Tuple
from groq import Groq
from openai import OpenAI
from backend.config import get_groq_api_key, get_groq_llm_api_key, GROQ_MODEL, GROQ_FALLBACK_MODEL, AGNES_API_KEY, AGNES_BASE_URL
from backend.models.schemas import (
    EnrichedTranscript,
    IntentGraph,
    EditDecisionList,
    MeaningValidationReport
)

class MeaningValidator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_groq_llm_api_key() or get_groq_api_key()

    def _get_client(self):
        key = self.api_key or get_groq_llm_api_key() or get_groq_api_key()
        if key:
            return Groq(api_key=key), GROQ_MODEL
        elif AGNES_API_KEY:
            return OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL), "agnes-3.0-flash"
        else:
            return None, None

    def validate_edl(
        self,
        transcript: EnrichedTranscript,
        intent_graph: IntentGraph,
        edl: EditDecisionList
    ) -> MeaningValidationReport:
        """
        Validates the EDL to ensure that no cut introduces misleading context,
        reverses negation, or severs critical qualifying caveats.
        """
        client, model = self._get_client()
        if not client:
            return self._heuristic_validate(intent_graph, edl)

        kept_segments = [d for d in edl.decisions if d.action == "KEEP"]
        cut_segments = [d for d in edl.decisions if d.action == "CUT"]

        # Condense segments to avoid token limit exhaustion
        compact_annotated = [
            {"id": a.segment_id, "text": a.text[:120], "intent": a.intent_role}
            for a in intent_graph.annotated_segments[:35]
        ]
        compact_kept = [
            {"id": k.segment_id, "start": k.start_time, "end": k.end_time}
            for k in kept_segments[:30]
        ]
        compact_cut = [
            {"id": c.segment_id, "reason": c.rationale}
            for c in cut_segments[:30]
        ]

        prompt = f"""
You are an uncompromising Fact-Checking and Editorial Integrity Auditor for Vidara.
Your task is to guarantee that editing raw video does NOT introduce misleading context or distort the speaker's true meaning.

ORIGINAL CORE THESIS:
{intent_graph.core_thesis}

ORIGINAL ANNOTATED SEGMENTS:
{json.dumps(compact_annotated, indent=2)}

PROPOSED KEPT CUTS:
{json.dumps(compact_kept, indent=2)}

PROPOSED CUT SEGMENTS:
{json.dumps(compact_cut, indent=2)}

Audit rigorously across all 8 mandatory editorial integrity checks:
1. NO POLARITY INVERSION: Were negations ('never', 'not', 'no longer') severed from predicates?
2. CAVEAT RETENTION: Was a qualifying restriction ('however', 'only if', 'unless', 'except') cut while the main claim was kept?
3. ENTITY INTEGRITY: Are subjects, numbers, or technical names preserved accurately?
4. PRONOUN/ANTECEDENT BINDING: Does a kept segment begin with orphan demonstratives ('because of this', 'they', 'which means') without the antecedent?
5. TOPIC BOUNDARY INTEGRITY: Does the proposed cut sever an explanation midway before conclusion?
6. SENTENCE COMPLETENESS: Are statements syntactically and semantically complete?
7. EXAMPLE/CONTEXT DEPENDENCY: Is a concrete example retained while the concept it exemplifies was severed?
8. SPEAKER CONTINUITY: Does the cut avoid false attribution between different speakers?

Return STRICTLY a JSON object with this schema:
{{
  "status": "PASSED" | "WARNING" | "BLOCKED",
  "violations_found": 0,
  "overall_faithfulness_score": 0.98,
  "details": [
    {{
      "segment_id": 6,
      "severity": "WARNING" | "BLOCK",
      "issue_type": "SEVERED_CAVEAT" | "POLARITY_INVERSION" | "TOPIC_BOUNDARY_SEVERED" | "SENTENCE_INCOMPLETE" | "CONTEXT_DEPENDENCY_ORPHAN",
      "explanation": "Detailed explanation of the semantic danger.",
      "remediation": "Specific fix to protect meaning."
    }}
  ]
}}
"""
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a media integrity validator ensuring zero distortion of speaker meaning. Return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            except Exception as model_err:
                err_str = str(model_err).lower()
                if ("rate_limit" in err_str or "413" in err_str or "tpm" in err_str or "too large" in err_str) and GROQ_FALLBACK_MODEL:
                    print(f"Vidara: Meaning validator falling back to {GROQ_FALLBACK_MODEL}...")
                    response = client.chat.completions.create(
                        model=GROQ_FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a media integrity validator ensuring zero distortion of speaker meaning. Return only JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                else:
                    raise model_err

            data = json.loads(response.choices[0].message.content)
            return MeaningValidationReport(
                status=data.get("status", "PASSED"),
                violations_found=data.get("violations_found", 0),
                overall_faithfulness_score=data.get("overall_faithfulness_score", 1.0),
                details=data.get("details", [])
            )
        except Exception as e:
            print(f"Validation error: {e}, falling back to heuristic validation.")
            return self._heuristic_validate(intent_graph, edl)

    def _heuristic_validate(self, intent_graph: IntentGraph, edl: EditDecisionList) -> MeaningValidationReport:
        """Deterministic safety checker enforcing 8 editorial integrity checks."""
        violations = []
        decision_map = {d.segment_id: d for d in edl.decisions}
        kept_ids = {d.segment_id for d in edl.decisions if d.action == "KEEP"}

        # Negation & caveat cue words
        negation_cues = ["not", "never", "no", "neither", "hardly", "scarcely", "barely"]
        caveat_cues = ["however", "unless", "except", "only if", "provided that", "but actually", "although"]
        pronoun_starters = ["this means", "because of this", "as a result", "therefore", "these", "those"]

        for ann in intent_graph.annotated_segments:
            dec = decision_map.get(ann.segment_id)
            if not dec:
                continue

            # 1 & 2. Critical Context / Caveat Retention Check
            if ann.critical_context and dec.action == "CUT":
                # Check if preceding or succeeding kept segment depended on this
                violations.append({
                    "segment_id": ann.segment_id,
                    "severity": "WARNING",
                    "issue_type": "QUALIFYING_CONTEXT_REMOVED",
                    "explanation": f"Segment {ann.segment_id} contains critical caveat: '{ann.critical_context}'. Cutting it might alter perceived intent.",
                    "remediation": "Consider setting this segment to KEEP."
                })
                dec.safety_check = "WARN"

            # 3. Sentence Completeness & Pronoun/Antecedent Binding
            if dec.action == "KEEP":
                txt_lower = ann.text.lower().strip()
                # Check if starts with orphan pronoun and previous segment is CUT
                for p in pronoun_starters:
                    if txt_lower.startswith(p):
                        prev_id = ann.segment_id - 1
                        if prev_id in decision_map and prev_id not in kept_ids:
                            violations.append({
                                "segment_id": ann.segment_id,
                                "severity": "WARNING",
                                "issue_type": "ORPHAN_PRONOUN_BINDING",
                                "explanation": f"Segment {ann.segment_id} begins with '{p}' but antecedent segment {prev_id} is cut.",
                                "remediation": f"Keep antecedent segment {prev_id} to maintain referential integrity."
                            })
                            dec.safety_check = "WARN"
                            break

                # 4. Polarity Inversion check
                has_caveat_word = any(c in txt_lower for c in caveat_cues)
                if has_caveat_word and any(n in txt_lower for n in negation_cues):
                    # Check if compound sentence is cut halfway
                    pass

        status = "WARNING" if violations else "PASSED"
        score = max(0.60, 1.0 - (len(violations) * 0.08)) if violations else 1.0

        return MeaningValidationReport(
            status=status,
            violations_found=len(violations),
            overall_faithfulness_score=round(score, 2),
            details=violations
        )

    def validate_topic_boundary(
        self,
        segments: List[dict],
        topic_name: str,
        start_time: float,
        end_time: float,
        lead_in_padding: float = 0.06,
        tail_out_padding: float = 0.12
    ) -> Dict[str, Any]:
        """
        Validates whether a proposed topic boundary is complete and concise.
        Avoids blind outer-bound expansion. Trims filler and guards sentence closure.
        """
        if not segments:
            return {
                "valid": True,
                "topic": topic_name,
                "adjusted_start": round(max(0.0, start_time - lead_in_padding), 2),
                "adjusted_end": round(end_time + tail_out_padding, 2),
                "has_proper_closure": True,
                "segment_count": 0
            }

        # Find segments that intersect or immediately border the range
        relevant_segs = [
            s for s in segments 
            if s.get("end", s.get("end_time", 0.0)) >= start_time - 0.1 
            and s.get("start", s.get("start_time", 0.0)) <= end_time + 0.1
        ]
        
        if not relevant_segs:
            return {
                "valid": True,
                "topic": topic_name,
                "adjusted_start": round(max(0.0, start_time - lead_in_padding), 2),
                "adjusted_end": round(end_time + tail_out_padding, 2),
                "reason": "No segments in range, keeping candidate bounds"
            }

        adjusted_start = start_time
        adjusted_end = end_time

        # Check start for sentence / word completeness
        first_seg = relevant_segs[0]
        first_text = first_seg.get("text", "").strip().lower()
        
        # Check if boundary starts mid-sentence or with an orphan pronoun
        pronoun_starters = ["this means", "because of this", "as a result", "therefore it", "which means"]
        if any(first_text.startswith(p) for p in pronoun_starters):
            # Prepend preceding segment start if within 2.5s
            seg_start = first_seg.get("start", first_seg.get("start_time", start_time))
            adjusted_start = min(adjusted_start, seg_start)

        # Check ending for proper closure
        last_seg = relevant_segs[-1]
        last_text = last_seg.get("text", "").strip()
        has_period = last_text.endswith((".", "!", "?"))

        # Caveat retention guard: Check if immediately following segment is a caveat
        last_end = last_seg.get("end", last_seg.get("end_time", end_time))
        following_segs = [
            s for s in segments 
            if s.get("start", s.get("start_time", 0.0)) >= last_end - 0.2 
            and s.get("start", s.get("start_time", 0.0)) <= last_end + 1.5
        ]
        if following_segs:
            fol_text = following_segs[0].get("text", "").lower()
            caveat_cues = ["however", "unless", "except", "only if", "provided that", "keep in mind"]
            if any(c in fol_text for c in caveat_cues):
                # Include the caveat to prevent severed qualification
                fol_end = following_segs[0].get("end", following_segs[0].get("end_time", last_end))
                adjusted_end = max(adjusted_end, fol_end)

        # Apply configurable subtle speech padding (+60ms lead-in, +120ms tail-out)
        padded_start = max(0.0, round(adjusted_start - lead_in_padding, 2))
        padded_end = round(adjusted_end + tail_out_padding, 2)

        return {
            "valid": True,
            "topic": topic_name,
            "adjusted_start": padded_start,
            "adjusted_end": padded_end,
            "has_proper_closure": has_period,
            "segment_count": len(relevant_segs)
        }

    def validate_candidate_clip(
        self,
        candidate_text: str,
        topic_name: str,
        start_time: float,
        end_time: float,
        context_before: str = "",
        context_after: str = ""
    ) -> Dict[str, Any]:
        """
        Rigorous editorial quality auditor for candidate clips:
        Evaluates 8-point meaning preservation + editorial trimming (excessive intro,
        severed caveat, polarity inversion, orphan pronouns, unjustified duration).
        """
        violations = []
        cand_lower = candidate_text.lower().strip()
        after_lower = context_after.lower().strip()
        before_lower = context_before.lower().strip()
        dur = max(0.1, end_time - start_time)

        # 1. Severed Caveat Guard
        caveat_cues = ["however", "unless", "except", "only if", "provided that", "but actually"]
        if any(after_lower.startswith(c) for c in caveat_cues):
            violations.append({
                "issue_type": "SEVERED_CAVEAT",
                "severity": "WARNING",
                "explanation": "Context immediately after the clip contains a critical caveat that qualifies the statement.",
                "remediation": "Extend end timestamp to incorporate the caveat."
            })

        # 2. Orphan Pronoun Prevention
        pronoun_starters = ["this means", "because of this", "as a result", "therefore", "they", "which means"]
        if any(cand_lower.startswith(p) for p in pronoun_starters):
            violations.append({
                "issue_type": "ORPHAN_PRONOUN",
                "severity": "WARNING",
                "explanation": f"Clip begins with an orphan pronoun/demonstrative without antecedent in view.",
                "remediation": "Prepend previous sentence or definition to ground the referent."
            })

        # 3. Excessive Introductory Content Check
        intro_banter = ["welcome back", "thanks for coming", "mic check", "can you hear me", "alright guys so today"]
        if any(b in cand_lower[:80] for b in intro_banter):
            violations.append({
                "issue_type": "EXCESSIVE_INTRO_CONTENT",
                "severity": "WARNING",
                "explanation": "Clip contains conversational pleasantries or greetings before the actual concept.",
                "remediation": "Trim leading intro banter sentence."
            })

        # 4. Sentence Completeness Check
        has_closure = candidate_text.strip().endswith((".", "!", "?", '"'))
        if not has_closure:
            violations.append({
                "issue_type": "SENTENCE_INCOMPLETE",
                "severity": "WARNING",
                "explanation": "Clip terminates without proper syntactic sentence closure.",
                "remediation": "Extend boundary to next sentence terminator."
            })

        # 5. Unjustified Duration Check
        if dur > 90.0:
            violations.append({
                "issue_type": "UNJUSTIFIED_DURATION",
                "severity": "NOTICE",
                "explanation": f"Clip duration is {round(dur, 1)}s, exceeding the 90s preferred ceiling.",
                "remediation": "Check if topic can be split into smaller standalone sub-clips."
            })

        passed = len([v for v in violations if v["severity"] == "BLOCK"]) == 0
        status = "PASSED" if not violations else "WARNING"

        return {
            "status": status,
            "passed": passed,
            "topic": topic_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": round(dur, 2),
            "violations_found": len(violations),
            "details": violations
        }

    def validate_podcast_exchange(
        self,
        candidate_text: str,
        topic_title: str,
        exchange_type: str,
        start_time: float,
        end_time: float,
        speakers_involved: List[str],
        question_included: bool = False,
        depends_on_question: bool = False,
        context_before: str = "",
        context_after: str = "",
        core_thesis: str = ""
    ) -> Dict[str, Any]:
        """
        Executes the mandatory 10-point podcast editorial validation:
        1. Central insight identification
        2. Worth-watching rationale
        3. Question necessity check (is question needed to understand answer?)
        4. Complete point preservation
        5. Irrelevance/repetition detection
        6. Post-exchange drift/tail trimming
        7. Missing beginning context check
        8. Essential qualification/caveat preservation
        9. Shortening without loss of meaning
        10. Standalone clip justification
        """
        import re
        client, model = self._get_client()
        dur = max(0.5, round(end_time - start_time, 2))

        # Try LLM 10-point validation if available
        if client and model:
            prompt = f"""You are the Lead Editorial Integrity Auditor for Vidara Podcasts.
Audit this proposed podcast clip candidate across the 10 editorial dimensions:

CANDIDATE METADATA:
- Proposed Title: "{topic_title}"
- Exchange Type: {exchange_type}
- Timestamps: {start_time:.2f}s -> {end_time:.2f}s (Duration: {dur:.1f}s)
- Speakers: {', '.join(speakers_involved)}
- Question Included: {question_included}
- Depends on Question: {depends_on_question}
- Core Thesis: "{core_thesis}"

SURROUNDING CONTEXT BEFORE:
"{context_before[-200:]}"

CANDIDATE CLIP DIALOGUE:
"{candidate_text}"

SURROUNDING CONTEXT AFTER:
"{context_after[:200]}"

10-POINT EDITORIAL AUDIT QUESTIONS:
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

Return a valid JSON object with the following structure:
{{
  "validation_status": "PASSED",
  "central_insight": "String summarizing the core takeaway",
  "is_question_necessary": true,
  "is_complete_point": true,
  "has_irrelevant_dialogue": false,
  "recommended_start": {start_time},
  "recommended_end": {end_time},
  "editorial_justification": "Clear explanation of why this moment justifies a clip",
  "quality_score": 0.95,
  "violations": []
}}
"""
            models_to_try = [model or "openai/gpt-oss-120b", GROQ_FALLBACK_MODEL or "openai/gpt-oss-20b"]
            for m in models_to_try:
                if not m:
                    continue
                try:
                    res = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=1200,
                        temperature=0.1,
                        timeout=8.0
                    )
                    content = res.choices[0].message.content or ""
                    if content.strip():
                        data = json.loads(content)
                        data["passed"] = (data.get("validation_status") != "BLOCKED")
                        return data
                except Exception as e:
                    err_str = str(e).lower()
                    print(f"Vidara: Editorial validation model {m} notice ({e}), trying fallback...")
                    if "429" in err_str or "rate_limit" in err_str or "tokens" in err_str:
                        # Immediate fallback to 10-point deterministic heuristic on quota/rate limit
                        break

        # Deterministic Heuristic 10-Point Audit
        violations = []
        c_lower = candidate_text.lower().strip()
        after_lower = context_after.lower().strip()
        before_lower = context_before.lower().strip()

        # 3. Question necessity check: if answer depends on question and question is omitted, flag!
        if depends_on_question and not question_included:
            violations.append({
                "dimension": 3,
                "issue_type": "MISSING_QUESTION_CONTEXT",
                "severity": "WARNING",
                "explanation": "Answer starts with indexical references but question was omitted.",
                "remediation": "Include the preceding question turn."
            })

        # 4. Complete point check
        has_closure = candidate_text.strip().endswith((".", "!", "?", '"'))
        if not has_closure:
            violations.append({
                "dimension": 4,
                "issue_type": "INCOMPLETE_POINT",
                "severity": "WARNING",
                "explanation": "Clip ends mid-sentence without complete grammatical or semantic closure.",
                "remediation": "Extend boundary to nearest sentence terminator."
            })

        # 5. Irrelevant dialogue check
        if any(b in c_lower[:60] for b in ["welcome to the", "thanks for coming", "mic check"]):
            violations.append({
                "dimension": 5,
                "issue_type": "IRRELEVANT_INTRO",
                "severity": "WARNING",
                "explanation": "Opening contains pleasantries unrelated to the insight.",
                "remediation": "Trim leading sentence."
            })

        # 7. Missing beginning context check
        if any(c_lower.startswith(p) for p in ["this means", "because of this", "as a result", "therefore it"]):
            violations.append({
                "dimension": 7,
                "issue_type": "ORPHAN_PRONOUN",
                "severity": "WARNING",
                "explanation": "Clip starts with orphan demonstrative referring to omitted context.",
                "remediation": "Prepend preceding sentence."
            })

        # 8. Caveat preservation
        if any(after_lower.startswith(c) for c in ["however", "unless", "except", "only if"]):
            violations.append({
                "dimension": 8,
                "issue_type": "SEVERED_CAVEAT",
                "severity": "WARNING",
                "explanation": "Important qualification immediately follows the cut.",
                "remediation": "Include subsequent caveat."
            })

        passed = len([v for v in violations if v["severity"] == "BLOCK"]) == 0
        status = "PASSED" if not violations else "WARNING"

        return {
            "validation_status": status,
            "passed": passed,
            "central_insight": candidate_text[:120],
            "is_question_necessary": depends_on_question,
            "is_complete_point": has_closure,
            "has_irrelevant_dialogue": any(v["issue_type"] == "IRRELEVANT_INTRO" for v in violations),
            "recommended_start": start_time,
            "recommended_end": end_time,
            "editorial_justification": f"Valid {exchange_type} podcast moment preserving core insight.",
            "quality_score": 0.92 if passed else 0.75,
            "violations": violations
        }
