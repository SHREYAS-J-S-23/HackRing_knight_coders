import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from backend.models.schemas import (
    EnrichedTranscript,
    TranscriptSegment,
    SentenceSegment,
    ConversationalTurn,
    PodcastExchangeCandidate,
    ValidatedClipCandidate
)
from backend.services.important_clip_selector import ImportantClipSelector, ROLE_LEXICONS, COMMON_STOP_WORDS

# Soft duration targets (seconds) for podcast moments
PODCAST_DURATION_TARGETS = {
    "STANDALONE_INSIGHT": {"min": 12.0, "pref_min": 15.0, "pref_max": 35.0, "max": 45.0},
    "QUESTION_AND_ANSWER": {"min": 18.0, "pref_min": 25.0, "pref_max": 60.0, "max": 75.0},
    "QA_WITH_FOLLOWUP": {"min": 35.0, "pref_min": 45.0, "pref_max": 85.0, "max": 95.0},
    "MEANINGFUL_DISAGREEMENT": {"min": 25.0, "pref_min": 35.0, "pref_max": 75.0, "max": 90.0},
    "REVELATION_OR_STORY": {"min": 20.0, "pref_min": 30.0, "pref_max": 80.0, "max": 95.0},
}

QUESTION_OPENERS = [
    "what is", "what are", "what do you", "why do you", "why is", "how do you",
    "how does", "could you explain", "can you tell us", "tell me about",
    "do you think", "is it true that", "would you say", "what was the moment",
    "how did you", "what's the difference", "where does", "who is"
]

FOLLOW_UP_MARKERS = [
    "but what about", "wait,", "so you're saying", "and then what happened",
    "does that mean", "how so", "why is that", "even if", "but isn't it",
    "what do you mean by that", "and how did"
]

DISAGREEMENT_MARKERS = [
    "i disagree", "i don't agree", "not necessarily", "on the contrary",
    "no, actually", "that's not true", "the problem with that is", "i look at it differently",
    "contrary to what people think", "push back on that", "no no no"
]

STORY_REVELATION_MARKERS = [
    "when i was", "i remember when", "the turning point was", "my rich dad said",
    "i realized that", "nobody talks about this", "the truth is", "the secret is",
    "the biggest lesson", "one day i", "in my experience", "what changed everything"
]

INDEXICAL_DEPENDENCY_STARTERS = [
    "because", "because of that", "that's why", "that is why", "yes, absolutely",
    "exactly, and", "it is", "they are", "which means that", "and so",
    "to answer that", "precisely", "well, yes"
]

class PodcastIntelligenceEngine:
    """
    Podcast-aware conversational intelligence engine that identifies,
    structures, ranks, and precisely trims high-value podcast moments.
    """

    def __init__(self, clip_selector: Optional[ImportantClipSelector] = None):
        self.clip_selector = clip_selector or ImportantClipSelector()

    # -------------------------------------------------------------------------
    # 1. BUILD CONVERSATIONAL TURNS FROM TRANSCRIPT SENTENCES
    # -------------------------------------------------------------------------
    def build_conversational_turns(
        self,
        sentences: List[SentenceSegment]
    ) -> List[ConversationalTurn]:
        """
        Aggregates sentence segments into coherent conversational turns.
        Assigns unit types: QUESTION, FOLLOW_UP, ANSWER, CLARIFICATION,
        EXAMPLE_STORY, DISAGREEMENT, CONCLUSION, TRANSITION, DIGRESSION.
        Links answers and follow-ups to preceding questions.
        """
        if not sentences:
            return []

        turns: List[ConversationalTurn] = []
        current_sentences: List[SentenceSegment] = [sentences[0]]
        current_speaker = sentences[0].speaker or "SPEAKER_00"

        for s in sentences[1:]:
            prev_s = current_sentences[-1]
            time_gap = s.start_time - prev_s.end_time

            # Turn shift conditions: speaker changes OR significant pause (>1.5s)
            speaker_changed = (s.speaker != current_speaker)
            is_significant_pause = (time_gap >= 1.8)

            if speaker_changed or is_significant_pause:
                # Flush current turn
                turn = self._create_turn_from_sentences(len(turns) + 1, current_sentences, current_speaker)
                turns.append(turn)
                current_sentences = [s]
                current_speaker = s.speaker or "SPEAKER_00"
            else:
                current_sentences.append(s)

        if current_sentences:
            turns.append(self._create_turn_from_sentences(len(turns) + 1, current_sentences, current_speaker))

        # Link questions and evaluate indexical dependencies
        self._link_turns(turns)
        return turns

    def _create_turn_from_sentences(
        self,
        turn_id: int,
        sentences: List[SentenceSegment],
        speaker: str
    ) -> ConversationalTurn:
        full_text = " ".join(s.text.strip() for s in sentences)
        start_t = sentences[0].start_time
        end_t = sentences[-1].end_time
        dur = max(0.5, round(end_t - start_t, 2))
        speaker_role = sentences[0].speaker_role or "UNKNOWN"

        unit_type = self._classify_turn_unit_type(full_text, speaker_role, dur)
        is_standalone = self._evaluate_standalone_insight(full_text, unit_type)
        key_claim = self._extract_key_claim(full_text)

        turn = ConversationalTurn(
            turn_id=turn_id,
            speaker=speaker,
            speaker_role=speaker_role,
            start_time=start_t,
            end_time=end_t,
            duration=dur,
            text=full_text,
            unit_type=unit_type,
            sentence_ids=[s.id for s in sentences],
            preceding_question_turn_id=None,
            depends_on_question=False,
            is_standalone_insight=is_standalone,
            key_claim=key_claim
        )

        # Update sentence metadata
        for s in sentences:
            s.conversational_unit = unit_type
        return turn

    def _classify_turn_unit_type(self, text: str, role: str, duration: float) -> str:
        t_lower = text.lower().strip()

        # Follow-up check
        if any(f in t_lower[:50] for f in FOLLOW_UP_MARKERS) and (text.strip().endswith("?") or duration < 15.0):
            return "FOLLOW_UP"

        # Disagreement check
        if any(d in t_lower[:60] for d in DISAGREEMENT_MARKERS):
            return "DISAGREEMENT"

        # Story / revelation check
        if any(r in t_lower[:80] for r in STORY_REVELATION_MARKERS) and len(text.split()) > 15:
            return "EXAMPLE_STORY"

        # Question check
        is_question_ending = text.strip().endswith("?")
        has_question_opener = any(t_lower.startswith(q) for q in QUESTION_OPENERS)
        if is_question_ending or (has_question_opener and duration < 25.0):
            return "QUESTION"

        # Conclusion check
        if any(c in t_lower for c in ROLE_LEXICONS["conclusion"]):
            return "CONCLUSION"

        # Transition / Housekeeping check
        if any(t in t_lower for t in ROLE_LEXICONS["intro_banter"]) and duration < 18.0:
            return "TRANSITION"

        # Digression / Filler check
        if any(t_lower.startswith(f) for f in ROLE_LEXICONS["filler"]) and len(text.split()) < 10:
            return "DIGRESSION"

        return "ANSWER"

    def _evaluate_standalone_insight(self, text: str, unit_type: str) -> bool:
        """Determines if a turn has enough substantive standalone context without preceding question."""
        if unit_type in ["QUESTION", "FOLLOW_UP", "TRANSITION", "DIGRESSION"]:
            return False
        tokens = text.split()
        if len(tokens) < 12:
            return False

        t_lower = text.lower()
        # If it starts with indexical dependencies ("because", "that's why", "yes"), it is NOT standalone
        if any(t_lower.startswith(d) for d in INDEXICAL_DEPENDENCY_STARTERS):
            return False

        # Has substantive conceptual weight
        has_insight_terms = any(w in t_lower for w in [
            "law", "rule", "principle", "truth", "secret", "wealth", "capital", "compounding",
            "means", "causes", "teaches", "lesson", "system", "realize", "fundamental",
            "creates", "requires", "foundation", "is that", "strategy", "concept", "focus",
            "value", "money", "business", "invest", "important", "fact", "reality"
        ])
        return has_insight_terms or len(tokens) >= 15

    def _extract_key_claim(self, text: str) -> str:
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean.split()) >= 6 and not any(s_clean.lower().startswith(f) for f in ROLE_LEXICONS["filler"]):
                return s_clean[:140]
        return text[:140]

    def _link_turns(self, turns: List[ConversationalTurn]) -> None:
        """Links answers, follow-ups, and disagreements back to originating questions."""
        last_question_id = None

        for turn in turns:
            if turn.unit_type in ["QUESTION", "FOLLOW_UP"]:
                last_question_id = turn.turn_id
            elif turn.unit_type in ["ANSWER", "CLARIFICATION", "EXAMPLE_STORY", "DISAGREEMENT", "CONCLUSION"]:
                turn.preceding_question_turn_id = last_question_id
                t_lower = turn.text.lower().strip()
                # Check indexical dependency
                if any(t_lower.startswith(d) for d in INDEXICAL_DEPENDENCY_STARTERS):
                    turn.depends_on_question = True
                elif len(turn.text.split()) < 15 and last_question_id is not None:
                    turn.depends_on_question = True

    # -------------------------------------------------------------------------
    # 2. DISCOVER IMPORTANT PODCAST MOMENTS
    # -------------------------------------------------------------------------
    def extract_podcast_moments(
        self,
        transcript: EnrichedTranscript,
        sentences: List[SentenceSegment],
        core_thesis: str = ""
    ) -> List[PodcastExchangeCandidate]:
        """
        Discovers high-value podcast moments across the entire conversation:
        - Host Question + Guest Answer
        - Standalone insightful guest answer
        - Question + Answer + Follow-up clarification
        - Meaningful disagreement
        - Personal revelation / actionable lesson
        """
        turns = self.build_conversational_turns(sentences)
        if not turns:
            return []

        candidates: List[PodcastExchangeCandidate] = []
        sent_map = {s.id: s for s in sentences}

        turn_idx = 0
        while turn_idx < len(turns):
            turn = turns[turn_idx]

            # -----------------------------------------------------------------
            # PATTERN A: QUESTION AND ANSWER (+ optional FOLLOW-UP)
            # -----------------------------------------------------------------
            if turn.unit_type == "QUESTION":
                q_turn = turn
                # Find corresponding answer turn(s)
                ans_turns = []
                next_idx = turn_idx + 1
                while next_idx < len(turns) and turns[next_idx].unit_type in ["ANSWER", "EXAMPLE_STORY", "CONCLUSION"]:
                    ans_turns.append(turns[next_idx])
                    next_idx += 1

                # Check if there's a valuable follow-up question/clarification
                follow_up_turns = []
                if next_idx < len(turns) and turns[next_idx].unit_type in ["FOLLOW_UP", "CLARIFICATION", "DISAGREEMENT"]:
                    follow_up_turns.append(turns[next_idx])
                    next_idx += 1
                    # Follow-up answer
                    if next_idx < len(turns) and turns[next_idx].unit_type in ["ANSWER", "CONCLUSION"]:
                        follow_up_turns.append(turns[next_idx])
                        next_idx += 1

                if ans_turns:
                    # Formulate QA or QA_WITH_FOLLOWUP exchange
                    if follow_up_turns:
                        cand = self._build_qa_exchange(
                            q_turn, ans_turns, follow_up_turns, sent_map, transcript.duration_seconds
                        )
                    else:
                        cand = self._build_qa_exchange(
                            q_turn, ans_turns, [], sent_map, transcript.duration_seconds
                        )
                    if cand:
                        candidates.append(cand)
                    turn_idx = next_idx
                    continue

            # -----------------------------------------------------------------
            # PATTERN B: STANDALONE INSIGHT (Profounds answer that needs no question)
            # -----------------------------------------------------------------
            elif turn.is_standalone_insight and turn.duration >= 10.0:
                cand = self._build_standalone_exchange(turn, sent_map, transcript.duration_seconds)
                if cand:
                    candidates.append(cand)

            # -----------------------------------------------------------------
            # PATTERN C: MEANINGFUL DISAGREEMENT / DEBATE
            # -----------------------------------------------------------------
            elif turn.unit_type == "DISAGREEMENT":
                prev_turn = turns[turn_idx - 1] if turn_idx > 0 else None
                cand = self._build_disagreement_exchange(prev_turn, turn, sent_map, transcript.duration_seconds)
                if cand:
                    candidates.append(cand)

            # -----------------------------------------------------------------
            # PATTERN D: REVELATION OR ACTIONABLE STORY
            # -----------------------------------------------------------------
            elif turn.unit_type == "EXAMPLE_STORY" and len(turn.text.split()) > 20:
                cand = self._build_story_exchange(turn, sent_map, transcript.duration_seconds)
                if cand:
                    candidates.append(cand)

            turn_idx += 1

        # Rank candidates by information value & editorial significance
        candidates = self._rank_candidates(candidates, core_thesis)
        # Deduplicate
        return self._deduplicate_podcast_exchanges(candidates)

    # -------------------------------------------------------------------------
    # 3. EXCHANGE CONSTRUCTORS WITH PRECISE BOUNDARY OPTIMIZATION
    # -------------------------------------------------------------------------
    def _build_qa_exchange(
        self,
        q_turn: ConversationalTurn,
        ans_turns: List[ConversationalTurn],
        follow_up_turns: List[ConversationalTurn],
        sent_map: Dict[int, SentenceSegment],
        total_duration: float
    ) -> Optional[PodcastExchangeCandidate]:
        """
        Builds a Question & Answer exchange.
        Optimizes start to clean question inquiry (trimming host pre-ramble).
        Trims answer to the core explanation.
        """
        all_turns = [q_turn] + ans_turns + follow_up_turns
        raw_sids = []
        for t in all_turns:
            raw_sids.extend(t.sentence_ids)

        sents = [sent_map[sid] for sid in raw_sids if sid in sent_map]
        if not sents:
            return None

        # 1. Trim leading host preamble in the question turn (e.g. "Welcome back, so you know, my question is")
        q_sents = [sent_map[sid] for sid in q_turn.sentence_ids if sid in sent_map]
        while len(q_sents) > 1:
            first_q = q_sents[0]
            if first_q.semantic_role in ["INTRO", "FILLER"] and not first_q.text.strip().endswith("?"):
                q_sents.pop(0)
            else:
                break

        # 2. Check if answer is completely standalone; if so and question is long/repetitive, question can be trimmed
        depends = any(t.depends_on_question for t in ans_turns)
        primary_ans_turn = ans_turns[0]
        question_included = True

        if not depends and primary_ans_turn.is_standalone_insight and q_turn.duration > 20.0:
            # Standalone answer can be selected directly without burdensome question
            selected_sents = [sent_map[sid] for sid in primary_ans_turn.sentence_ids if sid in sent_map]
            question_included = False
            exchange_type = "STANDALONE_INSIGHT"
        else:
            ans_sents = []
            for t in (ans_turns + follow_up_turns):
                ans_sents.extend([sent_map[sid] for sid in t.sentence_ids if sid in sent_map])
            selected_sents = q_sents + ans_sents
            exchange_type = "QA_WITH_FOLLOWUP" if follow_up_turns else "QUESTION_AND_ANSWER"

        # 3. Trim trailing chatter from the end of the exchange
        while len(selected_sents) > 2:
            last_s = selected_sents[-1]
            if last_s.semantic_role in ["FILLER", "INTRO"] or (last_s.duration < 1.2 and last_s.text.lower() in ["yeah", "right", "okay", "wow"]):
                selected_sents.pop(-1)
            else:
                break

        if not selected_sents:
            return None

        start_t = selected_sents[0].start_time
        end_t = selected_sents[-1].end_time
        dur = max(0.5, round(end_t - start_t, 2))

        # Respect soft ceiling for exchange type
        cfg = PODCAST_DURATION_TARGETS.get(exchange_type, PODCAST_DURATION_TARGETS["QUESTION_AND_ANSWER"])
        if dur > cfg["max"] and len(selected_sents) > 4:
            # Sub-window to maintain tight core
            target_max = cfg["pref_max"]
            sub = []
            curr_dur = 0.0
            for s in selected_sents:
                if curr_dur + s.duration <= target_max or len(sub) < 3:
                    sub.append(s)
                    curr_dur += s.duration
                else:
                    break
            selected_sents = sub
            start_t = selected_sents[0].start_time
            end_t = selected_sents[-1].end_time
            dur = max(0.5, round(end_t - start_t, 2))

        speakers = list(set(s.speaker for s in selected_sents if s.speaker))
        roles = {s.speaker: s.speaker_role for s in selected_sents if s.speaker}

        # Topic title formulation
        topic_title = self._formulate_exchange_title(q_turn, primary_ans_turn)
        core_insight = primary_ans_turn.key_claim or selected_sents[-1].text[:140]

        return PodcastExchangeCandidate(
            exchange_id=f"podcast_qa_{q_turn.turn_id}_{start_t:.1f}",
            exchange_type=exchange_type,
            topic_title=topic_title,
            core_insight=core_insight,
            start_time=round(max(0.0, start_t - 0.06), 2),
            end_time=round(min(total_duration, end_t + 0.12), 2),
            duration_seconds=dur,
            speakers_involved=speakers,
            speaker_roles=roles,
            turn_ids=[t.turn_id for t in all_turns],
            sentence_ids=[s.id for s in selected_sents],
            question_included=question_included,
            depends_on_question=depends,
            information_value=0.92,
            relevance_score=0.90,
            completeness_score=0.95 if selected_sents[-1].text.strip().endswith((".", "!", "?")) else 0.85,
            redundancy_score=0.04,
            editorial_justification=f"Substantive exchange: '{topic_title}' with clean question framing and complete resolution.",
            validation_status="PASSED"
        )

    def _build_standalone_exchange(
        self,
        turn: ConversationalTurn,
        sent_map: Dict[int, SentenceSegment],
        total_duration: float
    ) -> Optional[PodcastExchangeCandidate]:
        sents = [sent_map[sid] for sid in turn.sentence_ids if sid in sent_map]
        if not sents:
            return None

        # Trim filler at boundaries
        while len(sents) > 1 and sents[0].semantic_role in ["FILLER", "INTRO"]:
            sents.pop(0)
        while len(sents) > 1 and sents[-1].semantic_role in ["FILLER", "INTRO"]:
            sents.pop(-1)

        start_t = sents[0].start_time
        end_t = sents[-1].end_time
        dur = max(0.5, round(end_t - start_t, 2))

        topic_title = f"Insight: {turn.key_claim[:50]}..." if turn.key_claim else f"Key Perspective ({turn.speaker})"

        return PodcastExchangeCandidate(
            exchange_id=f"podcast_insight_{turn.turn_id}_{start_t:.1f}",
            exchange_type="STANDALONE_INSIGHT",
            topic_title=topic_title,
            core_insight=turn.key_claim or sents[0].text[:140],
            start_time=round(max(0.0, start_t - 0.06), 2),
            end_time=round(min(total_duration, end_t + 0.12), 2),
            duration_seconds=dur,
            speakers_involved=[turn.speaker],
            speaker_roles={turn.speaker: turn.speaker_role},
            turn_ids=[turn.turn_id],
            sentence_ids=[s.id for s in sents],
            question_included=False,
            depends_on_question=False,
            information_value=0.88,
            relevance_score=0.88,
            completeness_score=0.92,
            redundancy_score=0.03,
            editorial_justification="Standalone, self-contained lesson delivered with complete reasoning.",
            validation_status="PASSED"
        )

    def _build_disagreement_exchange(
        self,
        prev_turn: Optional[ConversationalTurn],
        disagree_turn: ConversationalTurn,
        sent_map: Dict[int, SentenceSegment],
        total_duration: float
    ) -> Optional[PodcastExchangeCandidate]:
        turns = [prev_turn, disagree_turn] if prev_turn else [disagree_turn]
        sents = []
        for t in turns:
            sents.extend([sent_map[sid] for sid in t.sentence_ids if sid in sent_map])
        if not sents:
            return None

        start_t = sents[0].start_time
        end_t = sents[-1].end_time
        dur = max(0.5, round(end_t - start_t, 2))
        speakers = list(set(s.speaker for s in sents if s.speaker))

        return PodcastExchangeCandidate(
            exchange_id=f"podcast_debate_{disagree_turn.turn_id}_{start_t:.1f}",
            exchange_type="MEANINGFUL_DISAGREEMENT",
            topic_title="Debate: Contrasting Perspectives",
            core_insight=disagree_turn.key_claim or sents[-1].text[:140],
            start_time=round(max(0.0, start_t - 0.06), 2),
            end_time=round(min(total_duration, end_t + 0.12), 2),
            duration_seconds=dur,
            speakers_involved=speakers,
            speaker_roles={s.speaker: s.speaker_role for s in sents if s.speaker},
            turn_ids=[t.turn_id for t in turns],
            sentence_ids=[s.id for s in sents],
            question_included=False,
            depends_on_question=False,
            information_value=0.94,
            relevance_score=0.92,
            completeness_score=0.90,
            redundancy_score=0.02,
            editorial_justification="Meaningful conversational counter-argument highlighting key distinction.",
            validation_status="PASSED"
        )

    def _build_story_exchange(
        self,
        story_turn: ConversationalTurn,
        sent_map: Dict[int, SentenceSegment],
        total_duration: float
    ) -> Optional[PodcastExchangeCandidate]:
        sents = [sent_map[sid] for sid in story_turn.sentence_ids if sid in sent_map]
        if not sents:
            return None

        start_t = sents[0].start_time
        end_t = sents[-1].end_time
        dur = max(0.5, round(end_t - start_t, 2))

        return PodcastExchangeCandidate(
            exchange_id=f"podcast_story_{story_turn.turn_id}_{start_t:.1f}",
            exchange_type="REVELATION_OR_STORY",
            topic_title="Lesson: Personal Experience & Revelation",
            core_insight=story_turn.key_claim or sents[0].text[:140],
            start_time=round(max(0.0, start_t - 0.06), 2),
            end_time=round(min(total_duration, end_t + 0.12), 2),
            duration_seconds=dur,
            speakers_involved=[story_turn.speaker],
            speaker_roles={story_turn.speaker: story_turn.speaker_role},
            turn_ids=[story_turn.turn_id],
            sentence_ids=[s.id for s in sents],
            question_included=False,
            depends_on_question=False,
            information_value=0.91,
            relevance_score=0.89,
            completeness_score=0.94,
            redundancy_score=0.04,
            editorial_justification="Concrete narrative revelation providing actionable takeaways.",
            validation_status="PASSED"
        )

    def _formulate_exchange_title(self, q_turn: ConversationalTurn, ans_turn: ConversationalTurn) -> str:
        q_text = q_turn.text.strip()
        # Clean up question to create a punchy title
        q_clean = re.sub(r'^(so|and|well|you know|tell me|can you tell us|what about)\s*,?\s*', '', q_text, flags=re.IGNORECASE)
        q_clean = q_clean.rstrip("?").strip()
        if len(q_clean.split()) >= 3 and len(q_clean) <= 65:
            return q_clean[:60]
        # Otherwise use key claim from answer
        if ans_turn.key_claim and len(ans_turn.key_claim) <= 65:
            return ans_turn.key_claim
        return q_text[:50] + ("..." if len(q_text) > 50 else "")

    # -------------------------------------------------------------------------
    # 4. RANKING & DEDUPLICATION
    # -------------------------------------------------------------------------
    def _rank_candidates(
        self,
        candidates: List[PodcastExchangeCandidate],
        core_thesis: str
    ) -> List[PodcastExchangeCandidate]:
        thesis_words = set(re.findall(r'\w+', core_thesis.lower()))

        for c in candidates:
            # Score signals
            cand_words = set(re.findall(r'\w+', c.topic_title.lower() + " " + c.core_insight.lower()))
            thesis_overlap = len(cand_words.intersection(thesis_words)) / max(1, len(thesis_words)) if thesis_words else 0.5

            # Multi-speaker synergy bonus
            multi_speaker_bonus = 0.08 if len(c.speakers_involved) > 1 else 0.0

            # Duration optimality factor
            cfg = PODCAST_DURATION_TARGETS.get(c.exchange_type, {"pref_min": 20.0, "pref_max": 60.0})
            if cfg["pref_min"] <= c.duration_seconds <= cfg["pref_max"]:
                dur_factor = 1.05
            elif c.duration_seconds < cfg["pref_min"]:
                dur_factor = 0.90
            else:
                dur_factor = 0.95

            composite_score = (
                (c.information_value * 0.40) +
                (c.relevance_score * 0.25) +
                (c.completeness_score * 0.20) +
                (thesis_overlap * 0.15) +
                multi_speaker_bonus
            ) * dur_factor - c.redundancy_score

            c.information_value = round(composite_score, 2)

        candidates.sort(key=lambda c: c.information_value, reverse=True)
        return candidates

    def _deduplicate_podcast_exchanges(
        self,
        candidates: List[PodcastExchangeCandidate],
        iou_threshold: float = 0.40
    ) -> List[PodcastExchangeCandidate]:
        unique: List[PodcastExchangeCandidate] = []
        for c in candidates:
            is_dup = False
            for kept in unique:
                # Time IoU
                inter_s = max(c.start_time, kept.start_time)
                inter_e = min(c.end_time, kept.end_time)
                inter = max(0.0, inter_e - inter_s)
                union = max(0.1, max(c.end_time, kept.end_time) - min(c.start_time, kept.start_time))
                iou = inter / union

                # Sentence ID overlap
                s_c = set(c.sentence_ids)
                s_k = set(kept.sentence_ids)
                sent_overlap = len(s_c.intersection(s_k)) / max(1, len(s_c))

                if iou >= iou_threshold or sent_overlap >= 0.50:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)

        unique.sort(key=lambda c: c.start_time)
        return unique

    def convert_exchanges_to_clip_candidates(
        self,
        exchanges: List[PodcastExchangeCandidate]
    ) -> List[ValidatedClipCandidate]:
        """Converts podcast exchange candidates to ValidatedClipCandidate objects."""
        clip_cands: List[ValidatedClipCandidate] = []
        for idx, ex in enumerate(exchanges):
            clip_cands.append(
                ValidatedClipCandidate(
                    topic_id=ex.exchange_id,
                    topic_title=ex.topic_title,
                    start_ms=int(ex.start_time * 1000),
                    end_ms=int(ex.end_time * 1000),
                    start_time=ex.start_time,
                    end_time=ex.end_time,
                    duration_seconds=ex.duration_seconds,
                    selected_sentence_ids=ex.sentence_ids,
                    key_information=ex.core_insight,
                    selection_reason=f"[{ex.exchange_type}] {ex.editorial_justification}",
                    excluded_content_reason="Trimmed conversational pleasantries, prolonged pauses, and off-topic banter",
                    completeness_score=ex.completeness_score,
                    relevance_score=ex.relevance_score,
                    redundancy_score=ex.redundancy_score,
                    standalone_score=0.95 if not ex.depends_on_question else 0.88,
                    requires_expansion=False,
                    validation_status=ex.validation_status,
                    exchange_type=ex.exchange_type,
                    speakers_involved=ex.speakers_involved,
                    speaker_roles=ex.speaker_roles,
                    question_included=ex.question_included,
                    depends_on_question=ex.depends_on_question,
                    editorial_justification=ex.editorial_justification
                )
            )
        return clip_cands
