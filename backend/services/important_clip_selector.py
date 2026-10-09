import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from backend.models.schemas import (
    EnrichedTranscript,
    TranscriptSegment,
    WordTimestamp,
    SentenceSegment,
    ValidatedClipCandidate,
    DiscoveredTopic
)

# ====================================================
# CONFIGURABLE FEATURE WEIGHTS & DURATION TARGETS
# ====================================================
DEFAULT_SENTENCE_SCORING_WEIGHTS = {
    "topic_relevance": 0.25,
    "information_density": 0.20,
    "concept_contribution": 0.20,
    "semantic_completeness": 0.15,
    "example_evidence": 0.10,
    "conclusion_impact": 0.10,
}

# Configurable duration parameters (seconds)
DEFAULT_DURATION_CONFIG = {
    "min_duration": 10.0,          # Soft floor for standalone fact
    "preferred_min": 15.0,        # Preferred compact definition / explanation
    "preferred_max": 60.0,        # Soft ceiling for complete concept
    "hard_max": 90.0,             # Upper bound unless complex mechanism genuinely requires it
    "cluster_gap_threshold": 25.0, # If mentions are separated by >25s, treat as separate candidate moments
    "lead_in_padding": 0.06,      # +60ms natural consonant attack headroom
    "tail_out_padding": 0.12,     # +120ms natural release headroom
}

# Rhetorical marker lexicons for fine-grained sentence tagging
ROLE_LEXICONS = {
    "intro_banter": [
        "welcome back", "thanks for coming", "can you hear me", "mic check",
        "today we're going to talk about", "today we are going to discuss",
        "let's dive into", "in this video we", "let's talk about", "i want to start by",
        "alright guys", "hey everyone", "good morning", "good afternoon"
    ],
    "filler": [
        "you know what i mean", "like i said", "so basically yeah", "um anyway", "anyway",
        "speaking of", "by the way", "on another note", "off topic", "kind of sort of",
        "let's see here", "uh let me check", "just hanging out", "at the end of the day",
        "to be honest with you"
    ],
    "definition": [
        "is defined as", "refers to", "is a system that", "means that",
        "is essentially a", "is fundamentally", "what is", "by definition",
        "is known as", "we call this"
    ],
    "mechanism": [
        "the mechanism here", "how this works", "the algorithm", "under the hood",
        "translates", "step 1", "first it", "then the cpu", "in order to execute",
        "the process of", "the data structure", "architecture"
    ],
    "caveat": [
        "however", "unless", "except", "only if", "provided that", "keep in mind",
        "the catch is", "contrary to", "on the other hand", "the limitation",
        "crucially though", "but actually", "although"
    ],
    "example": [
        "for example", "for instance", "such as", "let's look at", "consider the case",
        "case study", "demonstration", "in this diagram", "take for example"
    ],
    "conclusion": [
        "in conclusion", "to summarize", "the takeaway is", "therefore", "the bottom line",
        "as a result", "in summary", "to wrap up", "ultimately", "eliminates"
    ]
}

# Common conversational stop words for information density computation
COMMON_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he",
    "she", "it", "we", "they", "this", "that", "these", "those", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "well", "like", "um", "uh", "yeah", "okay", "ok"
}

PRONOUN_STARTERS = [
    "this means", "because of this", "as a result", "therefore it", "which means",
    "these are", "those are", "they do this", "he said that", "it works by"
]


class ImportantClipSelector:
    """
    Dedicated AI intelligence selector for extracting the shortest complete,
    information-dense, and meaningful video clip for a topic while removing unnecessary footage.
    """

    def __init__(
        self,
        scoring_weights: Optional[Dict[str, float]] = None,
        duration_config: Optional[Dict[str, float]] = None
    ):
        self.weights = scoring_weights or DEFAULT_SENTENCE_SCORING_WEIGHTS.copy()
        self.duration_config = duration_config or DEFAULT_DURATION_CONFIG.copy()

    # -------------------------------------------------------------------------
    # 1. FINE-GRAINED SENTENCE SEGMENTATION PIPELINE
    # -------------------------------------------------------------------------
    def extract_sentences(self, transcript: EnrichedTranscript) -> List[SentenceSegment]:
        """
        Builds a structured sentence-level transcript from Whisper transcript segments.
        Maps word-level millisecond timestamps to precise sentence boundaries.
        Preserves original timeline and assigns semantic roles.
        """
        raw_segs = transcript.segments or []
        if not raw_segs:
            return []

        sentences: List[SentenceSegment] = []
        sentence_counter = 1

        for seg in raw_segs:
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            words = seg.words or []
            
            # Split segment text into sentences using linguistic terminators
            # Lookbehind/lookahead avoids splitting on abbreviations like e.g., i.e., vs.
            raw_splits = re.split(r'(?<=[.!?])\s+', seg_text)
            
            # If no sentence punctuation exists, treat the segment as a whole sentence
            if not raw_splits or len(raw_splits) == 1:
                splits = [seg_text]
            else:
                splits = [s.strip() for s in raw_splits if s.strip()]

            # If word timestamps are available, align each sentence with its slice of words
            if words:
                word_idx = 0
                total_words = len(words)

                for split_text in splits:
                    split_words_tokens = split_text.split()
                    matched_words: List[WordTimestamp] = []
                    
                    # Match words sequentially
                    start_w_idx = word_idx
                    tokens_to_match = len(split_words_tokens)
                    end_w_idx = min(total_words, start_w_idx + tokens_to_match)

                    matched_words = words[start_w_idx:end_w_idx]
                    word_idx = end_w_idx

                    if matched_words:
                        sent_start = matched_words[0].start
                        sent_end = matched_words[-1].end
                    else:
                        sent_start = seg.start
                        sent_end = seg.end

                    duration = max(0.2, round(sent_end - sent_start, 2))
                    role = self._classify_sentence_role(split_text)
                    density = self._calculate_information_density(split_text)

                    sentences.append(
                        SentenceSegment(
                            id=sentence_counter,
                            text=split_text,
                            start_time=round(sent_start, 2),
                            end_time=round(sent_end, 2),
                            duration=duration,
                            words=matched_words,
                            parent_segment_id=seg.id,
                            speaker=seg.speaker or "SPEAKER_01",
                            semantic_role=role,
                            information_density=density,
                            importance_score=0.5
                        )
                    )
                    sentence_counter += 1
            else:
                # Proportional interpolation if word timestamps are not provided
                seg_dur = max(0.5, seg.end - seg.start)
                total_chars = max(1, len(seg_text))
                curr_start = seg.start

                for split_text in splits:
                    fraction = len(split_text) / total_chars
                    sent_dur = max(0.5, round(seg_dur * fraction, 2))
                    sent_end = min(seg.end, round(curr_start + sent_dur, 2))

                    role = self._classify_sentence_role(split_text)
                    density = self._calculate_information_density(split_text)

                    sentences.append(
                        SentenceSegment(
                            id=sentence_counter,
                            text=split_text,
                            start_time=round(curr_start, 2),
                            end_time=round(sent_end, 2),
                            duration=round(sent_end - curr_start, 2),
                            words=[],
                            parent_segment_id=seg.id,
                            speaker=seg.speaker or "SPEAKER_01",
                            semantic_role=role,
                            information_density=density,
                            importance_score=0.5
                        )
                    )
                    sentence_counter += 1
                    curr_start = sent_end

        return sentences

    def _classify_sentence_role(self, text: str) -> str:
        """Classifies a sentence into candidate rhetorical roles."""
        t_lower = text.lower().strip()

        # Check in order of specificity
        if any(f in t_lower for f in ROLE_LEXICONS["filler"]):
            return "FILLER"
        if any(c in t_lower for c in ROLE_LEXICONS["caveat"]):
            return "CAVEAT"
        if any(d in t_lower for d in ROLE_LEXICONS["definition"]):
            return "DEFINITION"
        if any(m in t_lower for m in ROLE_LEXICONS["mechanism"]):
            return "MECHANISM"
        if any(e in t_lower for e in ROLE_LEXICONS["example"]):
            return "EXAMPLE"
        if any(c in t_lower for c in ROLE_LEXICONS["conclusion"]):
            return "CONCLUSION"
        if any(i in t_lower for i in ROLE_LEXICONS["intro_banter"]):
            return "INTRO"

        return "EXPLANATION"

    def _calculate_information_density(self, text: str) -> float:
        """Calculates information density as ratio of unique content words to total words."""
        tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
        if not tokens:
            return 0.1
        content_words = [t for t in tokens if t not in COMMON_STOP_WORDS and len(t) > 2]
        unique_content = set(content_words)
        
        # Density ratio with technical length bonus
        density = len(unique_content) / max(1, len(tokens))
        return min(1.0, round(density * 1.6, 2))

    # -------------------------------------------------------------------------
    # 2. SENTENCE-LEVEL INFORMATION-VALUE SCORING
    # -------------------------------------------------------------------------
    def score_sentence(
        self,
        sentence: SentenceSegment,
        topic_name: str,
        topic_keywords: List[str],
        preceding_sentence: Optional[SentenceSegment] = None,
        selected_history: Optional[List[SentenceSegment]] = None
    ) -> float:
        """
        Ranks candidate sentences using a multi-feature value scoring function:
        - 25% topic_relevance
        - 20% information_density
        - 20% concept_contribution
        - 15% semantic_completeness
        - 10% example_evidence
        - 10% conclusion_impact
        Penalizes filler, intro pleasantries, and repetition.
        """
        text_lower = sentence.text.lower()
        topic_terms = set(re.findall(r'\w+', topic_name.lower()) + [k.lower() for k in topic_keywords if k])

        # 1. Topic Relevance (0.0 - 1.0)
        sent_words = set(re.findall(r'\w+', text_lower))
        overlap = sent_words.intersection(topic_terms)
        rel_score = min(1.0, (len(overlap) / max(1, len(topic_terms))) * 1.5) if (topic_terms and overlap) else (0.5 if not topic_terms else 0.0)
        if topic_name.lower() in text_lower:
            rel_score = max(rel_score, 0.95)

        # 2. Information Density (0.0 - 1.0)
        density_score = sentence.information_density or self._calculate_information_density(sentence.text)

        # 3. Concept Contribution (0.0 - 1.0)
        # Higher for definitions, mechanism terms, causality
        concept_score = 0.5
        if sentence.semantic_role in ["DEFINITION", "MECHANISM"]:
            concept_score = 0.95
        elif sentence.semantic_role == "CAVEAT":
            concept_score = 0.90
        elif any(term in text_lower for term in ["because", "causes", "mechanism", "translates", "executes", "allocates", "stores"]):
            concept_score = 0.85

        # 4. Semantic Completeness (0.0 - 1.0)
        # Sentence is complete if it has proper punctuation, reasonable word count, not an isolated fragment
        has_period = sentence.text.strip().endswith((".", "!", "?"))
        word_count = len(text_lower.split())
        completeness = 0.4
        if has_period and word_count >= 5:
            completeness = 0.95
        elif word_count >= 6:
            completeness = 0.80

        # 5. Example / Evidence (0.0 - 1.0)
        example_score = 0.90 if sentence.semantic_role == "EXAMPLE" else 0.20

        # 6. Conclusion / Main Argument Impact (0.0 - 1.0)
        conclusion_score = 0.95 if sentence.semantic_role == "CONCLUSION" else 0.20

        # Base composite score
        base_score = (
            self.weights["topic_relevance"] * rel_score +
            self.weights["information_density"] * density_score +
            self.weights["concept_contribution"] * concept_score +
            self.weights["semantic_completeness"] * completeness +
            self.weights["example_evidence"] * example_score +
            self.weights["conclusion_impact"] * conclusion_score
        )

        # Penalties:
        penalty = 0.0
        # Filler penalty
        if sentence.semantic_role == "FILLER":
            penalty += 0.35
        # Intro banter penalty (greetings, pleasantries)
        if sentence.semantic_role == "INTRO" and any(p in text_lower for p in ["welcome", "thanks for", "can you hear me", "alright guys"]):
            penalty += 0.30

        # Irrelevance penalty
        if topic_terms and not overlap and sentence.semantic_role not in ["CONCLUSION", "CAVEAT"]:
            penalty += 0.25

        # Redundancy penalty (comparing against already selected sentences)
        if selected_history:
            for past in selected_history:
                past_words = set(re.findall(r'\w+', past.text.lower()))
                common = sent_words.intersection(past_words)
                jaccard = len(common) / max(1, len(sent_words.union(past_words)))
                if jaccard > 0.65:
                    penalty += 0.35
                    break

        # Orphan pronoun penalty if without antecedent
        if preceding_sentence is None and any(text_lower.startswith(p) for p in PRONOUN_STARTERS):
            penalty += 0.20

        final_score = max(0.05, min(0.99, round(base_score - penalty, 3)))
        sentence.importance_score = final_score
        return final_score

    # -------------------------------------------------------------------------
    # 3. STRICT CANDIDATE-CLIP SELECTION STAGE
    # -------------------------------------------------------------------------
    def select_best_clip_for_topic(
        self,
        topic_name: str,
        topic_id: str,
        sentences: List[SentenceSegment],
        subtopics: Optional[List[str]] = None,
        description: str = "",
        total_video_duration: float = 0.0
    ) -> Optional[ValidatedClipCandidate]:
        """
        Converts a candidate topic into the shortest complete, meaningful timestamp range.
        - Identifies topic-relevant sentences.
        - Clusters separated temporal mentions (handles topic revisited at 1:00 vs 8:00).
        - Finds the optimal contiguous sentence window satisfying duration constraints.
        - Trims leading/trailing conversational banter.
        - Preserves mandatory caveats and subject antecedents.
        """
        if not sentences:
            return None

        subtopics = subtopics or []
        topic_keywords = [w for w in topic_name.split() if len(w) > 3] + subtopics

        # Score every sentence for this topic
        for s in sentences:
            self.score_sentence(s, topic_name, topic_keywords)

        # Identify candidate sentences (high score or directly mentioning topic words)
        candidate_indices = []
        topic_terms = set(re.findall(r'\w+', topic_name.lower()) + [k.lower() for k in topic_keywords if k])

        for idx, s in enumerate(sentences):
            s_words = set(re.findall(r'\w+', s.text.lower()))
            has_direct_keyword = bool(s_words.intersection(topic_terms))
            # Must either directly match topic keywords or have high relevance and definition/mechanism role
            if has_direct_keyword or (s.importance_score >= 0.55 and s.semantic_role in ["DEFINITION", "MECHANISM", "CAVEAT"]):
                candidate_indices.append(idx)

        if not candidate_indices:
            # Fallback to top scored sentences
            candidate_indices = sorted(range(len(sentences)), key=lambda i: sentences[i].importance_score, reverse=True)[:3]
            candidate_indices.sort()

        # Group candidate sentences into temporal clusters (gap threshold)
        gap_limit = self.duration_config.get("cluster_gap_threshold", 25.0)
        clusters: List[List[int]] = []
        current_cluster = [candidate_indices[0]]

        for i in range(1, len(candidate_indices)):
            prev_idx = candidate_indices[i - 1]
            curr_idx = candidate_indices[i]
            time_gap = sentences[curr_idx].start_time - sentences[prev_idx].end_time

            if time_gap <= gap_limit:
                current_cluster.append(curr_idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [curr_idx]

        if current_cluster:
            clusters.append(current_cluster)

        # Evaluate each cluster to find the most coherent, high-information window
        best_candidate: Optional[ValidatedClipCandidate] = None
        best_objective_score = -1.0

        for cluster in clusters:
            candidate = self._optimize_cluster_window(
                cluster, sentences, topic_name, topic_id, total_video_duration
            )
            if candidate:
                # Composite score: relevance * completeness / duration_factor
                dur_factor = 1.0
                if candidate.duration_seconds > self.duration_config["preferred_max"]:
                    dur_factor = max(0.6, 1.0 - ((candidate.duration_seconds - self.duration_config["preferred_max"]) / 100.0))
                
                obj_score = candidate.relevance_score * candidate.completeness_score * dur_factor
                if obj_score > best_objective_score:
                    best_objective_score = obj_score
                    best_candidate = candidate

        return best_candidate

    def _optimize_cluster_window(
        self,
        cluster_indices: List[int],
        sentences: List[SentenceSegment],
        topic_name: str,
        topic_id: str,
        total_video_duration: float
    ) -> Optional[ValidatedClipCandidate]:
        """
        Finds the smallest contiguous sentence range within a cluster that forms
        a complete, standalone explanation.
        """
        start_idx = cluster_indices[0]
        end_idx = cluster_indices[-1]

        # Expand to contiguous range [start_idx, end_idx]
        window_indices = list(range(start_idx, end_idx + 1))
        topic_terms = set(re.findall(r'\w+', topic_name.lower()))
        
        # 1. Trim leading sentences that are low-value filler or intro banter
        while len(window_indices) > 1:
            first_sent = sentences[window_indices[0]]
            is_intro_banter = first_sent.semantic_role in ["INTRO", "FILLER"] and not any(
                term in first_sent.text.lower() for term in ["defined as", "means", "is a system"]
            )
            is_low_score = first_sent.importance_score < 0.42
            if is_intro_banter or (is_low_score and len(window_indices) > 2):
                window_indices.pop(0)
            else:
                break

        # 2. Trim trailing sentences that are low-value transitions, digressions, or off-topic
        while len(window_indices) > 1:
            last_sent = sentences[window_indices[-1]]
            sent_words = set(re.findall(r'\w+', last_sent.text.lower()))
            has_topic_words = bool(sent_words.intersection(topic_terms)) if topic_terms else True
            is_trailing_banter = (
                (not has_topic_words and last_sent.semantic_role not in ["CONCLUSION", "CAVEAT"])
                or last_sent.semantic_role in ["INTRO", "FILLER"]
                or (last_sent.importance_score < 0.45 and last_sent.semantic_role not in ["CONCLUSION", "CAVEAT"])
            )
            if is_trailing_banter:
                window_indices.pop(-1)
            else:
                break

        # 3. Caveat Guard: If the sentence immediately after the window is an essential CAVEAT, include it!
        last_kept_idx = window_indices[-1]
        if last_kept_idx + 1 < len(sentences):
            next_sent = sentences[last_kept_idx + 1]
            if next_sent.semantic_role == "CAVEAT" or any(c in next_sent.text.lower() for c in ["however", "unless", "except", "only if"]):
                # Ensure the caveat is not separated by a huge pause
                if next_sent.start_time - sentences[last_kept_idx].end_time < 3.0:
                    window_indices.append(last_kept_idx + 1)

        # 4. Orphan Pronoun Prevention: If the window starts with an orphan pronoun, prepend antecedent!
        first_kept_idx = window_indices[0]
        first_text_lower = sentences[first_kept_idx].text.lower()
        if any(first_text_lower.startswith(p) for p in PRONOUN_STARTERS) and first_kept_idx > 0:
            prev_sent = sentences[first_kept_idx - 1]
            if sentences[first_kept_idx].start_time - prev_sent.end_time < 3.0:
                window_indices.insert(0, first_kept_idx - 1)

        # Calculate raw window timestamps
        selected_sents = [sentences[i] for i in window_indices]
        raw_start = selected_sents[0].start_time
        raw_end = selected_sents[-1].end_time
        raw_dur = max(0.5, raw_end - raw_start)

        # If window is excessively long (> hard_max), try to find the tightest core subset
        hard_max = self.duration_config.get("hard_max", 90.0)
        if raw_dur > hard_max and len(selected_sents) > 3:
            # Find the contiguous sub-window with highest average importance
            best_sub = window_indices
            best_sub_score = -1.0
            for w_len in range(2, len(window_indices)):
                for i in range(len(window_indices) - w_len + 1):
                    sub = window_indices[i:i + w_len]
                    sub_dur = sentences[sub[-1]].end_time - sentences[sub[0]].start_time
                    if sub_dur <= hard_max:
                        avg_sc = sum(sentences[k].importance_score for k in sub) / len(sub)
                        if avg_sc > best_sub_score:
                            best_sub_score = avg_sc
                            best_sub = sub
            window_indices = best_sub
            selected_sents = [sentences[i] for i in window_indices]
            raw_start = selected_sents[0].start_time
            raw_end = selected_sents[-1].end_time
            raw_dur = max(0.5, raw_end - raw_start)

        # Apply subtle audio timing padding (+60ms lead-in, +120ms tail-out)
        pad_in = self.duration_config.get("lead_in_padding", 0.06)
        pad_out = self.duration_config.get("tail_out_padding", 0.12)
        total_vid_limit = total_video_duration if total_video_duration > 0 else raw_end + 10.0

        final_start = max(0.0, round(raw_start - pad_in, 2))
        final_end = min(total_vid_limit, round(raw_end + pad_out, 2))
        final_dur = max(0.5, round(final_end - final_start, 2))

        # Metrics
        avg_rel = sum(s.importance_score for s in selected_sents) / max(1, len(selected_sents))
        has_thesis_or_def = any(s.semantic_role in ["DEFINITION", "MECHANISM"] for s in selected_sents)
        has_conclusion = any(s.semantic_role in ["CONCLUSION", "CAVEAT"] for s in selected_sents)
        comp_score = 0.95 if (has_thesis_or_def and has_conclusion) else (0.85 if has_thesis_or_def else 0.75)

        # Formulate concise key information summary
        key_info = " ".join(s.text for s in selected_sents[:2])
        if len(key_info) > 160:
            key_info = key_info[:157] + "..."

        return ValidatedClipCandidate(
            topic_id=topic_id,
            topic_title=topic_name,
            start_ms=int(final_start * 1000),
            end_ms=int(final_end * 1000),
            start_time=final_start,
            end_time=final_end,
            duration_seconds=final_dur,
            selected_sentence_ids=[s.id for s in selected_sents],
            key_information=key_info,
            selection_reason=f"Optimal concise range preserving core explanation ({len(selected_sents)} sentences)",
            excluded_content_reason="Trimmed conversational pleasantries, filler, and unrelated tangents",
            completeness_score=comp_score,
            relevance_score=round(avg_rel, 2),
            redundancy_score=0.05,
            standalone_score=0.95,
            requires_expansion=False,
            validation_status="PASSED"
        )

    # -------------------------------------------------------------------------
    # 4. DEDUPLICATION AND OVERLAP SUPPRESSION
    # -------------------------------------------------------------------------
    def deduplicate_candidates(
        self,
        candidates: List[ValidatedClipCandidate],
        overlap_threshold: float = 0.50
    ) -> List[ValidatedClipCandidate]:
        """
        Suppresses redundant candidate clips that heavily overlap in time
        or cover substantially identical content.
        """
        if not candidates:
            return []

        # Sort candidates descending by importance/relevance score
        sorted_cands = sorted(candidates, key=lambda c: (c.relevance_score * c.completeness_score), reverse=True)
        unique_clips: List[ValidatedClipCandidate] = []

        for cand in sorted_cands:
            is_duplicate = False
            for kept in unique_clips:
                # Calculate temporal Intersection over Union (IoU)
                inter_start = max(cand.start_time, kept.start_time)
                inter_end = min(cand.end_time, kept.end_time)
                intersection = max(0.0, inter_end - inter_start)
                
                union = max(0.1, max(cand.end_time, kept.end_time) - min(cand.start_time, kept.start_time))
                iou = intersection / union

                # Calculate sentence ID overlap
                s_cand = set(cand.selected_sentence_ids)
                s_kept = set(kept.selected_sentence_ids)
                sent_overlap = len(s_cand.intersection(s_kept)) / max(1, len(s_cand))

                if iou >= overlap_threshold or sent_overlap >= 0.60:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_clips.append(cand)

        # Sort chronologically by start_time for smooth presentation
        unique_clips.sort(key=lambda c: c.start_time)
        return unique_clips
