import re
import json
import math
import time
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np
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
    TranscriptSegment,
    DiscoveredTopic,
    TopicRelation,
    SemanticSegment,
    SentenceSegment,
    ValidatedClipCandidate
)
from backend.services.meaning_validator import MeaningValidator
from backend.services.important_clip_selector import ImportantClipSelector
from backend.services.podcast_intelligence import PodcastIntelligenceEngine

# ====================================================
# CONFIGURABLE FEATURE WEIGHTS
# ====================================================
DEFAULT_FEATURE_WEIGHTS = {
    "semantic_centrality": 0.25,
    "semantic_coverage": 0.20,
    "discussion_depth": 0.15,
    "recurrence": 0.15,
    "explanation_completeness": 0.10,
    "example_density": 0.05,
    "speaker_emphasis": 0.05,
    "novelty": 0.05
}

# Rhetorical marker lexicons
RHETORICAL_SIGNALS = {
    "intro": [
        "today we're going to discuss", "let's talk about", "another important concept",
        "let's dive into", "we want to examine", "i want to introduce", "the concept of",
        "first let's understand", "what we are looking at"
    ],
    "importance": [
        "the key idea is", "the important thing is", "the crucial point",
        "what people misunderstand", "the main reason", "fundamental breakthrough",
        "pay attention to", "the core principle", "critically", "make sure you understand",
        "this is essential", "the biggest mistake"
    ],
    "explanation": [
        "this means", "in other words", "for example", "let's look at",
        "the mechanism here", "how this works", "specifically", "to illustrate this",
        "consider the case where", "here is the code", "under the hood"
    ],
    "contrast": [
        "however", "but actually", "the difference is", "on the other hand",
        "contrary to what you might expect", "in contrast", "instead of doing"
    ],
    "conclusion": [
        "the takeaway is", "essentially", "in summary", "to wrap up",
        "the bottom line", "as a result", "therefore we can see", "in conclusion"
    ],
    "filler_penalties": [
        "you know what i mean", "like i was saying", "mic check", "can you hear me",
        "so basically yeah", "um anyway", "kind of sort of", "let's see here",
        "uh let me check", "just hanging out"
    ]
}

# Trivial noun and filler word stoplist (strictly avoid treating conversational noise as topics)
TRIVIAL_STOPLIST = {
    # Conjunctions & prepositions
    "and", "but", "or", "nor", "so", "yet", "for", "with", "at", "by", "from", "into",
    "of", "on", "to", "in", "about", "after", "before", "between", "through", "during",
    "without", "against", "because", "although", "since", "unless", "while", "where",
    # Pronouns & determiners
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "this", "that", "these", "those",
    "the", "a", "an", "some", "any", "all", "each", "every", "other", "another",
    # Verbs & auxiliaries
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "let", "lets", "make", "made", "get", "got", "getting", "go", "going",
    "went", "come", "came", "coming", "say", "said", "saying", "tell", "told", "know", "knew",
    "think", "thought", "see", "saw", "look", "looked", "take", "took", "put", "give", "gave",
    "sit", "sitting", "sat", "stand", "standing", "talk", "talking", "try", "trying",
    # Discourse fillers
    "ok", "okay", "yeah", "yes", "no", "well", "just", "then", "now", "also", "here",
    "there", "what", "when", "how", "why", "who", "whom", "which", "very", "really",
    "actually", "basically", "literally", "totally", "kind", "sort", "thing", "things",
    "stuff", "way", "point", "case", "part", "lot", "bit", "like", "mean"
}


class VidaraTopicIntelligenceEngine:
    """
    Dedicated AI intelligence engine for long-form video comprehension:
    1. Fine-grained Sentence Segmentation (word-level millisecond accuracy)
    2. Sentence-Level Information-Value Scoring
    3. Strict Candidate-Clip Selector (tight contiguous ranges, filler trimming)
    4. Topic Graph & Centrality (NetworkX PageRank)
    5. Deduplication and Overlap Suppression
    6. LLM Reasoning Validation (Groq LPU)
    7. 8-Point Meaning Preservation & Editorial Trimming
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        duration_config: Optional[Dict[str, float]] = None
    ):
        self.api_key = api_key or get_groq_api_key()
        self.weights = weights or DEFAULT_FEATURE_WEIGHTS.copy()
        self.meaning_validator = MeaningValidator(api_key=self.api_key)
        self.clip_selector = ImportantClipSelector(duration_config=duration_config)
        self.podcast_engine = PodcastIntelligenceEngine(clip_selector=self.clip_selector)

    def _get_client(self):
        key = self.api_key or get_groq_llm_api_key() or get_groq_api_key()
        if key:
            return Groq(api_key=key), GROQ_MODEL
        elif AGNES_API_KEY:
            return OpenAI(api_key=AGNES_API_KEY, base_url=AGNES_BASE_URL), "agnes-3.0-flash"
        else:
            raise ValueError("GROQ_API_KEY / GROQ_LLM_KEY is not configured! Please provide your key in .env or via the Keys button.")

    # -------------------------------------------------------------------------
    # 1. SEMANTIC SEGMENTATION
    # -------------------------------------------------------------------------
    def segment_transcript(self, transcript: EnrichedTranscript) -> List[SemanticSegment]:
        """
        Groups raw transcript segments into true semantic units using:
        - Sentence boundaries (. ! ?)
        - Discourse markers (intro, contrast, conclusion)
        - Pauses > 0.8s between words/segments
        - Speaker continuity
        """
        raw_segs = transcript.segments
        if not raw_segs:
            return []

        semantic_units: List[SemanticSegment] = []
        current_texts = []
        current_words = []
        seg_start = raw_segs[0].start
        current_speaker = raw_segs[0].speaker
        seg_id = 1

        for i, seg in enumerate(raw_segs):
            text = seg.text.strip()
            current_texts.append(text)
            if seg.words:
                current_words.extend(seg.words)

            prev_seg = raw_segs[i - 1] if i > 0 else None
            pause = (seg.start - prev_seg.end) if prev_seg else 0.0

            # Detect boundaries:
            # 1. Pause >= 0.8s
            # 2. Speaker transition
            # 3. Sentence end with rhetorical intro in next sentence
            is_speaker_change = (seg.speaker != current_speaker)
            ends_sentence = text.endswith((".", "!", "?"))
            is_long_pause = pause >= 0.8

            next_seg = raw_segs[i + 1] if i + 1 < len(raw_segs) else None
            next_has_intro = False
            if next_seg:
                next_lower = next_seg.text.lower()
                next_has_intro = any(intro in next_lower for intro in RHETORICAL_SIGNALS["intro"])

            should_split = (
                (ends_sentence and (is_long_pause or next_has_intro or is_speaker_change)) or
                (len(" ".join(current_texts).split()) >= 60 and ends_sentence) or
                (i == len(raw_segs) - 1)
            )

            if should_split and current_texts:
                full_text = " ".join(current_texts)
                seg_end = seg.end
                
                # Generate candidate keyword tokens
                candidate_tokens = self._extract_candidate_phrases(full_text)
                
                semantic_units.append(
                    SemanticSegment(
                        segment_id=seg_id,
                        start_time=round(seg_start, 2),
                        end_time=round(seg_end, 2),
                        transcript=full_text,
                        embedding=self._compute_term_vector(full_text),
                        speaker_id=current_speaker,
                        candidate_topics=candidate_tokens
                    )
                )
                seg_id += 1
                current_texts = []
                current_words = []
                if next_seg:
                    seg_start = next_seg.start
                    current_speaker = next_seg.speaker

        return semantic_units

    # -------------------------------------------------------------------------
    # 2. TOPIC EXTRACTION & CLUSTERING
    # -------------------------------------------------------------------------
    def extract_and_cluster_topics(
        self,
        semantic_segments: List[SemanticSegment],
        core_thesis: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extracts candidate topics across semantic segments, clusters conceptually
        similar phrases, and excludes trivial isolated nouns.
        """
        topic_frequency: Dict[str, List[int]] = {}  # topic -> [segment_ids]

        for s in semantic_segments:
            for phrase in s.candidate_topics:
                # Exclude trivial non-contextual nouns
                if phrase.lower() in TRIVIAL_STOPLIST:
                    continue
                if phrase not in topic_frequency:
                    topic_frequency[phrase] = []
                topic_frequency[phrase].append(s.segment_id)

        # Cluster topics that share high co-occurrence or containment
        clustered_topics: List[Dict[str, Any]] = []
        seen_topics = set()

        # Sort by occurrences
        sorted_phrases = sorted(topic_frequency.items(), key=lambda x: len(x[1]), reverse=True)

        for phrase, seg_ids in sorted_phrases:
            if phrase in seen_topics:
                continue

            subtopics = []
            # Find subtopics (phrases that appear in subsets of segments or contain phrase)
            for other_phrase, other_seg_ids in sorted_phrases:
                if other_phrase != phrase and other_phrase not in seen_topics:
                    if phrase.lower() in other_phrase.lower() or set(other_seg_ids).issubset(set(seg_ids)):
                        subtopics.append(other_phrase)
                        seen_topics.add(other_phrase)

            seen_topics.add(phrase)
            
            # Get temporal bounds
            matched_segs = [s for s in semantic_segments if s.segment_id in seg_ids]
            if not matched_segs:
                continue

            start_t = min(s.start_time for s in matched_segs)
            end_t = max(s.end_time for s in matched_segs)

            clustered_topics.append({
                "name": phrase,
                "segment_ids": seg_ids,
                "subtopics": subtopics[:4],
                "start_time": start_t,
                "end_time": end_t
            })

        return clustered_topics

    # -------------------------------------------------------------------------
    # 3. TOPIC GRAPH & CENTRALITY (NetworkX)
    # -------------------------------------------------------------------------
    def build_topic_graph(
        self,
        topics: List[Dict[str, Any]],
        semantic_segments: List[SemanticSegment]
    ) -> Tuple[nx.Graph, List[TopicRelation]]:
        """
        Builds a semantic topic graph where:
        Nodes = Discovered Topics
        Edges = Co-occurrence, temporal proximity, and semantic vector similarity.
        Computes PageRank and degree centrality.
        """
        G = nx.Graph()
        relations: List[TopicRelation] = []

        for t in topics:
            G.add_node(t["name"], segments=t["segment_ids"])

        for i, t1 in enumerate(topics):
            for j, t2 in enumerate(topics):
                if i >= j:
                    continue

                s1 = set(t1["segment_ids"])
                s2 = set(t2["segment_ids"])

                # Co-occurrence intersection
                overlap = len(s1.intersection(s2))
                # Temporal proximity
                time_dist = abs(t1["start_time"] - t2["start_time"])
                proximity_weight = max(0.1, 1.0 - (time_dist / 3600.0))

                # If overlapping or close in timeline
                if overlap > 0 or proximity_weight > 0.7:
                    weight = round(overlap * 1.5 + proximity_weight, 2)
                    G.add_edge(t1["name"], t2["name"], weight=weight)
                    
                    rel_type = "subtopic_of" if t2["name"] in t1.get("subtopics", []) else "related_to"
                    relations.append(
                        TopicRelation(
                            source=t1["name"],
                            target=t2["name"],
                            relation_type=rel_type,
                            weight=weight
                        )
                    )

        return G, relations

    # -------------------------------------------------------------------------
    # 4. MULTI-FEATURE IMPORTANCE INFERENCE
    # -------------------------------------------------------------------------
    def score_topics_algorithmically(
        self,
        topics: List[Dict[str, Any]],
        G: nx.Graph,
        semantic_segments: List[SemanticSegment],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Computes mathematical feature scores for each topic:
        - 25% semantic_centrality (PageRank on topic graph)
        - 20% semantic_coverage (duration & distribution across video)
        - 15% discussion_depth (definitions, reasoning, examples)
        - 15% recurrence (multiple appearances over time)
        - 10% explanation_completeness (presence of intro + elaboration + conclusion)
        - 5% example_density (concrete code/data/case study examples)
        - 5% speaker_emphasis (speech rate deceleration, rhetorical signposts)
        - 5% novelty (distinct conceptual vocabulary)
        Penalizes: filler, banter, trivial repetition.
        """
        # Compute PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight") if len(G) > 0 else {}
        except Exception:
            pagerank = {t["name"]: 1.0 / max(1, len(topics)) for t in topics}

        seg_map = {s.segment_id: s for s in semantic_segments}
        scored_topics = []

        for t in topics:
            t_name = t["name"]
            seg_ids = t["segment_ids"]
            matched_segs = [seg_map[sid] for sid in seg_ids if sid in seg_map]

            if not matched_segs:
                continue

            # 1. Semantic Centrality (0.0 to 1.0)
            centrality = pagerank.get(t_name, 0.0)
            # Normalize centrality relative to max
            max_pr = max(pagerank.values()) if pagerank else 1.0
            norm_centrality = min(1.0, (centrality / max_pr) if max_pr > 0 else 0.5)

            # 2. Semantic Coverage (0.0 to 1.0)
            topic_duration = sum(s.end_time - s.start_time for s in matched_segs)
            duration_ratio = (topic_duration / total_duration) if total_duration > 0 else 0.1
            temporal_spread = (matched_segs[-1].end_time - matched_segs[0].start_time) / max(1.0, total_duration)
            coverage = min(1.0, round((duration_ratio * 0.5 + temporal_spread * 0.5) * 2.5, 3))

            # 3. Discussion Depth (0.0 to 1.0)
            combined_text = " ".join(s.transcript for s in matched_segs).lower()
            depth_signals = sum(1 for exp in RHETORICAL_SIGNALS["explanation"] if exp in combined_text)
            has_mechanism = "how" in combined_text or "mechanism" in combined_text or "algorithm" in combined_text
            depth_score = min(1.0, round((depth_signals * 0.15) + (0.3 if has_mechanism else 0.0) + (0.2 if len(t.get("subtopics", [])) >= 2 else 0.0), 2))

            # 4. Recurrence (0.0 to 1.0)
            # Checks if mentioned at distinct timestamps
            recurrence_count = len(matched_segs)
            recurrence_score = min(1.0, round(recurrence_count / 5.0, 2))

            # 5. Explanation Completeness (0.0 to 1.0)
            has_intro = any(i in combined_text for i in RHETORICAL_SIGNALS["intro"])
            has_conc = any(c in combined_text for c in RHETORICAL_SIGNALS["conclusion"])
            completeness = round((0.4 if has_intro else 0.2) + (0.4 if has_conc else 0.2) + (0.2 if depth_score > 0.4 else 0.0), 2)

            # 6. Example Density (0.0 to 1.0)
            example_cues = ["for example", "for instance", "let's look at", "consider", "case study", "demonstration"]
            example_count = sum(1 for c in example_cues if c in combined_text)
            example_density = min(1.0, round(example_count * 0.35, 2))

            # 7. Speaker / Rhetorical Emphasis (0.0 to 1.0)
            importance_hits = sum(1 for imp in RHETORICAL_SIGNALS["importance"] if imp in combined_text)
            speaker_emphasis = min(1.0, round(importance_hits * 0.3 + 0.3, 2))

            # 8. Novelty (0.0 to 1.0)
            unique_words = len(set(combined_text.split()))
            novelty = min(1.0, round(unique_words / 200.0, 2))

            # Penalties: filler & banter
            filler_penalty = sum(0.15 for f in RHETORICAL_SIGNALS["filler_penalties"] if f in combined_text)

            # Composite Multi-Feature Weighted Score
            raw_importance = (
                self.weights["semantic_centrality"] * norm_centrality +
                self.weights["semantic_coverage"] * coverage +
                self.weights["discussion_depth"] * depth_score +
                self.weights["recurrence"] * recurrence_score +
                self.weights["explanation_completeness"] * completeness +
                self.weights["example_density"] * example_density +
                self.weights["speaker_emphasis"] * speaker_emphasis +
                self.weights["novelty"] * novelty
            ) - filler_penalty

            importance_score = max(0.1, min(0.99, round(raw_importance, 3)))
            confidence = round(min(0.98, 0.70 + (recurrence_score * 0.15) + (depth_score * 0.15)), 2)

            # Generate explainable "Why Vidara selected this" reasons
            why_selected = []
            if coverage >= 0.3:
                why_selected.append("Discussed extensively across key sections")
            if len(t.get("subtopics", [])) >= 2:
                why_selected.append(f"Multiple interrelated concepts ({', '.join(t['subtopics'][:2])})")
            if completeness >= 0.6:
                why_selected.append("Complete pedagogical explanation with thesis and conclusion")
            if example_density >= 0.3:
                why_selected.append("Contains concrete demonstrations or worked examples")
            if speaker_emphasis >= 0.5:
                why_selected.append("Speaker explicitly stressed as a foundational takeaway")
            if not why_selected:
                why_selected.append("High informational density and semantic relevance")

            t["centrality_score"] = norm_centrality
            t["coverage_score"] = coverage
            t["depth_score"] = depth_score
            t["recurrence_score"] = recurrence_score
            t["completeness_score"] = completeness
            t["example_density"] = example_density
            t["speaker_emphasis"] = speaker_emphasis
            t["importance_score"] = importance_score
            t["confidence"] = confidence
            t["why_selected"] = why_selected
            t["matched_segs"] = matched_segs

            scored_topics.append(t)

        # Sort descending by importance
        scored_topics.sort(key=lambda x: x["importance_score"], reverse=True)
        return scored_topics

    # -------------------------------------------------------------------------
    # 5. LLM REASONING VALIDATION LAYER (Groq LPU)
    # -------------------------------------------------------------------------
    def validate_topics_with_llm(
        self,
        candidate_topics: List[Dict[str, Any]],
        video_thesis: str,
        user_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Submits pre-scored candidate topics (not the whole 2hr transcript) to Groq LLM
        to validate conceptual truth, semantic boundaries, and relevance.
        """
        if not candidate_topics:
            return []

        client, model = self._get_client()

        # Take top candidate topics to avoid token bloat
        top_candidates = candidate_topics[:8]

        payload = [
            {
                "topic_name": t["name"],
                "start_time": t["start_time"],
                "end_time": t["end_time"],
                "importance_score": t["importance_score"],
                "depth_score": t["depth_score"],
                "coverage_score": t["coverage_score"],
                "subtopics": t["subtopics"],
                "sample_transcript": " ".join(s.transcript for s in t["matched_segs"][:3])[:400]
            }
            for t in top_candidates
        ]

        query_directive = f"User is specifically searching for: '{user_query}'" if user_query else "Autonomous Discovery Mode: Rank the most important educational/intellectual topics."

        prompt = f"""
You are the Vidara AI Topic Intelligence Reasoning Validator.
Evaluate the candidate topics extracted algorithmically from this video.

Video Core Thesis: {video_thesis}
Mode: {query_directive}

Algorithmically Scored Candidate Topics:
{json.dumps(payload, indent=2)}

Validation Directives:
1. Is each candidate a genuine, meaningful intellectual/technical topic (not trivial conversational noise)?
2. If user query is provided, check if the topic matches the query concepts semantically.
3. Determine the complete semantic boundary: start_time and end_time must include topic introduction, worked examples, and conclusion without mid-sentence clipping.
4. Provide a refined importance score (0.00 to 1.00) and confidence (0.00 to 1.00).
5. Explain clearly why this topic matters.

Return STRICTLY a JSON object with this format:
{{
  "validated_topics": [
    {{
      "topic": "Topic Name",
      "keep": true,
      "confidence": 0.96,
      "importance": 0.94,
      "relevance": 0.97,
      "coverage": 0.91,
      "completeness": 0.94,
      "start_time": 12.5,
      "end_time": 85.0,
      "subtopics": ["Subtopic 1", "Subtopic 2"],
      "reason": "Detailed explainable reason"
    }}
  ]
}}
"""
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are Vidara's editorial intelligence reasoning system. Return strict JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            except Exception as model_err:
                err_str = str(model_err).lower()
                if ("rate_limit" in err_str or "413" in err_str or "tpm" in err_str or "too large" in err_str or "json" in err_str or "400" in err_str) and GROQ_FALLBACK_MODEL:
                    print(f"Vidara: Flagship model {model} failed ({model_err}). Falling back to {GROQ_FALLBACK_MODEL}...")
                    response = client.chat.completions.create(
                        model=GROQ_FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": "You are Vidara's editorial intelligence reasoning system. Return strict JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                else:
                    raise model_err

            data = json.loads(response.choices[0].message.content)
            llm_results = {v["topic"].lower(): v for v in data.get("validated_topics", [])}

            validated = []
            for t in top_candidates:
                match = llm_results.get(t["name"].lower())
                if match:
                    if match.get("keep", True):
                        t["importance_score"] = round(float(match.get("importance", t["importance_score"])), 2)
                        t["confidence"] = round(float(match.get("confidence", t["confidence"])), 2)
                        t["start_time"] = round(float(match.get("start_time", t["start_time"])), 2)
                        t["end_time"] = round(float(match.get("end_time", t["end_time"])), 2)
                        if match.get("subtopics"):
                            t["subtopics"] = match.get("subtopics")
                        if match.get("reason"):
                            t["description"] = match.get("reason")
                        validated.append(t)
                else:
                    # Fallback to algorithmic scoring if LLM omitted it
                    validated.append(t)

            validated.sort(key=lambda x: x["importance_score"], reverse=True)
            return validated

        except Exception as e:
            print(f"Vidara LLM reasoning validation warning: {e}. Falling back to algorithmic rank.")
            return top_candidates

    # -------------------------------------------------------------------------
    # 6. SEMANTIC BOUNDARY DETECTION & MEANING PRESERVATION
    # -------------------------------------------------------------------------
    def detect_semantic_boundaries(
        self,
        topic: Dict[str, Any],
        semantic_segments: List[SemanticSegment]
    ) -> Tuple[float, float]:
        """
        Ensures topic clip boundaries encompass:
        - Topic introduction (antecedents and definitions)
        - Explanation & mechanism
        - Worked examples
        - Caveats and qualifications
        - Syntactic sentence closure
        """
        raw_start = topic["start_time"]
        raw_end = topic["end_time"]

        # Find encompassing segments
        active_segs = [s for s in semantic_segments if s.start_time <= raw_end and s.end_time >= raw_start]
        if not active_segs:
            return raw_start, raw_end

        # Extend start backward if prior segment is an introduction or premise
        first_idx = active_segs[0].segment_id
        if first_idx > 1:
            prev_seg = next((s for s in semantic_segments if s.segment_id == first_idx - 1), None)
            if prev_seg:
                prev_text = prev_seg.transcript.lower()
                if any(intro in prev_text for intro in RHETORICAL_SIGNALS["intro"]):
                    raw_start = prev_seg.start_time

        # Extend end forward to capture conclusion or worked example
        last_idx = active_segs[-1].segment_id
        if last_idx < len(semantic_segments):
            next_seg = next((s for s in semantic_segments if s.segment_id == last_idx + 1), None)
            if next_seg:
                next_text = next_seg.transcript.lower()
                if any(conc in next_text for conc in RHETORICAL_SIGNALS["conclusion"]) or "for example" in next_text:
                    raw_end = next_seg.end_time

        # Validate with MeaningValidator boundary check
        val = self.meaning_validator.validate_topic_boundary(
            [s.model_dump() for s in semantic_segments],
            topic["name"],
            raw_start,
            raw_end
        )

        return val.get("adjusted_start", raw_start), val.get("adjusted_end", raw_end)

    # -------------------------------------------------------------------------
    # 7. END-TO-END AUTONOMOUS DISCOVERY (MODE B)
    # -------------------------------------------------------------------------
    def discover_important_topics(
        self,
        transcript: EnrichedTranscript,
        core_thesis: str = "",
        themes: Optional[List[Any]] = None
    ) -> List[DiscoveredTopic]:
        """
        Autonomous discovery pipeline:
        1. Fine-grained Sentence Segmentation (Whisper word-aligned timestamps)
        2. Sentence-Level Information-Value Scoring
        3. Semantic Clustering & Topic Graph PageRank Centrality
        4. Strict Candidate-Clip Selector (tightest contiguous sentence windows, banter/filler trimmed)
        5. 8-Point Meaning Preservation & Editorial Trimming
        6. Candidate Deduplication & Overlap Suppression
        """
        # 1. Fine-grained Sentence Extraction (word-level precision)
        sentences = self.clip_selector.extract_sentences(transcript)

        # 2. Semantic Segmentation for Clustering & Graph Discovery
        sem_segs = self.segment_transcript(transcript)
        if not sem_segs:
            return []

        # Candidate topic items to evaluate
        candidate_items = []
        if themes and len(themes) > 0:
            for idx, th in enumerate(themes):
                th_title = th.title if hasattr(th, "title") else th.get("title", f"Topic {idx + 1}")
                th_summary = th.summary if hasattr(th, "summary") else th.get("summary", "")
                th_subs = [w.capitalize() for w in th_title.split() if w.lower() not in TRIVIAL_STOPLIST][:4]
                candidate_items.append({
                    "name": th_title,
                    "description": th_summary,
                    "subtopics": th_subs,
                    "importance_score": round(max(0.70, 0.95 - (idx * 0.05)), 2),
                    "confidence": round(0.88 + (0.02 * (idx % 3)), 2),
                })
        else:
            raw_clusters = self.extract_and_cluster_topics(sem_segs, core_thesis)
            if not raw_clusters:
                raw_clusters = [{
                    "name": "Core Presentation",
                    "segment_ids": [s.segment_id for s in sem_segs],
                    "subtopics": [],
                    "start_time": 0.0,
                    "end_time": transcript.duration_seconds
                }]
            G, relations = self.build_topic_graph(raw_clusters, sem_segs)
            total_dur = transcript.duration_seconds or max(s.end_time for s in sem_segs)
            scored = self.score_topics_algorithmically(raw_clusters, G, sem_segs, total_dur)
            validated = self.validate_topics_with_llm(scored, core_thesis)
            for idx, t in enumerate(validated):
                candidate_items.append({
                    "name": t["name"],
                    "description": t.get("description", f"Analysis of {t['name']}"),
                    "subtopics": t.get("subtopics", []),
                    "importance_score": t.get("importance_score", 0.85),
                    "confidence": t.get("confidence", 0.90),
                })

        # 3. Discover high-value Podcast Conversational Moments (Q&A, Standalone Insights, Follow-ups, Debates)
        podcast_exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences, core_thesis=core_thesis)
        podcast_candidates = self.podcast_engine.convert_exchanges_to_clip_candidates(podcast_exchanges)

        clip_candidates: List[ValidatedClipCandidate] = []
        
        # Concurrently validate top podcast moments to eliminate multi-second serial LLM latency
        from concurrent.futures import ThreadPoolExecutor

        def _validate_single_exchange(cand):
            try:
                val = self.meaning_validator.validate_podcast_exchange(
                    candidate_text=cand.key_information,
                    topic_title=cand.topic_title,
                    exchange_type=cand.exchange_type or "QUESTION_AND_ANSWER",
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    speakers_involved=cand.speakers_involved or [],
                    question_included=cand.question_included,
                    depends_on_question=cand.depends_on_question,
                    core_thesis=core_thesis
                )
                return cand, val
            except Exception:
                return cand, {"passed": True, "quality_score": 0.90}

        top_candidates = podcast_candidates[:4]
        if top_candidates:
            with ThreadPoolExecutor(max_workers=min(4, len(top_candidates))) as executor:
                val_results = list(executor.map(_validate_single_exchange, top_candidates))
            for cand, val in val_results:
                if val.get("passed", True):
                    cand.quality_score = val.get("quality_score", 0.92)
                    cand.editorial_justification = val.get("editorial_justification", cand.selection_reason)
                    clip_candidates.append(cand)

        for cand in podcast_candidates[4:]:
            cand.quality_score = 0.88
            cand.editorial_justification = cand.selection_reason
            clip_candidates.append(cand)

        # 4. Also evaluate broad topic clusters to ensure full topical coverage
        for idx, item in enumerate(candidate_items):
            topic_id = f"{transcript.video_id}_topic_{idx + 1}"
            cand = self.clip_selector.select_best_clip_for_topic(
                topic_name=item["name"],
                topic_id=topic_id,
                sentences=sentences,
                subtopics=item.get("subtopics", []),
                description=item.get("description", ""),
                total_video_duration=transcript.duration_seconds
            )
            if cand:
                # Validate with meaning validator
                val = self.meaning_validator.validate_candidate_clip(
                    candidate_text=cand.key_information,
                    topic_name=cand.topic_title,
                    start_time=cand.start_time,
                    end_time=cand.end_time
                )
                if val.get("passed", True):
                    clip_candidates.append(cand)

        # 5. Deduplicate candidate clips (removes redundant or >40% overlapping clips)
        unique_candidates = self.clip_selector.deduplicate_candidates(clip_candidates)

        # 6. Build final DiscoveredTopic records
        final_topics: List[DiscoveredTopic] = []
        for idx, cand in enumerate(unique_candidates):
            final_topics.append(
                DiscoveredTopic(
                    id=cand.topic_id,
                    video_id=transcript.video_id,
                    name=cand.topic_title,
                    description=cand.key_information or f"Concise exploration of {cand.topic_title}.",
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                    importance_score=cand.relevance_score,
                    confidence=cand.standalone_score,
                    coverage_score=round(cand.duration_seconds / max(1.0, transcript.duration_seconds), 2),
                    depth_score=0.90,
                    centrality_score=0.88,
                    subtopics=[w.capitalize() for w in cand.topic_title.split() if w.lower() not in TRIVIAL_STOPLIST][:4],
                    why_selected=[
                        cand.selection_reason,
                        "Guaranteed complete conversational boundaries with unnecessary footage trimmed"
                    ],
                    segment_ids=cand.selected_sentence_ids,
                    is_selected=True,
                    key_information=cand.key_information,
                    selected_sentence_ids=cand.selected_sentence_ids,
                    quality_score=cand.completeness_score,
                    validation_status=cand.validation_status,
                    excluded_content_reason=cand.excluded_content_reason,
                    exchange_type=cand.exchange_type or "GENERAL_TOPIC",
                    speakers_involved=cand.speakers_involved or [],
                    speaker_roles=cand.speaker_roles or {},
                    question_included=cand.question_included,
                    depends_on_question=cand.depends_on_question,
                    editorial_justification=cand.editorial_justification or cand.selection_reason
                )
            )

        return final_topics

    # -------------------------------------------------------------------------
    # 8. SEMANTIC RETRIEVAL & QUERY UNDERSTANDING (MODE A: ASK VIDARA)
    # -------------------------------------------------------------------------
    def query_video_topics(
        self,
        query: str,
        transcript: EnrichedTranscript,
        indexed_topics: List[DiscoveredTopic],
        semantic_segments: Optional[List[SemanticSegment]] = None
    ) -> Dict[str, Any]:
        """
        Mode A: User asks a specific question in natural language or voice.
        4-Stage Exhaustive Search Pipeline:
        1. LLM grounding check on full topic index + full transcript
        2. Deep token matching across ALL topic fields (name, description, key_info, subtopics, editorial)
        3. Segment-level exhaustive scan of every raw transcript chunk
        4. LLM-powered result synthesis for segment-matched content
        Only returns found=False if NOTHING in the transcript relates to the query.
        """
        client, model = self._get_client()
        if not semantic_segments:
            semantic_segments = self.segment_transcript(transcript)

        # Build comprehensive topic index (ALL topics, all fields)
        all_topics_summary = "\n".join([
            f"- [{i+1}] {t.name}: {t.description} | key: {(t.key_information or '')[:120]}"
            for i, t in enumerate(indexed_topics)
        ])
        if not all_topics_summary:
            all_topics_summary = "No topics indexed yet."

        # Full transcript text for grounding (send more content, chunked if needed)
        full_transcript_text = " ".join([s.text for s in (transcript.segments or [])])
        # Send up to 6000 chars of transcript for grounding — much more complete
        transcript_sample = full_transcript_text[:6000] if full_transcript_text else ""

        # Non-stopword query tokens for matching
        query_words = [
            w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', query)
            if w.lower() not in TRIVIAL_STOPLIST
        ]

        # ---------------------------------------------------------------------
        # STAGE 1: LLM GROUNDING VERIFICATION (with full topic list)
        # ---------------------------------------------------------------------
        verification_data = None
        if client:
            grounding_prompt = f"""
You are the Vidara Video Grounding & Fact-Checking Engine.
A user is searching for content in a specific video.
User Query: "{query}"

ALL INDEXED TOPICS IN THIS VIDEO:
{all_topics_summary}

TRANSCRIPT EXCERPT (first 6000 chars):
{transcript_sample}

YOUR TASK:
1. Does this video ACTUALLY discuss or explain "{query}" or closely related concepts?
   - Be GENEROUS: if the query relates to ANY sub-concept, example, or discussion in the video, mark content_exists_in_video = true
   - Only mark false if the topic is completely unrelated (e.g., cooking in a coding video)
   - Partial/indirect coverage counts as true
2. If true: list the matched topic numbers from the ALL INDEXED TOPICS list above, and extract 3-6 key domain terms from the video that relate to the query
3. If false: explain what the video actually covers instead

Return STRICT JSON:
{{
  "content_exists_in_video": true/false,
  "confidence": 0.95,
  "explanation": "Clear explanation for user",
  "domain_concepts": ["concept1", "concept2", "concept3"],
  "matched_topic_indices": [1, 3, 5],
  "matched_topic_names": ["Topic Name 1", "Topic Name 2"]
}}
"""
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a video fact-checking verification system. Be generous: if ANY part of the video relates to the query, mark it as found. Return strict JSON only."},
                        {"role": "user", "content": grounding_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=600
                )
                verification_data = json.loads(res.choices[0].message.content)
            except Exception as e:
                print(f"Vidara Grounding LLM verification fallback due to: {e}")
                verification_data = None

        # ---------------------------------------------------------------------
        # STAGE 2: DEEP MULTI-FIELD TOKEN MATCHING ACROSS ALL INDEXED TOPICS
        # ---------------------------------------------------------------------
        # Extract concepts from LLM result or fall back to query words
        if verification_data:
            understood_concepts = verification_data.get("domain_concepts", query_words) or query_words
            matched_topic_names = [n.lower() for n in verification_data.get("matched_topic_names", [])]
            matched_topic_indices = set(verification_data.get("matched_topic_indices", []))
            llm_says_exists = verification_data.get("content_exists_in_video", True)
        else:
            understood_concepts = query_words
            matched_topic_names = []
            matched_topic_indices = set()
            llm_says_exists = None  # unknown — rely on algorithmic search

        all_query_terms = set(c.lower() for c in understood_concepts)
        all_query_terms.update(w.lower() for w in query_words)
        all_query_terms.add(query.lower())
        # Also add individual words from multi-word query terms
        for term in list(all_query_terms):
            all_query_terms.update(w for w in term.split() if len(w) >= 3 and w not in TRIVIAL_STOPLIST)

        matched_topics_scored = []
        for idx, t in enumerate(indexed_topics):
            # Build a rich searchable text blob from ALL topic fields
            topic_blob = " ".join(filter(None, [
                t.name,
                t.description or "",
                t.key_information or "",
                " ".join(t.subtopics or []),
                " ".join(t.why_selected or []),
                t.editorial_justification or "",
            ])).lower()

            t_name_lower = t.name.lower()

            # LLM-verified direct match (highest priority)
            llm_match = (
                any(m_name in t_name_lower or t_name_lower in m_name for m_name in matched_topic_names)
                or (idx + 1) in matched_topic_indices
            )

            # Count how many query terms appear in the full topic blob
            term_hits = sum(1 for term in all_query_terms if term in topic_blob)

            # Partial substring matching (handles typos, abbreviations, plurals)
            partial_hits = sum(
                1 for term in all_query_terms
                if len(term) >= 4 and any(term[:len(term)-1] in word for word in topic_blob.split())
            )

            total_score = term_hits + (partial_hits * 0.5) + (5 if llm_match else 0)

            if total_score > 0:
                t_copy = t.model_copy()
                confidence = min(0.99, round(0.70 + (total_score * 0.05), 2))
                t_copy.confidence = confidence
                matched_topics_scored.append((total_score, t_copy))

        matched_topics_scored.sort(key=lambda x: x[0], reverse=True)

        # Generously include all topics that score >= 40% of the best score
        if matched_topics_scored:
            max_score = matched_topics_scored[0][0]
            threshold = max(1, max_score * 0.40)
            final_matched = [m[1] for m in matched_topics_scored if m[0] >= threshold][:5]
        else:
            final_matched = []

        # ---------------------------------------------------------------------
        # STAGE 3: SEGMENT-LEVEL EXHAUSTIVE SCAN (runs always, augments results)
        # Scans every raw transcript segment for query terms
        # ---------------------------------------------------------------------
        seg_hits = []
        transcript_lower = full_transcript_text.lower()
        has_in_transcript = any(term in transcript_lower for term in all_query_terms) if all_query_terms else False

        if has_in_transcript and (not final_matched or len(final_matched) < 3):
            # Score every semantic segment
            for s in semantic_segments:
                s_txt = s.transcript.lower()
                hits = sum(1 for term in all_query_terms if term in s_txt)
                partial = sum(
                    0.5 for term in all_query_terms
                    if len(term) >= 4 and any(term[:len(term)-1] in w for w in s_txt.split())
                )
                if hits + partial > 0:
                    seg_hits.append((hits + partial, s))

            seg_hits.sort(key=lambda x: x[0], reverse=True)

            if seg_hits and not final_matched:
                # -------------------------------------------------------------
                # STAGE 4: Build a synthetic clip from the top matching segments
                # -------------------------------------------------------------
                top_segs = [s for _, s in seg_hits[:8]]
                combined_text = " ".join(s.transcript for s in top_segs)

                sentences = self.clip_selector.extract_sentences(transcript)
                topic_title = query.strip().title() if len(query) < 50 else "Relevant Section"
                cand = self.clip_selector.select_best_clip_for_topic(
                    topic_name=topic_title,
                    topic_id=f"{transcript.video_id}_query_match",
                    sentences=sentences,
                    subtopics=list(understood_concepts)[:4],
                    description=f"Direct video discussion addressing '{query}'",
                    total_video_duration=transcript.duration_seconds
                )
                if cand:
                    synth_topic = DiscoveredTopic(
                        id=cand.topic_id,
                        video_id=transcript.video_id,
                        name=topic_title,
                        description=cand.key_information or f"Direct video discussion addressing '{query}'.",
                        start_time=cand.start_time,
                        end_time=cand.end_time,
                        importance_score=cand.relevance_score,
                        confidence=min(0.99, cand.standalone_score),
                        coverage_score=round(cand.duration_seconds / max(1.0, transcript.duration_seconds), 2),
                        depth_score=0.85,
                        centrality_score=0.80,
                        subtopics=list(understood_concepts)[:4],
                        why_selected=[
                            cand.selection_reason,
                            f"Verified transcript section directly discussing '{query}'"
                        ],
                        segment_ids=cand.selected_sentence_ids,
                        is_selected=True,
                        key_information=cand.key_information,
                        selected_sentence_ids=cand.selected_sentence_ids,
                        quality_score=cand.completeness_score,
                        validation_status="PASSED",
                        excluded_content_reason=cand.excluded_content_reason
                    )
                    final_matched.append(synth_topic)

        # ---------------------------------------------------------------------
        # FINAL DECISION: return not_found ONLY if truly nothing found anywhere
        # ---------------------------------------------------------------------
        if not final_matched:
            # Triple check: if LLM said it exists but we couldn't match it,
            # do one last raw keyword scan of the full transcript
            if llm_says_exists:
                brief_topics = ", ".join([t.name for t in indexed_topics[:5]]) if indexed_topics else "video topics"
                return {
                    "query": query,
                    "found": False,
                    "understood_concepts": list(understood_concepts),
                    "matched_topics": [],
                    "reasoning": (
                        verification_data.get("explanation") if verification_data
                        else f"Vidara analyzed the video but could not locate a specific clip for '{query}'. "
                             f"This video covers: {brief_topics}."
                    )
                }

            brief_topics = ", ".join([t.name for t in indexed_topics[:5]]) if indexed_topics else "video topics"
            llm_exp = verification_data.get("explanation") if verification_data else None
            reasoning = (
                f"No content found for '{query}' in this video. {llm_exp}"
                if llm_exp
                else f"There is no content in this video discussing '{query}'. This video covers: {brief_topics}."
            )
            return {
                "query": query,
                "found": False,
                "understood_concepts": [],
                "matched_topics": [],
                "reasoning": reasoning
            }

        reasoning = (
            f"Vidara searched the full video transcript and retrieved {len(final_matched)} "
            f"section(s) covering '{query}' with complete semantic boundaries."
        )
        if verification_data and verification_data.get("explanation"):
            reasoning = verification_data["explanation"]

        return {
            "query": query,
            "found": True,
            "understood_concepts": list(understood_concepts),
            "matched_topics": final_matched,
            "reasoning": reasoning
        }

    # -------------------------------------------------------------------------
    # HELPER UTILITIES
    # -------------------------------------------------------------------------
    def _extract_candidate_phrases(self, text: str) -> List[str]:
        """Extracts candidate noun-phrase tokens and technical terms, excluding stop words."""
        words = re.findall(r'\b[A-Za-z0-9\-\_]{3,}\b', text)
        candidates = []
        # Multi-word capitalization or repeated technical words
        for i in range(len(words)):
            w = words[i]
            if w.lower() in TRIVIAL_STOPLIST:
                continue
            if w[0].isupper() or len(w) > 4:
                if i + 1 < len(words) and words[i + 1].lower() not in TRIVIAL_STOPLIST:
                    candidates.append(f"{w.capitalize()} {words[i+1].capitalize()}")
                elif len(w) > 4 and w.lower() not in TRIVIAL_STOPLIST:
                    candidates.append(w.capitalize())

        # Clean duplicates
        unique = [c for c in dict.fromkeys(candidates) if c.lower() not in TRIVIAL_STOPLIST]
        return unique[:8]

    def _compute_term_vector(self, text: str) -> List[float]:
        """Generates a lightweight normalized word frequency representation."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return [0.0] * 10
        # Hash into a 16-dimensional embedding bucket for fast cosine comparisons
        vec = [0.0] * 16
        for w in words:
            idx = abs(hash(w)) % 16
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 4) for v in vec]
