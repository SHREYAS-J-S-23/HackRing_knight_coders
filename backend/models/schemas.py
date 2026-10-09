from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = "SPEAKER_01"
    speaker_role: Optional[str] = "UNKNOWN"
    speaker_confidence: Optional[float] = 1.0
    words: Optional[List[WordTimestamp]] = []

class EnrichedTranscript(BaseModel):
    video_id: str
    filename: str
    duration_seconds: float
    language: Optional[str] = "en"
    segments: List[TranscriptSegment]

class SegmentAnnotation(BaseModel):
    segment_id: int
    start: float
    end: float
    text: str
    category: Literal["CORE_POINT", "SUPPORTING", "TANGENT", "REPETITION", "FILLER", "TRANSITION", "MISTAKE_CORRECTION"]
    implicit_meaning: str = ""
    critical_context: str = ""
    importance_score: int = Field(ge=1, le=5, default=3)

class TopicTheme(BaseModel):
    title: str
    summary: str
    segment_ids: List[int]
    implicit_takeaway: str = ""

class IntentGraph(BaseModel):
    video_id: str
    core_thesis: str
    themes: List[TopicTheme]
    annotated_segments: List[SegmentAnnotation]
    redundancy_groups: List[List[int]] = []

class AudienceProfile(BaseModel):
    id: str
    name: str
    target_role: str
    description: str
    compression_target: str  # e.g., "70% reduction", "concise 3-minute brief", "full educational"
    priority_topics: List[str] = []
    deprioritize: List[str] = []
    tone: str = "balanced"

class EditDecision(BaseModel):
    segment_id: int
    action: Literal["KEEP", "CUT", "TRIM"]
    start: float
    end: float
    text: str
    reason: str
    meaning_impact: str
    safety_check: Literal["PASS", "WARN", "FLAGGED"] = "PASS"

class EditDecisionList(BaseModel):
    video_id: str
    audience_id: str
    audience_name: str
    decisions: List[EditDecision]
    original_duration: float
    edited_duration: float
    compression_percent: float
    kept_count: int
    cut_count: int

class MeaningValidationReport(BaseModel):
    status: Literal["PASSED", "WARNING", "BLOCKED"]
    violations_found: int
    details: List[dict]
    overall_faithfulness_score: float = 1.0  # 0.0 to 1.0

class RepurposeJobResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    progress: int
    message: str
    transcript: Optional[EnrichedTranscript] = None
    intent_graph: Optional[IntentGraph] = None
    edl: Optional[EditDecisionList] = None
    output_video_url: Optional[str] = None
    validation_report: Optional[MeaningValidationReport] = None

# ====================================================
# VIDARA TOPIC INTELLIGENCE & QUERY SCHEMAS
# ====================================================

class SentenceSegment(BaseModel):
    id: int
    text: str
    start_time: float
    end_time: float
    duration: float = 0.0
    words: Optional[List[WordTimestamp]] = []
    parent_segment_id: int = 1
    speaker: Optional[str] = "SPEAKER_01"
    speaker_role: Optional[str] = "UNKNOWN"  # HOST, GUEST, CO_HOST, PARTICIPANT, UNKNOWN
    speaker_confidence: Optional[float] = 1.0
    semantic_role: Optional[str] = "EXPLANATION"  # DEFINITION, EXPLANATION, MECHANISM, EXAMPLE, CAVEAT, CONCLUSION, INTRO, FILLER, REPETITION
    conversational_unit: Optional[str] = "ANSWER"  # QUESTION, ANSWER, FOLLOW_UP, CLARIFICATION, EXAMPLE_STORY, DISAGREEMENT, CONCLUSION, TRANSITION, DIGRESSION
    preceding_question_id: Optional[int] = None
    depends_on_question: Optional[bool] = False
    information_density: Optional[float] = 0.5
    importance_score: Optional[float] = 0.5
    topic_associations: Optional[List[str]] = []

class SemanticSegment(BaseModel):
    segment_id: int
    start_time: float
    end_time: float
    transcript: str
    embedding: Optional[List[float]] = []
    speaker_id: Optional[str] = "SPEAKER_01"
    speaker_role: Optional[str] = "UNKNOWN"
    candidate_topics: List[str] = []

class ConversationalTurn(BaseModel):
    turn_id: int
    speaker: str
    speaker_role: Optional[str] = "UNKNOWN"
    start_time: float
    end_time: float
    duration: float = 0.0
    text: str
    unit_type: str = "ANSWER"  # QUESTION, ANSWER, FOLLOW_UP, CLARIFICATION, EXAMPLE_STORY, DISAGREEMENT, CONCLUSION, TRANSITION, DIGRESSION
    sentence_ids: List[int] = []
    preceding_question_turn_id: Optional[int] = None
    depends_on_question: bool = False
    is_standalone_insight: bool = False
    key_claim: str = ""

class PodcastExchangeCandidate(BaseModel):
    exchange_id: str
    exchange_type: str = "QUESTION_AND_ANSWER"  # QUESTION_AND_ANSWER, STANDALONE_INSIGHT, QA_WITH_FOLLOWUP, MEANINGFUL_DISAGREEMENT, REVELATION_OR_STORY
    topic_title: str
    core_insight: str
    start_time: float
    end_time: float
    duration_seconds: float
    speakers_involved: List[str] = []
    speaker_roles: dict = {}
    turn_ids: List[int] = []
    sentence_ids: List[int] = []
    question_included: bool = False
    depends_on_question: bool = False
    information_value: float = 0.85
    relevance_score: float = 0.85
    completeness_score: float = 0.90
    redundancy_score: float = 0.05
    editorial_justification: str = ""
    validation_status: str = "PASSED"

class ValidatedClipCandidate(BaseModel):
    topic_id: str
    topic_title: str
    start_ms: int
    end_ms: int
    start_time: float
    end_time: float
    duration_seconds: float
    selected_sentence_ids: List[int] = []
    key_information: str = ""
    selection_reason: str = ""
    excluded_content_reason: Optional[str] = ""
    completeness_score: float = 1.0
    relevance_score: float = 1.0
    redundancy_score: float = 0.0
    standalone_score: float = 1.0
    requires_expansion: bool = False
    validation_status: str = "PASSED"
    exchange_type: Optional[str] = "GENERAL_TOPIC"
    speakers_involved: Optional[List[str]] = []
    speaker_roles: Optional[dict] = {}
    question_included: Optional[bool] = False
    depends_on_question: Optional[bool] = False
    editorial_justification: Optional[str] = ""

class DiscoveredTopic(BaseModel):
    id: str
    video_id: str
    name: str
    description: str
    start_time: float
    end_time: float
    importance_score: float = 0.0
    confidence: float = 0.0
    coverage_score: float = 0.0
    depth_score: float = 0.0
    centrality_score: float = 0.0
    subtopics: List[str] = []
    why_selected: List[str] = []
    segment_ids: List[int] = []
    is_selected: bool = True
    clip_url: Optional[str] = None
    clip_id: Optional[str] = None
    key_information: Optional[str] = ""
    selected_sentence_ids: Optional[List[int]] = []
    rendered_duration: Optional[float] = None
    quality_score: Optional[float] = None
    validation_status: Optional[str] = "PASSED"
    excluded_content_reason: Optional[str] = ""
    exchange_type: Optional[str] = "GENERAL_TOPIC"
    speakers_involved: Optional[List[str]] = []
    speaker_roles: Optional[dict] = {}
    question_included: Optional[bool] = False
    depends_on_question: Optional[bool] = False
    editorial_justification: Optional[str] = ""

class TopicRelation(BaseModel):
    source: str
    target: str
    relation_type: str = "related_to"
    weight: float = 1.0

class TopicDiscoveryResponse(BaseModel):
    status: str
    video_id: str
    total_topics: int
    topics: List[DiscoveredTopic]
    relations: List[TopicRelation] = []

class IngestUrlRequest(BaseModel):
    url: str
    groq_api_key: Optional[str] = None

class UserQueryRequest(BaseModel):
    query: str
    voice_input: Optional[bool] = False

class UserQueryResponse(BaseModel):
    query: str
    found: bool = True
    understood_concepts: List[str] = []
    matched_topics: List[DiscoveredTopic] = []
    reasoning: str = ""
    primary_clip: Optional[dict] = None

class GenerateClipsRequest(BaseModel):
    topic_ids: Optional[List[str]] = None

class MergeClipsRequest(BaseModel):
    clip_ids: Optional[List[str]] = None
    topic_ids: Optional[List[str]] = None

class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    progress: int
    current_stage: str
    message: str
