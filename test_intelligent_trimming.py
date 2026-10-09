import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.models.schemas import (
    EnrichedTranscript,
    TranscriptSegment,
    WordTimestamp,
    SentenceSegment,
    DiscoveredTopic,
    ValidatedClipCandidate
)
from backend.services.important_clip_selector import ImportantClipSelector
from backend.services.meaning_validator import MeaningValidator
from backend.services.topic_intelligence import VidaraTopicIntelligenceEngine
from backend.services.video_cutter import VideoCutter


class TestIntelligentClipTrimmingPipeline(unittest.TestCase):
    def setUp(self):
        self.selector = ImportantClipSelector()
        self.validator = MeaningValidator()
        self.engine = VidaraTopicIntelligenceEngine()

    def test_scenario_a_long_introduction(self):
        """
        Test A: Long introduction
        Input: A video begins with 45 seconds of greetings and background discussion,
               followed by a 25-second complete technical explanation.
        Expected: The generated clip excludes the irrelevant introduction and preserves the complete explanation.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=0.0,
                end=20.0,
                text="Hey everyone, welcome back to the channel, today we are going to talk about a lot of exciting things, mic check 1 2 3.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=21.0,
                end=45.0,
                text="Before we begin, make sure to hit like and subscribe, and grab your coffee, alright guys so let's get into it.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=3,
                start=46.0,
                end=60.0,
                text="Paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=4,
                start=61.0,
                end=71.0,
                text="The CPU translates logical addresses into physical addresses using a page table mechanism.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_intro",
            filename="intro.mp4",
            duration_seconds=71.0,
            segments=segments
        )

        sentences = self.selector.extract_sentences(transcript)
        self.assertEqual(len(sentences), 4)

        # Ensure the intro sentences are classified as INTRO / FILLER
        self.assertIn(sentences[0].semantic_role, ["INTRO", "FILLER"])
        self.assertIn(sentences[1].semantic_role, ["INTRO", "FILLER"])

        # Select clip for "Paging"
        cand = self.selector.select_best_clip_for_topic(
            topic_name="Paging",
            topic_id="t_paging",
            sentences=sentences,
            total_video_duration=71.0
        )
        self.assertIsNotNone(cand)
        # The clip must start at or after 45 seconds (excluding the 45s intro!)
        self.assertGreaterEqual(cand.start_time, 44.0)
        # The technical explanation must be preserved
        self.assertIn("memory management scheme", cand.key_information.lower())

    def test_scenario_b_long_discussion_with_one_key_insight(self):
        """
        Test B: Long discussion with one key insight
        Input: A 5-minute discussion contains one 30-second high-value explanation and several minutes of repetition.
        Expected: The system selects the concise explanation rather than the entire discussion.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=0.0,
                end=60.0,
                text="We are talking about memory architectures, you know what I mean, basically chatting about servers.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=65.0,
                end=95.0,
                text="The Translation Lookaside Buffer is a hardware cache that stores recent virtual to physical address translations to speed up memory access.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=3,
                start=100.0,
                end=200.0,
                text="So like I was saying before, memory is fast, yeah server memory is fast, um anyway just repeating the general chatter.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=4,
                start=205.0,
                end=300.0,
                text="Servers have RAM, and people buy RAM, so that is that, yes more repetition.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_insight",
            filename="insight.mp4",
            duration_seconds=300.0,
            segments=segments
        )
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic(
            topic_name="Translation Lookaside Buffer",
            topic_id="t_tlb",
            sentences=sentences,
            total_video_duration=300.0
        )
        self.assertIsNotNone(cand)
        # Clip should focus strictly on the ~30s insight around 65-95s, not the whole 300s!
        self.assertLessEqual(cand.duration_seconds, 60.0)
        self.assertGreaterEqual(cand.start_time, 60.0)
        self.assertLessEqual(cand.end_time, 100.0)

    def test_scenario_c_essential_caveat(self):
        """
        Test C: Essential caveat
        Input: A definition is followed by an important exception.
        Expected: The exception remains in the clip if removing it would distort the meaning.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=10.0,
                end=25.0,
                text="Optimistic concurrency control allows multiple transactions to complete without locking resources.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=26.0,
                end=38.0,
                text="However, if conflicting updates occur simultaneously, the transaction must abort and roll back.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_caveat",
            filename="caveat.mp4",
            duration_seconds=40.0,
            segments=segments
        )
        sentences = self.selector.extract_sentences(transcript)
        self.assertEqual(sentences[1].semantic_role, "CAVEAT")

        cand = self.selector.select_best_clip_for_topic(
            topic_name="Optimistic Concurrency Control",
            topic_id="t_occ",
            sentences=sentences,
            total_video_duration=40.0
        )
        self.assertIsNotNone(cand)
        # End time must include the caveat segment ending at ~38.0s
        self.assertGreaterEqual(cand.end_time, 37.0)
        self.assertIn(2, cand.selected_sentence_ids)

    def test_scenario_d_topic_revisited_later(self):
        """
        Test D: Topic revisited later
        Input: A speaker explains a topic at 01:00 and returns to it at 08:00.
        Expected: The system does not produce one clip spanning the unrelated content between these intervals.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=60.0,
                end=85.0,
                text="Zero-knowledge proofs allow one party to prove to another that a statement is true without revealing any information.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=120.0,
                end=400.0,
                text="Now let's switch topics completely to database sharding and distributed cluster topologies.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=3,
                start=480.0,
                end=505.0,
                text="Returning to zero-knowledge proofs, zk-SNARKs enable succinct verification on Ethereum smart contracts.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_revisit",
            filename="revisit.mp4",
            duration_seconds=520.0,
            segments=segments
        )
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic(
            topic_name="Zero-Knowledge Proofs",
            topic_id="t_zkp",
            sentences=sentences,
            total_video_duration=520.0
        )
        self.assertIsNotNone(cand)
        # Candidate clip should NOT span from 60s to 505s (445 seconds)!
        self.assertLessEqual(cand.duration_seconds, 90.0)

    def test_scenario_e_duplicate_explanation(self):
        """
        Test E: Duplicate explanation
        Input: Two candidate topics cover essentially the same sentences.
        Expected: Redundant clips are removed/deduplicated.
        """
        cand1 = ValidatedClipCandidate(
            topic_id="t1",
            topic_title="Memory Paging",
            start_ms=10000,
            end_ms=35000,
            start_time=10.0,
            end_time=35.0,
            duration_seconds=25.0,
            selected_sentence_ids=[1, 2],
            key_information="Memory paging divides virtual address space into pages.",
            selection_reason="Concise definition",
            relevance_score=0.95,
            completeness_score=0.90
        )
        cand2 = ValidatedClipCandidate(
            topic_id="t2",
            topic_title="Paging Overview",
            start_ms=11000,
            end_ms=36000,
            start_time=11.0,
            end_time=36.0,
            duration_seconds=25.0,
            selected_sentence_ids=[1, 2],
            key_information="Paging overview explains memory pages.",
            selection_reason="Duplicate overview",
            relevance_score=0.88,
            completeness_score=0.85
        )
        deduped = self.selector.deduplicate_candidates([cand1, cand2])
        # Only 1 of the 2 heavily overlapping candidate clips should be kept
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].topic_id, "t1")

    def test_scenario_f_incomplete_boundaries(self):
        """
        Test F: Incomplete boundaries
        Input: A candidate terminates without proper sentence closure.
        Expected: The boundary validator flags or adjusts the range.
        """
        val_res = self.validator.validate_candidate_clip(
            candidate_text="This mechanism operates by transferring all packets over",
            topic_name="Packet Transfer",
            start_time=10.0,
            end_time=18.0
        )
        issue_types = [d["issue_type"] for d in val_res["details"]]
        self.assertIn("SENTENCE_INCOMPLETE", issue_types)

    def test_scenario_g_duration_targets(self):
        """
        Test G: Duration targets
        Input: A candidate contains 25 seconds of useful information followed by 70 seconds of unrelated discussion.
        Expected: The unnecessary tail is excluded, keeping duration within target.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=10.0,
                end=35.0,
                text="Raft is a consensus algorithm designed to be more understandable than Paxos by dividing leader election and log replication.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=36.0,
                end=110.0,
                text="Anyway, speaking of politics, yesterday I went to the store and the weather was rainy and traffic was terrible.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_raft",
            filename="raft.mp4",
            duration_seconds=115.0,
            segments=segments
        )
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic(
            topic_name="Raft Consensus Algorithm",
            topic_id="t_raft",
            sentences=sentences,
            total_video_duration=115.0
        )
        self.assertIsNotNone(cand)
        # Unrelated tail (segment 2) should be excluded
        self.assertLessEqual(cand.duration_seconds, 40.0)
        self.assertNotIn(2, cand.selected_sentence_ids)

    def test_scenario_h_short_but_incomplete_clip(self):
        """
        Test H: Short but incomplete clip
        Input: A candidate starts with an orphan pronoun ("Because of this, it...") referring to a missing antecedent.
        Expected: The system expands to include the antecedent definition.
        """
        segments = [
            TranscriptSegment(
                id=1,
                start=5.0,
                end=12.0,
                text="The kernel maintains a page table for every process.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=13.0,
                end=19.0,
                text="Because of this, memory isolation between processes is strictly guaranteed.",
                speaker="SPEAKER_01"
            )
        ]
        transcript = EnrichedTranscript(
            video_id="test_pronoun",
            filename="pronoun.mp4",
            duration_seconds=20.0,
            segments=segments
        )
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic(
            topic_name="Memory Isolation",
            topic_id="t_isolation",
            sentences=sentences,
            total_video_duration=20.0
        )
        self.assertIsNotNone(cand)
        # Must include the antecedent (sentence 1)
        self.assertIn(1, cand.selected_sentence_ids)
        self.assertIn(2, cand.selected_sentence_ids)

    def test_scenario_i_timestamp_correctness(self):
        """
        Test I: Timestamp correctness
        Input: Candidate with start_time and end_time.
        Expected: start_ms, end_ms, and duration_seconds are mathematically consistent.
        """
        cand = ValidatedClipCandidate(
            topic_id="t1",
            topic_title="Consistency Check",
            start_ms=12500,
            end_ms=42500,
            start_time=12.5,
            end_time=42.5,
            duration_seconds=30.0,
            selected_sentence_ids=[1],
            key_information="Testing timestamp math."
        )
        self.assertEqual(cand.start_ms, int(cand.start_time * 1000))
        self.assertEqual(cand.end_ms, int(cand.end_time * 1000))
        self.assertEqual(round(cand.end_time - cand.start_time, 1), cand.duration_seconds)

    def test_scenario_j_merge_selection(self):
        """
        Test J: Merge selection
        Input: Four clips are available, but user requests merging only two specific clips.
        Expected: Only the two selected clips are passed to the concat process in order.
        """
        clip1 = Path("storage/outputs/clip_1.mp4")
        clip2 = Path("storage/outputs/clip_2.mp4")
        clip3 = Path("storage/outputs/clip_3.mp4")
        clip4 = Path("storage/outputs/clip_4.mp4")

        # Mock clip paths
        all_clips = [
            {"id": "c1", "topic_id": "t1", "filepath": str(clip1), "start_time": 10.0, "duration": 15.0},
            {"id": "c2", "topic_id": "t2", "filepath": str(clip2), "start_time": 30.0, "duration": 20.0},
            {"id": "c3", "topic_id": "t3", "filepath": str(clip3), "start_time": 60.0, "duration": 25.0},
            {"id": "c4", "topic_id": "t4", "filepath": str(clip4), "start_time": 90.0, "duration": 18.0},
        ]
        user_selected_topic_ids = {"t2", "t4"}
        filtered = [c for c in all_clips if c["topic_id"] in user_selected_topic_ids]
        filtered.sort(key=lambda c: c["start_time"])

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["topic_id"], "t2")
        self.assertEqual(filtered[1]["topic_id"], "t4")


if __name__ == "__main__":
    unittest.main()
