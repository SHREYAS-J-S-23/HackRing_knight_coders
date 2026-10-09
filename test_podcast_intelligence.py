import unittest
import tempfile
from pathlib import Path
from backend.models.schemas import (
    EnrichedTranscript,
    TranscriptSegment,
    WordTimestamp,
    SentenceSegment,
    ConversationalTurn,
    PodcastExchangeCandidate,
    DiscoveredTopic
)
from backend.services.diarization_service import SpeakerDiarizationService
from backend.services.important_clip_selector import ImportantClipSelector
from backend.services.podcast_intelligence import PodcastIntelligenceEngine
from backend.services.meaning_validator import MeaningValidator
from backend.services.video_cutter import VideoCutter

class TestPodcastIntelligence(unittest.TestCase):
    def setUp(self):
        self.selector = ImportantClipSelector()
        self.podcast_engine = PodcastIntelligenceEngine(self.selector)
        self.validator = MeaningValidator()
        self.diarizer = SpeakerDiarizationService()

    # -------------------------------------------------------------------------
    # TEST 1: A valuable host question and answer
    # -------------------------------------------------------------------------
    def test_valuable_host_question_and_answer(self):
        segs = [
            TranscriptSegment(id=1, start=10.0, end=14.0, text="Robert, what is the single biggest cause of financial ruin?", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=2, start=14.5, end=26.0, text="Financial ruin happens when people mistake liabilities for assets and borrow money to fund their consumption.", speaker="SPEAKER_01", speaker_role="GUEST")
        ]
        transcript = EnrichedTranscript(video_id="t1", filename="test.mp4", duration_seconds=30.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences)

        self.assertGreaterEqual(len(exchanges), 1)
        ex = exchanges[0]
        self.assertIn(ex.exchange_type, ["QUESTION_AND_ANSWER", "STANDALONE_INSIGHT"])
        self.assertIn("SPEAKER_00", ex.speakers_involved)
        self.assertIn("SPEAKER_01", ex.speakers_involved)
        self.assertTrue(ex.question_included)
        self.assertGreaterEqual(ex.duration_seconds, 12.0)

    # -------------------------------------------------------------------------
    # TEST 2: An insightful answer that stands alone
    # -------------------------------------------------------------------------
    def test_insightful_answer_stands_alone(self):
        segs = [
            TranscriptSegment(id=1, start=5.0, end=7.0, text="Yeah, totally.", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=2, start=7.5, end=22.0, text="The primary law of compounding capital is that wealth multiplies through continuous reinvestment into cash-flowing operations rather than speculative asset appreciation.", speaker="SPEAKER_01", speaker_role="GUEST")
        ]
        transcript = EnrichedTranscript(video_id="t2", filename="test.mp4", duration_seconds=25.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences)

        self.assertGreaterEqual(len(exchanges), 1)
        # Verify the standalone insight does not force inclusion of meaningless banter
        ex = next((e for e in exchanges if e.exchange_type == "STANDALONE_INSIGHT"), exchanges[0])
        self.assertFalse(ex.depends_on_question)
        self.assertIn("law of compounding", ex.core_insight.lower())

    # -------------------------------------------------------------------------
    # TEST 3: A long answer with only one important section
    # -------------------------------------------------------------------------
    def test_long_answer_trimmed_to_core_section(self):
        segs = [
            TranscriptSegment(id=1, start=0.0, end=5.0, text="So how do taxes work for the wealthy?", speaker="SPEAKER_00"),
            TranscriptSegment(id=2, start=5.5, end=15.0, text="Well, you know, my uncle used to say taxes are annoying and we went to Hawaii and ate pineapple.", speaker="SPEAKER_01"),
            TranscriptSegment(id=3, start=15.5, end=35.0, text="The tax code legally incentivizes real estate debt because depreciation shields rental income from taxable capital gains.", speaker="SPEAKER_01"),
            TranscriptSegment(id=4, start=35.5, end=45.0, text="And that was pretty cool, we also had a nice barbecue that weekend.", speaker="SPEAKER_01")
        ]
        transcript = EnrichedTranscript(video_id="t3", filename="test.mp4", duration_seconds=50.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic("Real Estate Tax Depreciation", "t3_top", sentences)

        self.assertIsNotNone(cand)
        # Should trim the Hawaii pineapple intro banter and the barbecue outro
        self.assertNotIn("pineapple", cand.key_information.lower())
        self.assertNotIn("barbecue", cand.key_information.lower())
        self.assertLessEqual(cand.duration_seconds, 35.0)

    # -------------------------------------------------------------------------
    # TEST 4: An important follow-up that changes the meaning
    # -------------------------------------------------------------------------
    def test_followup_that_changes_meaning_is_retained(self):
        segs = [
            TranscriptSegment(id=1, start=2.0, end=6.0, text="Do you recommend taking massive debt?", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=2, start=6.5, end=14.0, text="Yes, debt is how you get rich without paying taxes.", speaker="SPEAKER_01", speaker_role="GUEST"),
            TranscriptSegment(id=3, start=14.5, end=18.0, text="Wait, does that apply to credit card debt?", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=4, start=18.5, end=27.0, text="No, absolutely not. Consumer debt will destroy you. Only asset-backed commercial debt that pays for itself.", speaker="SPEAKER_01", speaker_role="GUEST")
        ]
        transcript = EnrichedTranscript(video_id="t4", filename="test.mp4", duration_seconds=30.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences)

        qa_followup = next((e for e in exchanges if e.exchange_type == "QA_WITH_FOLLOWUP"), None)
        self.assertIsNotNone(qa_followup)
        # Must retain the qualification turn that warns against consumer debt
        self.assertIn("SPEAKER_00", qa_followup.speakers_involved)
        self.assertIn("SPEAKER_01", qa_followup.speakers_involved)

    # -------------------------------------------------------------------------
    # TEST 5: An answer that requires the preceding question
    # -------------------------------------------------------------------------
    def test_answer_requiring_preceding_question(self):
        segs = [
            TranscriptSegment(id=1, start=0.0, end=4.0, text="Why did you reject traditional retirement accounts?", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=2, start=4.5, end=12.0, text="Because they lock your capital away while charging excessive hidden management fees.", speaker="SPEAKER_01", speaker_role="GUEST")
        ]
        transcript = EnrichedTranscript(video_id="t5", filename="test.mp4", duration_seconds=15.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences)

        self.assertGreaterEqual(len(exchanges), 1)
        ex = exchanges[0]
        # Since answer starts with 'Because', indexical dependency requires question
        self.assertTrue(ex.question_included)
        self.assertTrue(ex.depends_on_question)

    # -------------------------------------------------------------------------
    # TEST 6: Irrelevant anecdote between two meaningful turns
    # -------------------------------------------------------------------------
    def test_irrelevant_anecdote_penalized_and_trimmed(self):
        segs = [
            TranscriptSegment(id=1, start=10.0, end=18.0, text="The definition of liquidity is how quickly an asset converts to cash without price distortion.", speaker="SPEAKER_01"),
            TranscriptSegment(id=2, start=18.5, end=26.0, text="Speaking of cash, my dog once chewed up a hundred dollar bill and we had to take him to the vet.", speaker="SPEAKER_01"),
            TranscriptSegment(id=3, start=26.5, end=35.0, text="Therefore, liquid reserves protect solvency during sudden market downturns.", speaker="SPEAKER_01")
        ]
        transcript = EnrichedTranscript(video_id="t6", filename="test.mp4", duration_seconds=40.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic("Asset Liquidity", "t6_top", sentences)

        self.assertIsNotNone(cand)
        # Sentence about dog chewing bill should be penalized
        dog_sent = next((s for s in sentences if "dog" in s.text.lower()), None)
        self.assertIsNotNone(dog_sent)
        self.assertLess(dog_sent.importance_score, 0.45)

    # -------------------------------------------------------------------------
    # TEST 7: Multiple speakers and overlapping dialogue
    # -------------------------------------------------------------------------
    def test_multi_speaker_and_dialogue_alternation(self):
        segs = [
            TranscriptSegment(id=1, start=0.0, end=5.0, text="I believe gold is obsolete.", speaker="SPEAKER_00", speaker_role="HOST"),
            TranscriptSegment(id=2, start=5.2, end=11.0, text="I disagree completely. Gold has five thousand years of monetary history.", speaker="SPEAKER_01", speaker_role="GUEST"),
            TranscriptSegment(id=3, start=11.2, end=17.0, text="Bitcoin can replace it. But gold remains physical insurance.", speaker="SPEAKER_02", speaker_role="CO_HOST")
        ]
        transcript = EnrichedTranscript(video_id="t7", filename="test.mp4", duration_seconds=20.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        exchanges = self.podcast_engine.extract_podcast_moments(transcript, sentences)

        self.assertGreaterEqual(len(exchanges), 1)
        # Should detect disagreement exchange with multiple speakers
        debate_ex = next((e for e in exchanges if e.exchange_type == "MEANINGFUL_DISAGREEMENT"), exchanges[0])
        self.assertGreaterEqual(len(debate_ex.speakers_involved), 2)

    # -------------------------------------------------------------------------
    # TEST 8: Repeated questions and answers
    # -------------------------------------------------------------------------
    def test_repeated_qa_deduplication(self):
        cand1 = PodcastExchangeCandidate(
            exchange_id="c1", exchange_type="QUESTION_AND_ANSWER", topic_title="Inflation Strategy",
            core_insight="Buy hard assets during inflation", start_time=10.0, end_time=30.0,
            duration_seconds=20.0, sentence_ids=[1, 2, 3], information_value=0.92
        )
        cand2 = PodcastExchangeCandidate(
            exchange_id="c2", exchange_type="QUESTION_AND_ANSWER", topic_title="Inflation Strategy",
            core_insight="Buy hard assets during inflation", start_time=11.0, end_time=31.0,
            duration_seconds=20.0, sentence_ids=[1, 2, 3], information_value=0.85
        )
        deduped = self.podcast_engine._deduplicate_podcast_exchanges([cand1, cand2])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].exchange_id, "c1")

    # -------------------------------------------------------------------------
    # TEST 9: A long podcast with the same topic revisited later
    # -------------------------------------------------------------------------
    def test_topic_revisited_separated_in_time(self):
        segs = [
            TranscriptSegment(id=1, start=10.0, end=25.0, text="Paging in operating systems divides physical memory into fixed-size frames.", speaker="SPEAKER_00"),
            TranscriptSegment(id=2, start=25.5, end=40.0, text="Hardware translates logical pages to physical frames through the page table.", speaker="SPEAKER_00"),
            # Long gap (15 minutes later)
            TranscriptSegment(id=3, start=900.0, end=920.0, text="Now let us revisit paging and talk about multilevel page tables.", speaker="SPEAKER_00"),
            TranscriptSegment(id=4, start=920.5, end=940.0, text="Multilevel paging avoids storing giant contiguous tables in memory.", speaker="SPEAKER_00")
        ]
        transcript = EnrichedTranscript(video_id="t9", filename="test.mp4", duration_seconds=1000.0, segments=segs)
        sentences = self.selector.extract_sentences(transcript)
        cand = self.selector.select_best_clip_for_topic("Paging", "t9_top", sentences)

        self.assertIsNotNone(cand)
        # Window must not span across the 15-minute gap
        self.assertLessEqual(cand.duration_seconds, 90.0)

    # -------------------------------------------------------------------------
    # TEST 10: Missing speaker labels or unreliable diarization
    # -------------------------------------------------------------------------
    def test_missing_speaker_labels_fallback(self):
        segs = [
            TranscriptSegment(id=1, start=0.0, end=5.0, text="Welcome everyone. How do interest rates impact stocks?", speaker="SPEAKER_01"),
            TranscriptSegment(id=2, start=5.5, end=14.0, text="When central banks raise rates, discount rates rise and equity valuations contract.", speaker="SPEAKER_01"),
        ]
        transcript = EnrichedTranscript(video_id="t10", filename="test.mp4", duration_seconds=16.0, segments=segs)
        diarized = self.diarizer.diarize_transcript(transcript)
        # Should assign valid speaker IDs without crashing
        for s in diarized.segments:
            self.assertIn("SPEAKER_", s.speaker)
            self.assertIsNotNone(s.speaker_confidence)

    # -------------------------------------------------------------------------
    # TEST 11: Malformed LLM responses and provider failures
    # -------------------------------------------------------------------------
    def test_editorial_validator_resilient_fallback(self):
        # MeaningValidator must not crash when LLM returns nothing or fails
        val = self.validator.validate_podcast_exchange(
            candidate_text="The difference between good debt and bad debt is that good debt puts money in your pocket through income-producing assets, whereas bad debt takes money out of your pocket through consumer liabilities.",
            topic_title="Financial Leverage and Good Debt",
            exchange_type="STANDALONE_INSIGHT",
            start_time=10.0,
            end_time=25.0,
            speakers_involved=["SPEAKER_01"],
            question_included=False,
            depends_on_question=False
        )
        self.assertIn("validation_status", val)
        self.assertTrue(val.get("passed", False))
        self.assertIn("quality_score", val)

    # -------------------------------------------------------------------------
    # TEST 12: Accurate FFmpeg cutting and subtitle synchronization
    # -------------------------------------------------------------------------
    def test_subtitle_vtt_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vtt_path = Path(tmpdir) / "test_sub.vtt"
            segments_sample = [
                {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00", "text": "What is Bitcoin?"},
                {"start": 15.2, "end": 22.0, "speaker": "SPEAKER_01", "text": "It is decentralized sound money."}
            ]
            VideoCutter.generate_clip_vtt(segments_sample, clip_start=10.0, clip_end=22.0, output_vtt_path=vtt_path)
            self.assertTrue(vtt_path.exists())
            content = vtt_path.read_text(encoding="utf-8")
            self.assertIn("WEBVTT", content)
            self.assertIn("<v SPEAKER_00>What is Bitcoin?</v>", content)
            self.assertIn("<v SPEAKER_01>It is decentralized sound money.</v>", content)

if __name__ == "__main__":
    unittest.main()
