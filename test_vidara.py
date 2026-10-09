import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.database import DatabaseService, init_db
from backend.models.schemas import EnrichedTranscript, TranscriptSegment, WordTimestamp, DiscoveredTopic
from backend.services.topic_intelligence import VidaraTopicIntelligenceEngine
from backend.services.meaning_validator import MeaningValidator
from backend.services.video_cutter import VideoCutter

class TestVidaraPlatform(unittest.TestCase):
    def setUp(self):
        init_db()
        self.video_id = "test_vid_01"
        self.sample_segments = [
            TranscriptSegment(
                id=1,
                start=0.0,
                end=8.5,
                text="Today we are going to discuss paging and how the operating system handles virtual memory.",
                speaker="SPEAKER_01",
                words=[]
            ),
            TranscriptSegment(
                id=2,
                start=9.0,
                end=22.0,
                text="Paging divides logical memory into fixed-size pages. Each page is mapped using a page table.",
                speaker="SPEAKER_01",
                words=[]
            ),
            TranscriptSegment(
                id=3,
                start=23.0,
                end=40.5,
                text="The key idea is that the CPU generates a logical address containing a page number and an offset.",
                speaker="SPEAKER_01",
                words=[]
            ),
            TranscriptSegment(
                id=4,
                start=41.0,
                end=58.0,
                text="For example, let's look at address translation. The page table translates page number to frame number, but actually TLB caches this.",
                speaker="SPEAKER_01",
                words=[]
            ),
            TranscriptSegment(
                id=5,
                start=59.0,
                end=75.0,
                text="In conclusion, the takeaway is paging eliminates external fragmentation completely.",
                speaker="SPEAKER_01",
                words=[]
            )
        ]
        self.transcript = EnrichedTranscript(
            video_id=self.video_id,
            filename="lecture_paging.mp4",
            duration_seconds=75.0,
            language="en",
            segments=self.sample_segments
        )
        DatabaseService.save_video(
            video_id=self.video_id,
            filename="lecture_paging.mp4",
            filepath="storage/uploads/test.mp4",
            duration=75.0
        )
        DatabaseService.save_segments(
            self.video_id,
            [s.model_dump() for s in self.sample_segments]
        )

    def test_database_persistence(self):
        """Test video and topic persistence in SQLite."""
        DatabaseService.save_video(
            video_id=self.video_id,
            filename="lecture_paging.mp4",
            filepath="storage/uploads/test.mp4",
            duration=75.0
        )
        vid = DatabaseService.get_video(self.video_id)
        self.assertIsNotNone(vid)
        self.assertEqual(vid["filename"], "lecture_paging.mp4")

        # Save topic
        topic = {
            "id": "t1",
            "name": "Paging & Address Translation",
            "description": "Operating system memory paging mechanism",
            "start_time": 9.0,
            "end_time": 58.0,
            "importance_score": 0.94,
            "confidence": 0.96,
            "subtopics": ["Page Tables", "TLB"],
            "why_selected": ["Complete explanation with worked example"]
        }
        DatabaseService.save_topics(self.video_id, [topic])
        topics = DatabaseService.get_topics(self.video_id)
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["name"], "Paging & Address Translation")
        self.assertEqual(topics[0]["importance_score"], 0.94)

    def test_semantic_segmentation_and_clustering(self):
        """Test segmenting transcript using rhetorical signals and pauses."""
        engine = VidaraTopicIntelligenceEngine()
        sem_segs = engine.segment_transcript(self.transcript)
        self.assertGreaterEqual(len(sem_segs), 1)
        
        # Test candidate topic extraction
        clusters = engine.extract_and_cluster_topics(sem_segs)
        self.assertIsInstance(clusters, list)
        self.assertGreater(len(clusters), 0)

    def test_topic_graph_and_scoring(self):
        """Test NetworkX topic graph generation and multi-feature scoring."""
        engine = VidaraTopicIntelligenceEngine()
        sem_segs = engine.segment_transcript(self.transcript)
        clusters = engine.extract_and_cluster_topics(sem_segs)
        
        G, relations = engine.build_topic_graph(clusters, sem_segs)
        self.assertIsNotNone(G)
        
        scored = engine.score_topics_algorithmically(clusters, G, sem_segs, self.transcript.duration_seconds)
        self.assertGreater(len(scored), 0)
        top_topic = scored[0]
        self.assertIn("importance_score", top_topic)
        self.assertIn("confidence", top_topic)
        self.assertIn("why_selected", top_topic)
        self.assertGreater(top_topic["importance_score"], 0.0)

    def test_semantic_boundary_detection(self):
        """Test semantic boundary detection preserves intro and conclusions."""
        engine = VidaraTopicIntelligenceEngine()
        sem_segs = engine.segment_transcript(self.transcript)
        candidate = {
            "name": "Paging",
            "start_time": 9.0,
            "end_time": 40.0
        }
        start_t, end_t = engine.detect_semantic_boundaries(candidate, sem_segs)
        self.assertLessEqual(start_t, 9.0)
        self.assertGreaterEqual(end_t, 40.0)

    def test_meaning_validator_boundaries(self):
        """Test MeaningValidator boundary verification."""
        validator = MeaningValidator()
        res = validator.validate_topic_boundary(
            [s.model_dump() for s in self.sample_segments],
            "Paging",
            9.0,
            75.0
        )
        self.assertTrue(res["valid"])
        self.assertEqual(res["topic"], "Paging")

    def test_mode_a_query_retrieval(self):
        """Test user query concept expansion and matching."""
        engine = VidaraTopicIntelligenceEngine()
        sem_segs = engine.segment_transcript(self.transcript)
        indexed_topics = [
            DiscoveredTopic(
                id="t1",
                video_id=self.video_id,
                name="Address Translation",
                description="Logical address to physical address translation with page tables",
                start_time=9.0,
                end_time=58.0,
                importance_score=0.92,
                confidence=0.95,
                subtopics=["Page Tables", "TLB", "Logical Address"],
                why_selected=["High relevance"]
            )
        ]
        
        query_res = engine.query_video_topics(
            query="how the operating system translates logical address",
            transcript=self.transcript,
            indexed_topics=indexed_topics,
            semantic_segments=sem_segs
        )
        self.assertIn("understood_concepts", query_res)
        self.assertIn("matched_topics", query_res)
        self.assertGreater(len(query_res["matched_topics"]), 0)

    def test_unrelated_query_not_found(self):
        """Test that random/unrelated questions not in the video return found=False with 0 clips."""
        engine = VidaraTopicIntelligenceEngine()
        sem_segs = engine.segment_transcript(self.transcript)
        indexed_topics = [
            DiscoveredTopic(
                id="t1",
                video_id=self.video_id,
                name="Address Translation",
                description="Logical address to physical address translation with page tables",
                start_time=9.0,
                end_time=58.0,
                importance_score=0.92,
                confidence=0.95,
                subtopics=["Page Tables", "TLB", "Logical Address"],
                why_selected=["High relevance"]
            )
        ]

        query_res = engine.query_video_topics(
            query="how to bake a chocolate cake with flour and sugar in the oven",
            transcript=self.transcript,
            indexed_topics=indexed_topics,
            semantic_segments=sem_segs
        )
        self.assertFalse(query_res.get("found", True))
        reasoning_lower = query_res.get("reasoning", "").lower()
        self.assertTrue(any(phrase in reasoning_lower for phrase in ["no content", "not contain", "no relevant content", "does not discuss"]))

    def test_api_endpoints(self):
        """Test Vidara REST API endpoints via FastAPI TestClient."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)

        # 1. Audiences
        res = client.get("/api/audience/profiles")
        self.assertEqual(res.status_code, 200)

        # 2. Status
        res_stat = client.get(f"/videos/{self.video_id}/status")
        self.assertEqual(res_stat.status_code, 200)
        self.assertEqual(res_stat.json()["video_id"], self.video_id)

        # 3. Topics
        res_topics = client.get(f"/videos/{self.video_id}/topics")
        self.assertEqual(res_topics.status_code, 200)
        self.assertIn("topics", res_topics.json())

        # 4. Mode A Query
        res_query = client.post(
            f"/videos/{self.video_id}/query",
            json={"query": "virtual memory and paging"}
        )
        self.assertEqual(res_query.status_code, 200)
        qdata = res_query.json()
        self.assertIn("understood_concepts", qdata)
        self.assertIn("matched_topics", qdata)

    def test_ai_provider_abstraction(self):
        """Test VideoAnalysisProvider, ReasoningProvider, and build_reasoning_context."""
        from backend.services.ai_providers import (
            EmbeddingProvider,
            VideoAnalysisProvider,
            ReasoningProvider,
            build_reasoning_context,
            estimate_token_count
        )

        # 1. Embedding Provider
        vec1 = EmbeddingProvider.embed_text("operating system page table memory")
        vec2 = EmbeddingProvider.embed_text("virtual memory paging TLB cache")
        vec3 = EmbeddingProvider.embed_text("cooking recipe pasta tomato basil")
        sim_rel = EmbeddingProvider.cosine_similarity(vec1, vec2)
        sim_unrel = EmbeddingProvider.cosine_similarity(vec1, vec3)
        self.assertGreater(sim_rel, sim_unrel)

        # 2. Context Builder & Token Protection
        compact_ctx = build_reasoning_context(
            user_intent="Find everything about paging",
            candidate_topics=[
                {"name": "Paging", "importance_score": 0.95, "relevance_score": 0.98, "start_time": 9.0, "end_time": 58.0}
            ],
            candidate_segments=[
                {"start_time": 9.0, "end_time": 22.0, "text": "Paging divides logical memory into fixed-size pages.", "score": 0.9},
                {"start_time": 23.0, "end_time": 40.5, "text": "The key idea is that the CPU generates a logical address.", "score": 0.85}
            ],
            max_tokens=1000
        )
        self.assertIn("user_intent", compact_ctx)
        self.assertIn("candidate_topics", compact_ctx)
        self.assertIn("supporting_evidence", compact_ctx)
        self.assertLessEqual(estimate_token_count(str(compact_ctx)), 1000)

    def test_vector_similarity_search(self):
        """Test hybrid vector and semantic search in SQLite database."""
        hits = DatabaseService.search_segments(self.video_id, "virtual memory paging", top_k=5)
        self.assertGreater(len(hits), 0)
        self.assertIn("score", hits[0])
        self.assertGreater(hits[0]["score"], 0.0)

    def test_link_downloader_validation(self):
        """Test URL format validation in LinkDownloader."""
        from backend.services.link_downloader import LinkDownloader
        self.assertTrue(LinkDownloader.is_valid_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(LinkDownloader.is_valid_url("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"))
        self.assertFalse(LinkDownloader.is_valid_url("not_a_url"))
        self.assertFalse(LinkDownloader.is_valid_url("ftp://unsupported.com/video.mp4"))
        self.assertFalse(LinkDownloader.is_valid_url(""))

    def test_ingest_url_endpoint_validation(self):
        """Test that /api/videos/ingest-url validates requests."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        res = client.post("/api/videos/ingest-url", json={"url": "invalid_url_string"})
        self.assertEqual(res.status_code, 400)

if __name__ == "__main__":
    unittest.main()

