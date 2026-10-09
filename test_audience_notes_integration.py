import sys
import unittest
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.database import DatabaseService, init_db
from backend.models.schemas import (
    EnrichedTranscript,
    TranscriptSegment,
    ClipNotesRequest,
    ClipNotesResponse
)
from backend.services.notes_service import ClipNotesService
from fastapi.testclient import TestClient
from backend.main import app

from backend.services.auth_service import AuthService
from backend.database import get_db_connection

class TestAudienceAndNotesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.user_id = "test_user_owner"
        cls.other_user_id = "test_user_intruder"
        cls.video_id = "test_vid_audience_01"
        cls.clip_id = "clip_audience_101"

        cls.token_owner = AuthService.create_fallback_jwt(cls.user_id, {"email": "owner@vidara.ai"})
        cls.owner_headers = {"Authorization": f"Bearer {cls.token_owner}"}

        # Clear any prior test entries in clip_notes
        conn = get_db_connection()
        conn.execute("DELETE FROM clip_notes WHERE clip_id LIKE 'clip_%'")
        conn.commit()
        conn.close()

        # Register/save test video
        DatabaseService.save_video(
            video_id=cls.video_id,
            filename="lecture_paging.mp4",
            filepath="storage/uploads/test_paging.mp4",
            duration=90.0,
            user_id=cls.user_id,
            audience_mode="professional"
        )

        # Segments
        cls.segments = [
            TranscriptSegment(
                id=1,
                start=0.0,
                end=15.0,
                text="In this lecture, we introduce operating system virtual memory and the paging architecture.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=2,
                start=16.0,
                end=35.0,
                text="Paging avoids external fragmentation by allocating fixed-size physical frames and mapping logical pages via page tables.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=3,
                start=36.0,
                end=55.0,
                text="The Translation Lookaside Buffer, or TLB, provides hardware acceleration for address translation caching.",
                speaker="SPEAKER_01"
            ),
            TranscriptSegment(
                id=4,
                start=56.0,
                end=85.0,
                text="To summarize, paging with TLB hardware caching delivers optimal memory utilization with near zero latency overhead.",
                speaker="SPEAKER_01"
            )
        ]
        DatabaseService.save_segments(cls.video_id, [s.model_dump() for s in cls.segments])

        # Register clip
        cls.clip_obj = {
            "id": cls.clip_id,
            "video_id": cls.video_id,
            "clip_index": 1,
            "filename": "clip_01.mp4",
            "filepath": "storage/clips/clip_01.mp4",
            "title": "Paging and TLB Hardware Acceleration",
            "start_time": 16.0,
            "end_time": 55.0,
            "duration": 39.0,
            "importance_score": 0.95,
            "confidence": 0.92,
            "subtopics": ["Page Tables", "TLB"],
            "why_selected": ["Explains core hardware caching mechanism"],
            "download_url": "/api/clips/download/test.mp4",
            "video_url": "/api/clips/preview/test.mp4"
        }
        DatabaseService.save_clips(cls.video_id, [cls.clip_obj])

    def test_01_audience_mode_persistence_and_retrieval(self):
        """Verify audience_mode is saved and retrieved from DB."""
        video = DatabaseService.get_video(self.video_id)
        self.assertIsNotNone(video)
        self.assertEqual(video.get("audience_mode"), "professional")

        # Update audience_mode to content_creator
        DatabaseService.save_video(
            video_id=self.video_id,
            filename="lecture_paging.mp4",
            filepath="storage/uploads/test_paging.mp4",
            duration=90.0,
            user_id=self.user_id,
            audience_mode="content_creator"
        )
        updated = DatabaseService.get_video(self.video_id)
        self.assertEqual(updated.get("audience_mode"), "content_creator")

    def test_02_audience_mode_validation_in_endpoints(self):
        """Verify invalid audience_modes are rejected and valid ones accepted."""
        # 1. Invalid mode in ingest URL
        res = self.client.post("/api/videos/ingest-url", json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "audience_mode": "invalid_mode_xyz"
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid audience_mode", res.json()["detail"])

        # 2. Invalid mode in notes generation
        res_notes_invalid = self.client.post(f"/api/clips/{self.clip_id}/notes", json={
            "audience_mode": "super_student"
        })
        self.assertEqual(res_notes_invalid.status_code, 400)
        self.assertIn("Invalid audience_mode", res_notes_invalid.json()["detail"])

    def test_03_notes_service_transcript_segmentation_isolation(self):
        """Verify notes only extract transcript overlapping the clip boundaries."""
        service = ClipNotesService()
        segments = service.get_overlapping_transcript(self.video_id, 16.0, 55.0)
        clip_transcript = service.format_transcript_text(segments)
        self.assertIn("Paging avoids external fragmentation", clip_transcript)
        self.assertIn("Translation Lookaside Buffer", clip_transcript)
        # Should NOT include introductory segment (0.0 to 15.0)
        self.assertNotIn("In this lecture, we introduce", clip_transcript)
        # Should NOT include concluding segment (56.0 to 85.0)
        self.assertNotIn("To summarize, paging with TLB hardware caching", clip_transcript)

    def test_04_education_notes_generation_and_schema(self):
        """Verify Education mode generates required sections and conforms to schema."""
        service = ClipNotesService()
        notes = service.generate_notes_for_clip(
            video_id=self.video_id,
            clip=self.clip_obj,
            audience_mode="education",
            user_id=self.user_id
        )
        self.assertEqual(notes["audience_mode"], "education")
        self.assertEqual(notes["clip_id"], self.clip_id)
        self.assertIn("clip_summary", notes)
        self.assertIn("key_points", notes)
        self.assertIsInstance(notes["key_points"], list)
        
        # Education specific sections
        sections = notes.get("sections", {})
        self.assertIn("clip_overview", sections)
        self.assertIn("key_concepts", sections)
        self.assertIn("important_points", sections)
        self.assertIn("revision_notes", sections)
        self.assertIn("practice_questions", sections)
        self.assertIn("quick_recap", sections)

        # Validate practice questions structure
        pq = sections["practice_questions"]
        self.assertTrue("short_answer_questions" in pq or "short_answer" in pq)
        self.assertTrue("conceptual_questions" in pq or "conceptual" in pq)
        self.assertTrue("multiple_choice_question" in pq or "multiple_choice" in pq)
        mcq = pq.get("multiple_choice_question") or pq.get("multiple_choice")
        self.assertIn("question", mcq)
        self.assertIn("options", mcq)
        self.assertIn("correct_answer", mcq)
        self.assertIn("explanation", mcq)

    def test_05_professional_notes_generation_and_schema(self):
        """Verify Professional mode generates required sections and conforms to schema."""
        service = ClipNotesService()
        notes = service.generate_notes_for_clip(
            video_id=self.video_id,
            clip=self.clip_obj,
            audience_mode="professional",
            user_id=self.user_id
        )
        self.assertEqual(notes["audience_mode"], "professional")
        sections = notes.get("sections", {})
        self.assertIn("executive_summary", sections)
        self.assertIn("technical_breakdown", sections)
        self.assertIn("industry_relevance", sections)
        self.assertIn("implementation_insights", sections)
        self.assertIn("key_takeaways", sections)
        self.assertIn("action_items", sections)
        self.assertIn("terminology", sections)

    def test_06_content_creator_notes_generation_and_schema(self):
        """Verify Content Creator mode generates required sections and conforms to schema."""
        service = ClipNotesService()
        notes = service.generate_notes_for_clip(
            video_id=self.video_id,
            clip=self.clip_obj,
            audience_mode="content_creator",
            user_id=self.user_id
        )
        self.assertEqual(notes["audience_mode"], "content_creator")
        sections = notes.get("sections", {})
        self.assertIn("core_message", sections)
        self.assertIn("suggested_titles", sections)
        self.assertEqual(len(sections["suggested_titles"]), 5)
        self.assertIn("hooks", sections)
        self.assertEqual(len(sections["hooks"]), 2)
        self.assertIn("story_structure", sections)
        self.assertIn("storytelling_improvements", sections)
        self.assertIn("suggested_conclusion", sections)
        self.assertIn("social_caption", sections)
        self.assertIn("call_to_action", sections)

    def test_07_persistence_and_caching_per_mode(self):
        """Verify notes are persisted and cached per (clip_id, audience_mode)."""
        test_clip_id = "clip_cache_test_777"
        # Save education notes
        mock_notes_edu = {
            "clip_id": test_clip_id,
            "audience_mode": "education",
            "clip_title": "Paging and TLB",
            "clip_summary": "Test education summary",
            "key_points": ["Point A"],
            "sections": {"quick_recap": ["Recap 1"]},
            "source_start": 16.0,
            "source_end": 55.0,
            "generation_status": "completed"
        }
        DatabaseService.save_clip_notes(
            clip_id=test_clip_id,
            video_id=self.video_id,
            user_id=self.user_id,
            audience_mode="education",
            notes_json=mock_notes_edu,
            model_metadata={"model": "test-mock"}
        )

        # Retrieve education notes
        cached_edu = DatabaseService.get_clip_notes(test_clip_id, "education")
        self.assertIsNotNone(cached_edu)
        self.assertEqual(cached_edu["clip_summary"], "Test education summary")

        # Professional notes should be None initially for this new clip
        cached_prof = DatabaseService.get_clip_notes(test_clip_id, "professional")
        self.assertIsNone(cached_prof)

        # Save professional notes
        mock_notes_prof = {
            "clip_id": test_clip_id,
            "audience_mode": "professional",
            "clip_title": "Paging and TLB",
            "clip_summary": "Test professional summary",
            "key_points": ["Point B"],
            "sections": {"executive_summary": "Exec summary text"},
            "source_start": 16.0,
            "source_end": 55.0,
            "generation_status": "completed"
        }
        DatabaseService.save_clip_notes(
            clip_id=test_clip_id,
            video_id=self.video_id,
            user_id=self.user_id,
            audience_mode="professional",
            notes_json=mock_notes_prof,
            model_metadata={"model": "test-mock"}
        )

        cached_prof_after = DatabaseService.get_clip_notes(test_clip_id, "professional")
        self.assertIsNotNone(cached_prof_after)
        self.assertEqual(cached_prof_after["clip_summary"], "Test professional summary")

        # Verify education still distinct
        cached_edu_after = DatabaseService.get_clip_notes(test_clip_id, "education")
        self.assertEqual(cached_edu_after["clip_summary"], "Test education summary")

    def test_08_api_notes_endpoints_and_authorization(self):
        """Test API endpoints for notes retrieval, caching, and unauthorized access."""
        # 1. GET cached notes
        res_get = self.client.get(
            f"/api/clips/{self.clip_id}/notes?audience_mode=education",
            headers=self.owner_headers
        )
        self.assertEqual(res_get.status_code, 200)
        data = res_get.json()
        self.assertEqual(data["audience_mode"], "education")
        self.assertIn("clip_summary", data)

        # 2. POST endpoint returns cached or generates
        res_post = self.client.post(
            f"/api/clips/{self.clip_id}/notes",
            json={"audience_mode": "education"},
            headers=self.owner_headers
        )
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["audience_mode"], "education")

        # 3. Nonexistent clip returns 404
        res_nonexistent = self.client.get(
            "/api/clips/nonexistent_clip_999/notes?audience_mode=education",
            headers=self.owner_headers
        )
        self.assertEqual(res_nonexistent.status_code, 404)

        # 4. Unauthorized access prevention:
        # Create a video owned by user_b
        isolated_vid_id = "test_vid_isolated_b"
        isolated_clip_id = "clip_isolated_b"
        DatabaseService.save_video(
            video_id=isolated_vid_id,
            filename="private.mp4",
            filepath="storage/uploads/private.mp4",
            duration=30.0,
            user_id="user_b",
            audience_mode="education"
        )
        DatabaseService.save_clips(isolated_vid_id, [{
            "id": isolated_clip_id,
            "video_id": isolated_vid_id,
            "title": "Private Clip",
            "start_time": 0.0,
            "end_time": 10.0
        }])

        # An authenticated session for user_a trying to access user_b's clip
        from backend.services.auth_service import AuthService
        token_user_a = AuthService.create_fallback_jwt("user_a", {"email": "user_a@test.com"})
        headers_user_a = {"Authorization": f"Bearer {token_user_a}"}
        res_auth_denied = self.client.get(
            f"/api/clips/{isolated_clip_id}/notes?audience_mode=education",
            headers=headers_user_a
        )
        self.assertEqual(res_auth_denied.status_code, 403)
        self.assertIn("Forbidden", res_auth_denied.json()["detail"])

if __name__ == "__main__":
    unittest.main()
