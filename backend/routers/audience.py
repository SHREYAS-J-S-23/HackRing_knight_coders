from fastapi import APIRouter
from typing import List
from backend.models.schemas import AudienceProfile

router = APIRouter(tags=["Audiences"])

DEFAULT_PROFILES: List[AudienceProfile] = [
    AudienceProfile(
        id="exec_brief",
        name="Executive / Investor Brief",
        target_role="Decision Makers & Investors",
        description="Extracts bottom-line conclusions, architectural decisions, and metrics. Eliminates pleasantries and troubleshooting.",
        compression_target="70% - 80% reduction (Under 3-5 mins)",
        priority_topics=["System Architecture", "Business Value", "Guarantees"],
        deprioritize=["Microphone banter", "Personal anecdotes", "Detailed line-by-line code"],
        tone="authoritative and concise"
    ),
    AudienceProfile(
        id="students_edu",
        name="Students & Developers Tutorial",
        target_role="Students, Engineers, Learners",
        description="Preserves full step-by-step technical concepts, caveats, and logic flow while eliminating dead air and false starts.",
        compression_target="30% - 40% reduction (Paced for learning)",
        priority_topics=["Concept explanations", "Why things work", "Guarantees & Caveats"],
        deprioritize=["Equipment check", "Unrelated personal digressions"],
        tone="educational, clear, methodical"
    ),
    AudienceProfile(
        id="social_viral",
        name="Social Media / Short-Form Highlights",
        target_role="General tech audience on LinkedIn/X",
        description="Highlights high-energy, memorable insights, big-picture problem statements, and key revelations.",
        compression_target="85% reduction (Sub-90 seconds)",
        priority_topics=["Problem statements", "Striking quotes", "Final takeaway"],
        deprioritize=["Nuanced qualifications", "Preamble", "Pauses"],
        tone="engaging, snappy, provocative"
    ),
    AudienceProfile(
        id="community_multilingual",
        name="Community & General Public",
        target_role="Community members, non-technical stakeholders",
        description="Simplifies jargon, preserves cultural nuances and storytelling while removing technical jargon overload.",
        compression_target="50% reduction",
        priority_topics=["Community impact", "Core mission", "Actionable next steps"],
        deprioritize=["Technical architecture details", "Banter"],
        tone="accessible, warm, inclusive"
    )
]

@router.get("/api/audiences", response_model=List[AudienceProfile])
@router.get("/api/audiences/profiles", response_model=List[AudienceProfile])
@router.get("/api/audience", response_model=List[AudienceProfile])
@router.get("/api/audience/profiles", response_model=List[AudienceProfile])
async def list_audience_profiles():
    """Returns pre-configured audience profiles."""
    return DEFAULT_PROFILES
