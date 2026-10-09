import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import UPLOAD_DIR, OUTPUT_DIR, HOST, PORT
from backend.routers import pipeline, audience, videos, auth, library

app = FastAPI(
    title="Vidara — AI Video Intelligence Platform",
    description="Understand Every Moment. AI-powered long-form video intelligence with autonomous topic discovery, voice/text semantic retrieval, and verified deterministic clip assembly.",
    version="2.0.0"
)

# Enable CORS for local Next.js / React dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers FIRST
app.include_router(auth.router, prefix="/api")
app.include_router(library.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(videos.router)  # Direct access: /videos/...
app.include_router(pipeline.router)  # Preserved backwards compatibility
app.include_router(audience.router)  # Preserved backwards compatibility

# Mount media preview directories
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Mount frontend web application at root (AFTER all API routes)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
