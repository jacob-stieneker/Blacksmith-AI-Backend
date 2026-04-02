from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.audio.mastering import MasteringEngineError, run_mastering_job
from app.audio.types import MasteringSettings
from app.core.config import MEDIA_DIR, TEMP_DIR, get_allowed_origins

app = FastAPI(title="Blacksmith AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def ensure_allowed_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use WAV, MP3, or M4A.")
    return ext


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/master")
async def master_audio(
    file: UploadFile = File(...),
    target_lufs: float = Form(-14.0),
    warmth: int = Form(35),
    brightness: int = Form(35),
    punch: int = Form(45),
    low_eq: float = Form(0.0),
    mid_eq: float = Form(0.0),
    high_eq: float = Form(0.0),
    compression: float = Form(2.0),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    ext = ensure_allowed_file(file.filename)

    job_id = str(uuid4())
    job_dir = MEDIA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_input_path = job_dir / f"original_upload{ext}"

    settings = MasteringSettings(
        target_lufs=target_lufs,
        warmth=warmth,
        brightness=brightness,
        punch=punch,
        low_eq=low_eq,
        mid_eq=mid_eq,
        high_eq=high_eq,
        compression=compression,
    ).normalized()

    try:
        with uploaded_input_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    finally:
        file.file.close()

    try:
        result = run_mastering_job(uploaded_input_path, job_dir, settings)
    except MasteringEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mastering failed: {exc}") from exc

    return {
        "preview_url": f"/media/{job_id}/{result['preview_filename']}",
        "download_url": f"/media/{job_id}/{result['download_filename']}",
        "stats": {
            "input_lufs": result["input_analysis"].get("input_lufs"),
            "input_true_peak": result["input_analysis"].get("input_true_peak"),
            "input_lra": result["input_analysis"].get("input_lra"),
            "target_lufs": settings.target_lufs,
        },
        "analysis": {
            "input": result["input_analysis"],
            "output": result["output_analysis"],
        },
        "recipe": result["recipe"],
        "settings_received": result["settings"],
    }
