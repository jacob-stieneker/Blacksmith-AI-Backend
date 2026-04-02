from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.audio.mastering import MasteringEngineError, run_mastering_job
from app.audio.types import MasteringSettings
from app.core.config import MEDIA_DIR, get_allowed_origins

app = FastAPI(title="Blacksmith AI Backend")

allowed_origins = get_allowed_origins()
allow_all_origins = allowed_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.Lock()


def ensure_allowed_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use WAV, MP3, or M4A.")
    return ext


def _set_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = time.time()


def _get_job(job_id: str) -> dict[str, Any]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Mastering job not found.")
        return dict(job)


def _public_job_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status", "queued"),
        "stage": job.get("stage", "upload"),
        "progress_percent": job.get("progress_percent", 0),
        "message": job.get("message", ""),
        "source_filename": job.get("source_filename"),
    }

    if job.get("error"):
        payload["error"] = job["error"]

    if job.get("status") == "ready":
        target_lufs = None
        if job.get("recipe"):
            target_lufs = job["recipe"].get("target_lufs")
        payload["preview_url"] = f"/media/{job_id}/{job['preview_filename']}"
        payload["download_url"] = f"/api/master/jobs/{job_id}/download"
        payload["stats"] = {
            "input_lufs": job["input_analysis"].get("input_lufs"),
            "input_true_peak": job["input_analysis"].get("input_true_peak"),
            "input_lra": job["input_analysis"].get("input_lra"),
            "target_lufs": target_lufs,
        }
        payload["analysis"] = {
            "input": job.get("input_analysis", {}),
            "output": job.get("output_analysis", {}),
        }
        payload["recipe"] = job.get("recipe", {})
        payload["settings_received"] = job.get("settings", {})
        payload["loudnorm"] = job.get("loudnorm", {})
    return payload


def _process_job(job_id: str, uploaded_input_path: Path, job_dir: Path, settings: MasteringSettings) -> None:
    def progress_callback(stage: str, percent: int, message: str) -> None:
        _set_job(
            job_id,
            status="processing" if stage != "ready" else "ready",
            stage=stage,
            progress_percent=percent,
            message=message,
        )

    try:
        result = run_mastering_job(
            uploaded_input_path=uploaded_input_path,
            job_dir=job_dir,
            settings=settings,
            progress_callback=progress_callback,
        )
    except MasteringEngineError as exc:
        _set_job(job_id, status="error", stage="render", error=str(exc), message=str(exc))
        return
    except Exception as exc:
        _set_job(job_id, status="error", stage="render", error=f"Mastering failed: {exc}", message=f"Mastering failed: {exc}")
        return

    _set_job(
        job_id,
        status="ready",
        stage="ready",
        progress_percent=100,
        message="Master complete",
        preview_filename=result["preview_filename"],
        download_filename=result["download_filename"],
        input_analysis=result["input_analysis"],
        output_analysis=result["output_analysis"],
        recipe=result["recipe"],
        settings=result["settings"],
        loudnorm=result.get("loudnorm", {}),
    )


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/master/jobs")
async def create_mastering_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    low_eq: float = Form(0.0),
    mid_eq: float = Form(0.0),
    high_eq: float = Form(0.0),
    compression_mode: str = Form("normal"),
    loudness_mode: str = Form("normal"),
    stereo_width_mode: str = Form("normal"),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    ext = ensure_allowed_file(file.filename)
    job_id = str(uuid4())
    job_dir = MEDIA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_input_path = job_dir / f"original_upload{ext}"
    source_filename = file.filename

    settings = MasteringSettings(
        low_eq=low_eq,
        mid_eq=mid_eq,
        high_eq=high_eq,
        compression_mode=compression_mode,
        loudness_mode=loudness_mode,
        stereo_width_mode=stereo_width_mode,
    ).normalized()

    try:
        with uploaded_input_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    finally:
        file.file.close()

    _set_job(
        job_id,
        status="queued",
        stage="upload",
        progress_percent=15,
        message="Upload complete. Waiting for backend analysis",
        source_filename=source_filename,
        created_at=time.time(),
    )

    background_tasks.add_task(_process_job, job_id, uploaded_input_path, job_dir, settings)
    return _public_job_payload(job_id, _get_job(job_id))


@app.get("/api/master/jobs/{job_id}")
async def get_mastering_job(job_id: str) -> dict[str, Any]:
    return _public_job_payload(job_id, _get_job(job_id))


@app.get("/api/master/jobs/{job_id}/download")
async def download_mastered_file(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Mastered file is not ready yet.")

    download_filename = job.get("download_filename")
    if not download_filename:
        raise HTTPException(status_code=404, detail="Mastered file not found.")

    file_path = MEDIA_DIR / job_id / download_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Mastered file not found on disk.")

    source_filename = Path(job.get("source_filename") or "mastered.wav").stem
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"{source_filename}-mastered.wav",
    )
