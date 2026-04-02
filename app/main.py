from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from uuid import uuid4
import shutil

APP_ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = APP_ROOT / "media"
TEMP_DIR = APP_ROOT / "temp"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Blacksmith AI Backend")

# Update these later to your real Wix and custom domain URLs.
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://your-site.wixsite.com",
    "https://www.yourdomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
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
async def health_check():
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

    original_path = job_dir / f"original{ext}"
    preview_path = job_dir / f"preview{ext}"
    download_path = job_dir / f"mastered{ext}"

    with original_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    # Placeholder behavior for now:
    # copy the uploaded file as both preview and mastered output
    shutil.copy2(original_path, preview_path)
    shutil.copy2(original_path, download_path)

    return {
        "preview_url": f"/media/{job_id}/{preview_path.name}",
        "download_url": f"/media/{job_id}/{download_path.name}",
        "stats": {
            "input_lufs": None,
            "input_true_peak": None,
            "input_lra": None,
            "target_lufs": target_lufs,
        },
        "settings_received": {
            "target_lufs": target_lufs,
            "warmth": warmth,
            "brightness": brightness,
            "punch": punch,
            "low_eq": low_eq,
            "mid_eq": mid_eq,
            "high_eq": high_eq,
            "compression": compression,
        },
    }
