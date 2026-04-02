# Blacksmith AI Backend

Backend for the Blacksmith AI mastering UI.

This service accepts an uploaded mix, decodes it to a working WAV, analyzes loudness and tonal balance,
runs a rule-based mastering pass, creates a preview MP3 plus a downloadable mastered WAV, and returns the
response shape the current HTML UI expects.

## Repo structure

```text
blacksmith-ai-backend/
  app/
    __init__.py
    main.py
    core/
      __init__.py
      config.py
    audio/
      __init__.py
      analyze.py
      io.py
      mastering.py
      process.py
      types.py
  media/
    .gitkeep
  temp/
    .gitkeep
  requirements.txt
  Dockerfile
  README.md
```

## What the backend does

- `POST /api/master`
  - accepts `file`, `target_lufs`, `warmth`, `brightness`, `punch`, `low_eq`, `mid_eq`, `high_eq`, `compression`
  - saves the upload into a per-job folder under `media/`
  - decodes the input with `ffmpeg`
  - analyzes loudness, peaks, dynamics, and spectral balance
  - runs a mastering chain built from the analysis plus slider settings
  - writes:
    - `mastered.wav`
    - `preview.mp3`
  - returns:
    - `preview_url`
    - `download_url`
    - `stats`
    - `analysis`
    - `recipe`
    - `settings_received`

## Important front-end note

Your current HTML UI still needs two small changes:

1. It should send `low_eq`, `mid_eq`, `high_eq`, and `compression` in `getMasteringSettings()`.
2. If this API is not hosted on the same origin as the page, your HTML needs a real backend base URL instead of assuming `window.location.origin`.

See `FRONTEND_PATCH.md` for the exact snippet.

## Local setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

You need `ffmpeg` available on your machine path unless you use Docker.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

Visit:

- Health check: `http://127.0.0.1:8000/api/health`
- Docs: `http://127.0.0.1:8000/docs`

## Docker

Build and run:

```bash
docker build -t blacksmith-ai-backend .
docker run --rm -p 8000:8000 blacksmith-ai-backend
```

## Environment variables

- `BSAI_ALLOWED_ORIGINS`
  - comma-separated list of allowed browser origins for CORS
- `BSAI_FFMPEG_BINARY`
  - path to the ffmpeg binary
- `BSAI_WORKING_SAMPLE_RATE`
  - default: `48000`
- `BSAI_PREVIEW_BITRATE`
  - default: `192k`
- `BSAI_CLEAN_INTERMEDIATE_FILES`
  - default: `true`

Example:

```bash
export BSAI_ALLOWED_ORIGINS="https://your-site.wixsite.com,https://www.yourdomain.com"
```

## API example

```bash
curl -X POST "http://127.0.0.1:8000/api/master" \
  -F "file=@/path/to/song.wav" \
  -F "target_lufs=-11.5" \
  -F "warmth=50" \
  -F "brightness=55" \
  -F "punch=60" \
  -F "low_eq=0.5" \
  -F "mid_eq=-0.5" \
  -F "high_eq=1.0" \
  -F "compression=2.6"
```

## Notes on the mastering approach

This is a rule-based mastering engine, not an end-to-end neural network. The signal chain is:

- high-pass cleanup
- low shelf
- broad mid contour
- presence contour
- high shelf
- bus compression
- loudness normalization
- final limiting

The recipe is shaped by both user controls and measured features from the input mix.
