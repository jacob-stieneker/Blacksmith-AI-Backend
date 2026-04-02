# Blacksmith AI Backend

Updated backend for the Blacksmith AI mastering UI.

This version adds:
- asynchronous mastering jobs with backend-driven progress
- polling-friendly job status endpoints
- direct download endpoint for mastered WAV files
- revised mastering controls: low EQ, mid EQ, high EQ, compression, saturation, and target LUFS
- safer mastering defaults with gentler EQ, lighter bus compression, parallel saturation, and two-pass FFmpeg loudness normalization

## API routes

- `GET /api/health`
- `POST /api/master/jobs`
- `GET /api/master/jobs/{job_id}`
- `GET /api/master/jobs/{job_id}/download`

## Front-end expectations

The matching front-end should:
- upload audio to `POST /api/master/jobs`
- poll `GET /api/master/jobs/{job_id}` for real progress
- use `preview_url` for playback
- use `download_url` for blob download

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
