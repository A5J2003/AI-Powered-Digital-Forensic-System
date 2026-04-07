"""
api.py — FastAPI server for the AI-Powered Digital Forensic System
===========================================================================
Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health            — heartbeat
    POST /analyze           — upload file → returns { job_id } immediately
    GET  /status/{job_id}   — poll: pending | running | done | error
    GET  /result/{job_id}   — full JSON result once done
    GET  /report/{job_id}   — stream PDF report as file download
    GET  /jobs              — list all past jobs (newest first)
===========================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Forensic Analysis API",
    description="Multimodal deepfake detection: video · audio · text",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store { job_id: { status, case_path, result, error } }
# In production, replace with Redis / SQLite.
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Thread pool — limits concurrent heavy inference to 1 by default so the
# GPU / CPU is not overloaded. Increase max_workers if you have GPUs.
_executor = ThreadPoolExecutor(max_workers=1)

OUTPUT_DIR = Path("output")
UPLOAD_TMP = Path("upload_tmp")
UPLOAD_TMP.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(kwargs)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def _scan_output_jobs() -> list[Dict[str, Any]]:
    """Discover cases that were run via CLI (not this API session)."""
    found = []
    if not OUTPUT_DIR.exists():
        return found
    for case_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not case_dir.is_dir():
            continue
        result_file = case_dir / "results" / "final_result.json"
        if result_file.exists():
            found.append({"job_id": case_dir.name, "case_path": str(case_dir)})
    return found


def _run_pipeline_job(job_id: str, input_path: str, original_filename: str):
    """Worker function executed in the thread pool."""
    _set_job(job_id, status="running")
    try:
        # Import here so that startup is fast and model loading is lazy
        from main import run_pipeline  # noqa: PLC0415

        result = run_pipeline(input_path, original_filename=original_filename)

        # FIX: run_pipeline now returns _case_path directly in the result dict
        case_path = None
        if isinstance(result, dict):
            case_path = result.pop("_case_path", None)

        # Fallback: infer case_path from hash_db if pipeline didn't return it
        if not case_path:
            hash_db_path = Path("hash_db.json")
            if hash_db_path.exists():
                with open(hash_db_path) as f:
                    db: dict = json.load(f)
                if db:
                    case_path = list(db.values())[-1]

        _set_job(
            job_id,
            status="done",
            result=result,
            case_path=case_path,
        )

    except Exception as exc:
        _set_job(
            job_id,
            status="error",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        # Clean up temporary upload file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Simple heartbeat — returns 200 OK when the server is up."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", tags=["Analysis"])
async def analyze(file: UploadFile = File(...)):
    """
    Upload a file (video / audio / text / PDF / DOCX) and start analysis.
    Returns a `job_id` immediately. Poll `/status/{job_id}` to track progress.
    """
    job_id = str(uuid.uuid4())

    # Save the upload to a temporary location
    suffix = Path(file.filename or "upload").suffix
    tmp_path = UPLOAD_TMP / f"{job_id}{suffix}"

    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    original_filename = file.filename or f"{job_id}{suffix}"

    _set_job(
        job_id,
        status="pending",
        filename=original_filename,
        content_type=file.content_type,
        case_path=None,
        result=None,
        error=None,
    )

    # Kick off analysis in the background thread pool
    # FIX: pass original_filename so the pipeline can save it into the result JSON
    _executor.submit(_run_pipeline_job, job_id, str(tmp_path), original_filename)

    return {"job_id": job_id, "filename": original_filename, "status": "pending"}


@app.get("/status/{job_id}", tags=["Analysis"])
def get_status(job_id: str):
    """
    Poll the status of an analysis job.

    Returns:
        status: "pending" | "running" | "done" | "error"
        error:  error message (only when status == "error")
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "filename": job.get("filename"),
    }
    if job.get("status") == "error":
        response["error"] = job.get("error")
    return response


@app.get("/result/{job_id}", tags=["Analysis"])
def get_result(job_id: str):
    """
    Return the full analysis result for a completed job.

    The result includes:
      - `final`      — fusion verdict: label (0=REAL,1=FAKE), confidence, explanation
      - `modalities` — per-modality results (video / audio / text)
      - `file_hash`  — SHA-256 of the original file
    """
    job = _get_job(job_id)

    # Also check disk (jobs run via CLI won't be in memory)
    if not job:
        case_dir = OUTPUT_DIR / job_id
        if case_dir.exists():
            result_file = case_dir / "results" / "final_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    return json.load(f)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    status = job.get("status")
    if status == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Pipeline error"))
    if status in ("pending", "running"):
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {status}. Try again shortly.",
        )

    # Try in-memory result first
    if job.get("result"):
        return job["result"]

    # Fall back to disk
    case_path = job.get("case_path")
    if case_path:
        result_file = Path(case_path) / "results" / "final_result.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)

    raise HTTPException(status_code=404, detail="Result file not found on disk")


@app.get("/report/{job_id}", tags=["Reports"])
def get_report(job_id: str):
    """
    Download the PDF forensic report for a completed job.
    """
    job = _get_job(job_id)
    case_path: Optional[str] = None

    if job:
        case_path = job.get("case_path")
    else:
        # Check disk for CLI-run cases
        candidate = OUTPUT_DIR / job_id
        if candidate.exists():
            case_path = str(candidate)

    if not case_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Locate the report PDF
    report_dir = Path(case_path) / "report"
    pdf_files = list(report_dir.glob("*.pdf")) if report_dir.exists() else []

    if not pdf_files:
        raise HTTPException(
            status_code=404,
            detail="PDF report not yet generated for this job",
        )

    pdf_path = pdf_files[0]
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@app.get("/jobs", tags=["Analysis"])
def list_jobs():
    """
    List all analysis jobs — both from this API session and any CLI-run cases
    found in the `output/` directory.

    Returns a list sorted newest-first.
    """
    result = []

    # In-memory jobs (this session)
    with _jobs_lock:
        for job_id, meta in _jobs.items():
            entry = {
                "job_id": job_id,
                "status": meta.get("status", "unknown"),
                "filename": meta.get("filename"),
                "case_path": meta.get("case_path"),
            }
            # FIX: include final result data so the frontend card shows
            # the correct confidence and verdict (not just the report page)
            if meta.get("status") == "done" and meta.get("result"):
                entry["final"] = meta["result"].get("final", {})
            result.append(entry)

    # Disk jobs (CLI runs or previous sessions)
    # FIX: use case_path-based dedup so UI-uploaded jobs (whose job_id is a
    # UUID, different from the case folder name) are not shown twice
    in_memory_case_paths = {
        j["case_path"] for j in result if j.get("case_path")
    }
    in_memory_ids = {j["job_id"] for j in result}

    for disk_job in _scan_output_jobs():
        disk_case_path = disk_job["case_path"]

        # Skip if this case folder is already represented by an in-memory job
        if disk_case_path in in_memory_case_paths:
            continue
        if disk_job["job_id"] in in_memory_ids:
            continue

        case_path = disk_case_path
        result_file = Path(case_path) / "results" / "final_result.json"
        final = {}
        filename = None

        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            final = data.get("final", {})
            # FIX: recover original filename saved by the pipeline
            filename = data.get("original_filename")
            # Fallback for old cases that don't have original_filename
            if not filename:
                ingestion = data.get("ingestion", {})
                filename = (
                    ingestion.get("metadata", {})
                    .get("General", {})
                    .get("CompleteName")  # MediaInfo full path — last resort
                )
                if filename:
                    filename = os.path.basename(filename)

        result.append({
            "job_id": disk_job["job_id"],
            "status": "done",
            "filename": filename,
            "case_path": case_path,
            "final": final,
        })

    return result


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)