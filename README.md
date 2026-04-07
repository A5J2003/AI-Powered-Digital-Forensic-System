# AI-Powered Digital Forensic System

> A multimodal deepfake detection and forensic analysis framework that verifies the authenticity of digital media — videos, audio, and text — using state-of-the-art AI models, explainability tools, and digital forensics principles.


---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Backend Setup](#2-backend-setup)
  - [3. Frontend Setup](#3-frontend-setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Models & Techniques](#models--techniques)
- [Explainability](#explainability)
- [Digital Forensics Capabilities](#digital-forensics-capabilities)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Overview

As AI-generated media becomes increasingly convincing, the need for robust, trustworthy detection systems is critical. This project addresses that challenge with a **comprehensive forensic analysis pipeline** that goes beyond simple detection.

The system analyses media files (video, audio, PDF, DOCX, TXT) for signs of AI-generated content and deepfakes across three modalities — **video, audio, and text** — and produces forensic-grade PDF reports with full explainability, chain-of-custody logging, and GradCAM visualisations. It is designed for use in digital forensics investigations, media verification, and research contexts.

| Modality | Models |
|---|---|
| **Video** | Xception + Swin Transformer (frame-level) |
| **Audio** | CRNN on log-mel spectrograms |
| **Text** | RoBERTa + DeBERTa-v3 ensemble |

---

## Key Features

| Feature | Description |
|---|---|
| Video Deepfake Detection | Frame-level analysis using XceptionNet and Swin Transformer |
| Audio Deepfake Detection | Temporal pattern analysis via CRNN on Log-Mel spectrograms |
| AI Text Detection | RoBERTa & DeBERTa models to detect AI-generated writing |
| Cross-Modal Forensics | ASR + OCR consistency scoring to catch dubbing and tampering |
| Explainability | Grad-CAM heatmaps, attention maps, and human-readable explanations |
| Forensic Integrity | SHA-256 hashing, chain-of-custody logging, metadata analysis |
| Forensic Reports | Automated, evidence-traceable PDF forensic report generation |
| REST API | FastAPI backend with full Swagger UI documentation |
| Modern Frontend | React + TypeScript + Tailwind CSS dashboard for file upload and report viewing |

---

## System Architecture

```
input-canvas  (React + Vite + Tailwind)
      │  HTTP (fetch)
      ▼
project-x  (FastAPI backend)
  └── api.py          ← FastAPI server
  └── main.py         ← Core pipeline entry point
  └── pipeline/       ← video / audio / text sub-pipelines
  └── models/         ← trained model weights (.pth / .pt files)
  └── input_pipeline/ ← file ingestion, OCR, ASR
  └── explainability/ ← GradCAM, Integrated Gradients
  └── report.py       ← PDF report generator
  └── output/         ← per-case results (created at runtime)
```

```
                          Input Source
                               │
                               ▼
                   Data Cleaning and Sorting
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ Video        │    │ Audio        │    │ Textual      │
  │ Detector     │    │ Detector     │    │ Detector     │
  │              │    │              │    │              │
  │ XceptionNet  │    │ CRNN on      │    │ RoBERTa      │
  │ Swin Trans.  │    │ Log-Mel Spec │    │ DeBERTa-v3   │
  │              │    │              │    │              │
  │ Grad-CAM     │    │ Grad-CAM     │    │ Integrated   │
  │ Attention    │    │ Heatmap      │    │ Gradients    │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌──────────────────────────┐
              │   fuse_results()         │
              │   Video 50% + Audio 30%  │
              │   + Text 20%             │
              └──────────────┬───────────┘
                             ▼
              ┌──────────────────────────┐
              │  Forensic PDF Report     │
              │  + SHA-256 Verification  │
              │  + Chain-of-Custody Log  │
              └──────────────────────────┘
```

---

## Project Structure

```
AI-Powered-Digital-Forensic-System/
│
├── project-x/                            ← Python backend (FastAPI)
│   ├── api.py                            ← FastAPI server ← START HERE
│   ├── main.py                           ← Pipeline entry point (CLI / API)
│   ├── report.py                         ← PDF report generator
│   ├── hash_db.json
│   ├── requirements.txt
│   │
│   ├── explainability/
│   │   ├── audio_explainer.py
│   │   ├── text_explainer.py
│   │   └── video_explainer.py
│   │
│   ├── input_pipeline/
│   │   ├── __init__.py
│   │   ├── asr.py
│   │   ├── consistency.py
│   │   ├── ingestion.py
│   │   ├── metadata.py
│   │   ├── ocr.py
│   │   ├── subtitle_utils.py
│   │   ├── video_processing.py
│   │   └── text_strategy.py
│   │
│   ├── models/
│   │   ├── audio/
│   │   │   └── CRNN_aug_best_model.pth   ← required
│   │   ├── video/
│   │   │   ├── xception/
│   │   │   │   └── xceptionnet_best.pth  ← required
│   │   │   └── swin/
│   │   │       └── swin_best.pth         ← required
│   │   └── text/
│   │       ├── config.json               ← required
│   │       ├── roberta_best_model.pt     ← required
│   │       └── deberta_best_model.pt     ← required
│   │
│   ├── pipeline/
│   │   ├── audio_pipeline.py
│   │   ├── text_pipeline.py
│   │   └── video_pipeline.py
│   │
│   ├── utils/
│   │   └── case_manager.py
│   │
│   └── output/                           ← created at runtime
│       └── CASE_<timestamp>/
│           ├── input/
│           ├── extracted/
│           ├── results/
│           ├── explainability/
│           ├── logs/
│           ├── metadata/
│           └── report/
│
└── input-canvas/                         ← React frontend (Vite + TypeScript)
    ├── index.html
    ├── .env                              ← VITE_API_URL=http://localhost:8000
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── components.json
    │
    ├── public/
    ├── supabase/
    └── src/
        ├── lib/
        │   └── api.ts                    ← typed API client
        ├── pages/
        │   ├── Upload.tsx
        │   ├── MyFiles.tsx
        │   └── Reports.tsx
        └── components/
            └── dashboard/
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 (recommended) |
| Node.js | 18+ |
| npm | 9+ |
| ffmpeg | Required by Whisper ASR — [download](https://ffmpeg.org/download.html) |
| CUDA (optional) | 11.8 or 12.1 for GPU acceleration |

> **Windows note:** After installing ffmpeg, add it to your PATH. You can verify it works by running `ffmpeg -version` in your terminal.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/A5J2003/AI-Powered-Digital-Forensic-System.git
cd AI-Powered-Digital-Forensic-System
```

---

### 2. Backend Setup

```bash
cd project-x
```

**Create and activate a virtual environment:**

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# Mac / Linux
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

> **GPU users:** Install a CUDA-compatible PyTorch wheel **before** running the above command.
> Visit https://pytorch.org/get-started and copy the matching pip command, for example:
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
> ```

**Place model files:**

The pipeline requires trained model weights. Place them at the exact paths below (relative to the `project-x/` root):

```
models/
├── audio/
│   └── CRNN_aug_best_model.pth
├── video/
│   ├── xception/
│   │   └── xceptionnet_best.pth
│   └── swin/
│       └── swin_best.pth
└── text/
    ├── config.json
    ├── roberta_best_model.pt
    └── deberta_best_model.pt
```

**Example `models/text/config.json`:**

```json
{
  "roberta_model_path": "models/text/roberta_best_model.pt",
  "deberta_model_path": "models/text/deberta_best_model.pt",
  "ensemble_weights": {
    "roberta": 0.5,
    "deberta": 0.5
  },
  "max_length": 512
}
```

---

### 3. Frontend Setup

Open a **new terminal** and navigate to the frontend folder:

```bash
cd input-canvas
npm install
```

**Check the `.env` file** — it should already contain:

```
VITE_API_URL=http://localhost:8000
```

If you're running the backend on a different port, update this value accordingly.

---

## Running the Project

You need **two terminals running simultaneously** — one for the backend and one for the frontend.

**Terminal 1 — Start the backend** (with venv activated, inside `project-x/`):

```bash
uvicorn api:app --reload --port 8000
```

The API server starts at **http://localhost:8000**.
Visit **http://localhost:8000/docs** for the interactive Swagger UI.

> **Note:** The first request may take a few minutes as Whisper, EasyOCR, and the deep learning models are loaded lazily on first use.

**Terminal 2 — Start the frontend** (inside `input-canvas/`):

```bash
npm run dev
```

Open **http://localhost:5173** in your browser to access the dashboard.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Heartbeat — returns `{ status: "ok" }` |
| `POST` | `/analyze` | Upload a file (multipart/form-data, field: `file`). Returns `{ job_id, filename, status }` |
| `GET` | `/status/{job_id}` | Poll status: `pending` → `running` → `done` or `error` |
| `GET` | `/result/{job_id}` | Full JSON result (once status is `done`) |
| `GET` | `/report/{job_id}` | Download the PDF forensic report |
| `GET` | `/jobs` | List all jobs |

**Example — upload a file:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/video.mp4"
```

Response:
```json
{ "job_id": "abc123...", "filename": "video.mp4", "status": "pending" }
```

**Example — poll until done:**

```bash
curl http://localhost:8000/status/abc123...
# { "job_id": "...", "status": "running" }

curl http://localhost:8000/result/abc123...
# Full result JSON
```

---

## How the Pipeline Works

```
Upload file
    │
    ▼
detect_file_type()
    │
    ├── video → extract frames, audio, OCR → run_video_pipeline + run_audio_pipeline + run_text_pipeline
    ├── audio → run_audio_pipeline
    └── text  → extract_text → run_text_pipeline
    │
    ▼
fuse_results()  — weighted combination (Video 50%, Audio 30%, Text 20%)
    │
    ▼
generate_report()  — PDF with GradCAM images, confidence gauges, chain-of-custody log
    │
    ▼
output/<case_id>/
    ├── input/          original file copy
    ├── extracted/      audio, frames, text
    ├── results/        JSON outputs
    ├── explainability/ GradCAM / IG visualisations
    ├── logs/           chain_of_custody.json
    ├── metadata/       metadata.json
    └── report/         forensic_report_<case_id>.pdf
```

---

## Models & Techniques

### Video Deepfake Detection
- **XceptionNet** — CNN-based spatial analysis of individual frames; excels at detecting GAN artifacts and pixel-level manipulations
- **Swin Transformer** — Attention-based model that captures long-range spatial dependencies across frame patches
- Frame-level anomaly scoring with temporal aggregation

### Audio Deepfake Detection
- **CRNN (CNN + RNN hybrid)** — Processes Log-Mel spectrogram representations of audio
- Captures both local frequency patterns (CNN) and temporal sequences (RNN)
- Temporal pattern analysis across the full audio timeline

### Text AI Detection
- **RoBERTa** and **DeBERTa-v3** transformers fine-tuned for AI-generated text classification
- Detects statistical and stylistic patterns characteristic of LLM outputs
- Outputs a confidence-based classification with token-level attribution via Captum

### Cross-Modal Consistency
- **ASR (Automatic Speech Recognition)** — Transcribes the audio track via Whisper
- **OCR (Optical Character Recognition)** — Extracts on-screen text from video frames via EasyOCR
- **Consistency scoring** — Flags mismatches between spoken content and visual text, indicating potential dubbing or tampering

---

## Explainability

Every detection decision is backed by visual and textual evidence:

- **Video Explainer** (`video_explainer.py`) — Grad-CAM heatmaps highlighting frame regions that influenced the model's prediction, plus Swin Transformer attention maps
- **Audio Explainer** (`audio_explainer.py`) — Highlights frequency-time regions in spectrograms that triggered detection
- **Text Explainer** (`text_explainer.py`) — Token-level attribution showing which words drove the AI-text classification via Captum
- **Human-readable Explanations** — Natural language summaries of findings included in the forensic PDF report

---

## Digital Forensics Capabilities

| Capability | Details |
|---|---|
| **File Integrity** | SHA-256 hashing of all input files; hashes stored in `hash_db.json` |
| **Case Management** | Evidence organisation and case tracking via `case_manager.py` |
| **Metadata Extraction** | EXIF data, codec info, creation timestamps extracted via `metadata.py` |
| **Cross-Modal Tamper Detection** | ASR-OCR consistency check (`consistency.py`) flags dubbing and splicing |
| **Evidence Traceability** | All outputs linked back to the original file hash and case record |
| **Chain-of-Custody Logging** | Full audit trail saved to `logs/chain_of_custody.json` per case |

---

## Dependencies

| Category | Libraries |
|---|---|
| Deep Learning | `tensorflow==2.16.1`, `torch==2.2.2`, `torchvision`, `torchaudio` |
| Transformers | `transformers`, `captum`, `sentencepiece`, `accelerate` |
| Video | `opencv-python`, `mtcnn`, `timm`, `pytorch-grad-cam` |
| Audio | `librosa==0.10.2`, `soundfile`, `scipy`, `whisper` |
| OCR / ASR | `easyocr`, `openai-whisper` |
| Document Parsing | `PyPDF2`, `python-docx`, `Pillow` |
| Backend | `fastapi`, `uvicorn` |
| Frontend | `react`, `typescript`, `vite`, `tailwindcss`, `shadcn-ui` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `numpy`, `pandas`, `scikit-learn`, `tqdm` |

Install all backend dependencies with:
```bash
pip install -r requirements.txt
```

Install all frontend dependencies with:
```bash
npm install
```

---

## Troubleshooting

### `ModuleNotFoundError` on startup
Make sure your venv is activated and all packages are installed:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### `ffmpeg not found` error (Whisper)
Install ffmpeg and add it to your PATH:
- Windows: https://www.gyan.dev/ffmpeg/builds/ → download `ffmpeg-release-essentials.zip`
- Unzip and copy `bin/ffmpeg.exe` to a folder in your PATH (e.g. `C:\Windows\System32\`)

### CORS errors in the browser
Check that `VITE_API_URL` in `input-canvas/.env` matches the port uvicorn is running on. The FastAPI server allows `localhost:5173` by default.

### Model file not found
Verify your `models/text/config.json` has correct relative paths (relative to the `project-x/` root where you run `uvicorn`).

### GPU / CUDA issues
PyTorch will fall back to CPU automatically. If you see CUDA errors, reinstall the CPU version:
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

### Port already in use
Change the uvicorn port:
```bash
uvicorn api:app --reload --port 8001
```
Then update `VITE_API_URL=http://localhost:8001` in `input-canvas/.env`.

### Frontend not connecting to backend
Make sure both servers are running at the same time in separate terminals. The frontend at `localhost:5173` needs the backend at `localhost:8000` to be active.

---

<p align="center">Built for truth in a world of synthetic media.</p>
