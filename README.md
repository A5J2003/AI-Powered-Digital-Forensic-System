# AI-Powered Digital Forensic System

> A multimodal deepfake detection and forensic analysis framework that verifies the authenticity of digital media — videos, audio, and text — using state-of-the-art AI models, explainability tools, and digital forensics principles.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Techniques](#models--techniques)
- [Explainability](#explainability)
- [Digital Forensics Capabilities](#digital-forensics-capabilities)
- [Dependencies](#dependencies)

---

## Overview

As AI-generated media becomes increasingly convincing, the need for robust, trustworthy detection systems is critical. This project addresses that challenge with a **comprehensive forensic analysis pipeline** that goes beyond simple detection.

The system analyzes multimedia inputs across three modalities — **video, audio, and text** — and produces forensic-grade reports with full explainability. It is designed for use in digital forensics investigations, media verification, and research contexts.

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
| Forensic Reports | Automated, evidence-traceable forensic report generation |

---

## System Architecture

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
  │ Preprocessing│    │ Preprocessing│    │ Preprocessing│
  │ - Frame      │    │ - Resampling │    │ - Tokeniz-   │
  │   Extraction │    │ - Log-Mel    │    │   ation      │
  │ - Face Det.  │    │   Spectrogram│    │ - Stopword   │
  │   (MTCNN)    │    │              │    │   Removal    │
  │ - Normaliz-  │    │              │    │ - Lemmatiz-  │
  │   ation      │    │              │    │   ation      │
  │              │    │              │    │              │
  │ Models       │    │ Models       │    │ Models       │
  │ - XceptionNet│    │ - CRNN on    │    │ - RoBERTa    │
  │ - Swin       │    │   Spectrogram│    │ - DeBERTa    │
  │   Transformer│    │              │    │              │
  │              │    │              │    │              │
  │ Explainabil. │    │ Explainabil. │    │ Explainabil. │
  │ - Grad-CAM   │    │ - Grad-CAM   │    │ - Integrated │
  │   Visualiz.  │    │   Heatmap    │    │   Gradients  │
  │ - Attention  │    │ - Confidence │    │ - Confidence │
  │   Rollout    │    │   Score      │    │   Score      │
  │ - Confidence │    │              │    │              │
  │   Score      │    │              │    │              │
  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌──────────────────────────┐
              │ Explainability Report    │
              │ Generation               │
              │ + SHA-256 Verification   │
              └──────────────────────────┘
```

---

## Project Structure

```
AI-Powered-Digital-Forensic-System/
│
├── main.py
├── report.py
├── hash_db.json
├── requirements.txt
│
├── explainability/
│   ├── audio_explainer.py
│   ├── text_explainer.py
│   └── video_explainer.py
│
├── input_pipeline/
│   ├── __init__.py
│   ├── asr.py
│   ├── consistency.py
│   ├── ingestion.py
│   ├── metadata.py
│   ├── ocr.py
│   ├── subtitle_utils.py
│   ├── video_processing.py
│   ├── test_asr.py
│   ├── test_ocr.py
│   ├── test_pipeline.py
│   └── text_strategy.py
│
├── models/
│   ├── audio/
│   │   ├── CRNN_aug_best_model.pth
│   │   └── crnn_model.py
│   ├── text/
│   │   ├── config_loader.py
│   │   ├── config.json
│   │   ├── debert_best_model.pt
│   │   └── robert_best_model.pt
│   └── video/
│       ├── swin/
│       │   ├── inference.py
│       │   └── swin_best.pth
│       └── xception/
│           ├── inference.py
│           └── xceptionnet_best.pth
│
├── output/
│   └── CASE_<timestamp>/
│       ├── explainability/
│       │   ├── _charts/
│       │   │   ├── audio_segment_scores.png
│       │   │   ├── gauge.png
│       │   │   ├── modality_probs.png
│       │   │   └── video_frame_scores.png
│       │   ├── audio_explanation.json
│       │   ├── audio_gradcam.png
│       │   ├── audio_logmel.png
│       │   ├── text_attribution.png
│       │   ├── text_explanation.json
│       │   ├── video_gradcam_grid.png
│       │   └── video_swin_attention.png
│       ├── extracted/
│       │   ├── audio/
│       │   └── frames/
│       ├── input/
│       ├── logs/
│       │   └── chain_of_custody.json
│       ├── metadata/
│       │   └── metadata.json
│       ├── report/
│       │   └── report.pdf
│       └── results/
│           ├── audio_result.json
│           ├── final_result.json
│           ├── ingestion_result.json
│           ├── text_result.json
│           └── video_result.json
│
├── pipeline/
│   ├── audio_pipeline.py
│   ├── text_pipeline.py
│   └── video_pipeline.py
│
└── utils/
    └── case_manager.py
```

---

## Installation

### Prerequisites

- Python 3.10+
- `ffmpeg` installed system-wide (required for audio/video processing)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/A5J2003/AI-Powered-Digital-Forensic-System.git
cd AI-Powered-Digital-Forensic-System

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** Installing both TensorFlow and PyTorch together can be heavy. If you only need one detection modality, comment out the unused framework in `requirements.txt` before installing.

---

## Usage

Run the system by executing `main.py`:

```bash
python main.py
```

Results and forensic reports will be saved to the `output/` directory.

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
- **RoBERTa** and **DeBERTa** transformers fine-tuned for AI-generated text classification
- Detects statistical and stylistic patterns characteristic of LLM outputs
- Outputs a confidence-based classification with token-level attribution (via Captum)

### Cross-Modal Consistency
- **ASR (Automatic Speech Recognition)** — Transcribes audio track
- **OCR (Optical Character Recognition)** — Extracts on-screen text from video frames
- **Consistency scoring** — Flags mismatches between spoken content and visual text, indicating potential dubbing or content tampering

---

## Explainability

Every detection decision is backed by visual and textual evidence:

- **Video Explainer** (`video_explainer.py`) — Grad-CAM heatmaps highlighting frame regions that influenced the model's prediction, plus Swin Transformer attention maps
- **Audio Explainer** (`audio_explainer.py`) — Highlights frequency-time regions in spectrograms that triggered detection
- **Text Explainer** (`text_explainer.py`) — Token-level attribution showing which words drove the AI-text classification (via Captum)
- **Human-readable Explanations** — Natural language summaries of findings, suitable for inclusion in forensic reports

---

## Digital Forensics Capabilities

This system is built with forensic rigor, not just model accuracy:

| Capability | Details |
|---|---|
| **File Integrity** | SHA-256 hashing of all input files; hashes stored in `hash_db.json` |
| **Case Management** | Evidence organisation and case tracking via `case_manager.py` |
| **Metadata Extraction** | EXIF data, codec info, creation timestamps extracted via `metadata.py` |
| **Cross-Modal Tamper Detection** | ASR-OCR consistency check (`consistency.py`) flags dubbing and splicing |
| **Evidence Traceability** | All outputs are linked back to the original file hash and case record |

---

## Dependencies

Key libraries used in this project:

| Category | Libraries |
|---|---|
| Deep Learning | `tensorflow==2.16.1`, `torch==2.2.2`, `torchvision`, `torchaudio` |
| Transformers | `transformers`, `captum`, `sentencepiece`, `accelerate` |
| Video | `opencv-python`, `mtcnn`, `timm`, `pytorch-grad-cam` |
| Audio | `librosa==0.10.2`, `soundfile`, `scipy` |
| Document Parsing | `PyPDF2`, `python-docx`, `textract`, `Pillow` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `numpy`, `pandas`, `scikit-learn`, `tqdm` |

Install all with:
```bash
pip install -r requirements.txt
```

---



---

<p align="center">Built for truth in a world of synthetic media.</p>
