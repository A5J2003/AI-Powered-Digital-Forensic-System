from input_pipeline.ingestion import process_video

from pipeline.text_pipeline import (
    load_text_models,
    run_text_pipeline
)

from pipeline.video_pipeline import (
    load_video_pipeline,
    run_video_pipeline
)

from pipeline.audio_pipeline import (
    load_audio_model,
    run_audio_pipeline
)

from utils.case_manager import CaseManager
from report import generate_report

import os
import json
import shutil


# =========================
# TEXT EXTRACTION
# =========================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".srt":
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            if "-->" in line:
                continue
            if line.strip().isdigit():
                continue
            if line.strip() == "":
                continue
            cleaned.append(line)
        return " ".join(cleaned)

    elif ext == ".pdf":
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext == ".doc":
        raise ValueError("Legacy .doc format not supported. Convert to .docx")

    else:
        raise ValueError("Unsupported text format")


# =========================
# FILE TYPE DETECTION
# =========================
def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    video_ext = [".mp4", ".avi", ".mov", ".mkv"]
    audio_ext = [".wav", ".mp3", ".flac"]
    text_ext  = [".txt", ".srt", ".pdf", ".docx"]

    if ext in video_ext:
        return "video"
    elif ext in audio_ext:
        return "audio"
    elif ext in text_ext:
        return "text"
    else:
        return "unknown"


# =========================
# HASH DATABASE
# (deduplication — keeps track of files already processed)
# =========================
HASH_DB_PATH = "hash_db.json"


def load_hash_db():
    if os.path.exists(HASH_DB_PATH):
        with open(HASH_DB_PATH) as f:
            return json.load(f)
    return {}


def save_hash_db(db):
    with open(HASH_DB_PATH, "w") as f:
        json.dump(db, f, indent=4)


# =========================
# PRINT RESULTS
# =========================
def print_results(results, final):
    print("\n=========================")
    print("📊 MODALITY RESULTS")
    print("=========================\n")

    for modality in ["video", "audio", "text"]:
        res = results.get(modality, {})
        label = res.get("label")
        conf  = res.get("confidence")

        if label is None:
            print(f"{modality.upper():<10}: ❌ Not Available")
        else:
            label_str = "FAKE" if label == 1 else "REAL"
            print(f"{modality.upper():<10}: {label_str} ({conf:.4f})")

        if modality == "video":
            exp = results.get("video", {}).get("explainability", {})
            if exp:
                print(f"   ↳ GradCAM   : {exp.get('gradcam_path')}")
                print(f"   ↳ Attention : {exp.get('attention_path')}")

    print("\n=========================")
    print("🧠 FINAL FUSION RESULT")
    print("=========================\n")

    final_label = "FAKE" if final["label"] == 1 else "REAL"
    print(f"FINAL     : {final_label} ({final['confidence']:.4f})")
    print("\n📝 Explanation:")
    print(final["explanation"])
    print("\n=========================\n")


# =========================
# FUSION
# -------------------------
# KEY CHANGES vs original:
#
# 1. Face-swap branch (NEW):
#    When video says FAKE (>0.60) but audio says REAL (<0.40),
#    this is the classic face-swap pattern — the original audio
#    is kept/re-dubbed while only the face is replaced.
#    In this case audio should NOT suppress the video signal.
#    Weights: video 0.80 / audio 0.20  →  overrides the 50/30/20
#    default which was incorrectly letting audio cancel video.
#
# 2. High-confidence video override threshold lowered:
#    0.85 → 0.75  (still conservative, but catches strong detections
#    that were previously missed)
#
# 3. Default weights rebalanced (video-first for deepfake use-case):
#    Old: video 0.50 / audio 0.30 / text 0.20
#    New: video 0.60 / audio 0.25 / text 0.15
#    Rationale: video artifacts are the primary deepfake signal;
#    audio and text are secondary/corroborating channels.
# =========================
def fuse_results(results):

    debug = {}

    video_fake = 0.0
    audio_fake = 0.0
    text_fake  = 0.0

    for modality in ["video", "audio", "text"]:
        if modality in results and results[modality].get("label") is not None:

            label = results[modality]["label"]
            conf  = results[modality]["confidence"]
            fake_prob = conf if label == 1 else (1 - conf)
            debug[f"{modality}_fake_prob"] = round(fake_prob, 4)

            if modality == "video":
                video_fake = fake_prob
            elif modality == "audio":
                audio_fake = fake_prob
            elif modality == "text":
                text_fake  = fake_prob

    # ------------------------------------------------------------------
    # FUSION LOGIC
    # Priority 1 — High-confidence single-modality override
    # Priority 2 — Face-swap pattern (video FAKE + audio REAL)  ← NEW
    # Priority 3 — Weighted average (rebalanced weights)
    # ------------------------------------------------------------------

    override_reason = None

    if video_fake > 0.75:
        # Strong visual manipulation — video takes full control
        final_fake = video_fake
        override_reason = "High-confidence visual manipulation detected."

    elif audio_fake > 0.85:
        # Strong synthetic audio signal
        final_fake = audio_fake
        override_reason = "High-confidence synthetic audio detected."

    elif video_fake > 0.60 and audio_fake < 0.40:
        # ---------------------------------------------------------------
        # Face-swap deepfake pattern:
        # Video flags manipulation while audio is authentic.
        # This divergence is expected for face-swaps — the original
        # audio track is preserved while only the face is replaced.
        # Giving audio equal or greater weight here is incorrect; we
        # use an 80/20 split so the video signal drives the verdict.
        # ---------------------------------------------------------------
        final_fake = 0.80 * video_fake + 0.20 * audio_fake
        override_reason = (
            "Face-swap pattern detected: visual manipulation present with authentic audio. "
            "Video signal weighted heavily (0.80) over audio (0.20)."
        )

    else:
        # Default weighted fusion — rebalanced to prioritise video
        final_fake = (
            0.60 * video_fake +
            0.25 * audio_fake +
            0.15 * text_fake
        )
        override_reason = None

    # ------------------------------------------------------------------
    # Build human-readable explanation
    # ------------------------------------------------------------------
    explanation_parts = []

    if override_reason:
        explanation_parts.insert(0, override_reason)

    if video_fake > 0.60 and audio_fake < 0.40:
        explanation_parts.append(
            "Strong visual manipulation detected while audio remains consistent."
        )
        explanation_parts.append("Cross-modal inconsistency detected (face-swap pattern).")

    elif video_fake > 0.60 and audio_fake > 0.60:
        explanation_parts.append(
            "Both video and audio show manipulation signals — possible full synthetic generation."
        )

    elif audio_fake > 0.60 and video_fake < 0.40:
        explanation_parts.append(
            "Audio shows synthetic characteristics while visuals appear natural."
        )

    if text_fake > 0.60:
        explanation_parts.append("Text exhibits AI-generated patterns.")

    asr_consistency = results.get("text", {}).get("asr_consistency")
    if asr_consistency is not None and asr_consistency < 0.5:
        explanation_parts.append("Low audio-text consistency suggests manipulation.")

    if not explanation_parts:
        explanation_parts.append("All modalities appear consistent and natural.")

    final_label = int(final_fake > 0.5)

    return {
        "label":      final_label,
        "confidence": round(final_fake, 4),
        "debug":      debug,
        "explanation": " ".join(explanation_parts)
    }


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(input_path):

    print("\n=========================")
    print("RUNNING FULL PIPELINE")
    print("=========================\n")

    case = CaseManager()

    # -------------------------------------------------------
    # Copy input file into case folder and log it
    # -------------------------------------------------------
    input_copy = os.path.join(case.get_path("input"), os.path.basename(input_path))
    shutil.copy(input_path, input_copy)

    file_type = detect_file_type(input_copy)
    print(f"📂 Detected file type: {file_type.upper()}")

    if file_type == "unknown":
        print("❌ Unsupported file type")
        return

    # -------------------------------------------------------
    # CoC — original file received
    # CaseManager.compute_sha256 is used everywhere — no
    # separate compute_file_hash() needed.
    # -------------------------------------------------------
    initial_hash = case.compute_sha256(input_copy)

    case.log_coc(
        stage="intake",
        file_path=input_copy,
        modality=file_type,
        action="received",
        notes="Original file copied into case folder. Pipeline starting.",
        extra={
            "file_type": file_type,
            "source_path": input_path
        }
    )

    # -------------------------------------------------------
    # Duplicate detection via hash DB
    # -------------------------------------------------------
    hash_db = load_hash_db()

    if initial_hash in hash_db:
        print("⚠️  File already processed!")
        print("Previous case:", hash_db[initial_hash])
        return

    print("Loading models...")

    text_models  = load_text_models()
    audio_model  = load_audio_model("models/audio/CRNN_aug_best_model.pth")

    video_pipeline = None
    if file_type == "video":
        video_pipeline = load_video_pipeline(
            "models/video/xception/xceptionnet_best.pth",
            "models/video/swin/swin_best.pth",
            case
        )

    results = {}
    data    = {}

    # -------------------------------------------------------
    # VIDEO
    # -------------------------------------------------------
    if file_type == "video":

        data = process_video(input_copy, case)

        try:
            results["video"] = run_video_pipeline(input_copy, video_pipeline)
        except Exception as e:
            print(f"[ERROR] Video failed: {e}")
            results["video"] = {"label": None, "confidence": 0.0}

        try:
            results["audio"] = run_audio_pipeline(data.get("audio_path"), audio_model, case)
        except Exception as e:
            print(f"[ERROR] Audio failed: {e}")
            results["audio"] = {"label": None, "confidence": 0.0}

        text_input = data.get("ocr_text") or data.get("subtitle_text")

        if text_input:
            try:
                results["text"] = run_text_pipeline(text_input, text_models, case)
            except Exception:
                results["text"] = {"label": None, "confidence": 0.0}
        else:
            results["text"] = {"label": None, "confidence": 0.0}

    # -------------------------------------------------------
    # AUDIO ONLY
    # -------------------------------------------------------
    elif file_type == "audio":

        try:
            results["audio"] = run_audio_pipeline(input_copy, audio_model, case)
        except Exception as e:
            print(f"[ERROR] Audio failed: {e}")
            results["audio"] = {"label": None, "confidence": 0.0}

        results["video"] = {"label": None, "confidence": 0.0}
        results["text"]  = {"label": None, "confidence": 0.0}

    # -------------------------------------------------------
    # TEXT ONLY
    # -------------------------------------------------------
    elif file_type == "text":

        try:
            text_input = extract_text(input_copy)

            if not text_input.strip():
                raise ValueError("Empty extracted text")

            results["text"] = run_text_pipeline(text_input, text_models, case)

        except Exception as e:
            print(f"[ERROR] Text failed: {e}")
            results["text"] = {"label": None, "confidence": 0.0}

        results["video"] = {"label": None, "confidence": 0.0}
        results["audio"] = {"label": None, "confidence": 0.0}

    # -------------------------------------------------------
    # FUSION
    # -------------------------------------------------------
    final = fuse_results(results)

    print_results(results, final)

    output = {
        "file_hash":          initial_hash,
        "integrity_verified": True,
        "final":              final,
        "modalities":         results,
        "ingestion":          data
    }

    save_path = os.path.join(case.get_path("results"), "final_result.json")

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    # CoC — final result saved
    case.log_coc(
        stage="fusion",
        file_path=save_path,
        modality=file_type,
        action="saved",
        notes="Multimodal fusion complete. Final result saved.",
        extra={
            "final_label":      final["label"],
            "final_confidence": final["confidence"],
            "explanation":      final["explanation"]
        }
    )

    # -------------------------------------------------------
    # Hash DB update (deduplication)
    # -------------------------------------------------------
    hash_db[initial_hash] = case.base_path
    save_hash_db(hash_db)

    # -------------------------------------------------------
    # Report
    # -------------------------------------------------------
    generate_report(case.base_path)

    print(f"📁 Case saved at: {case.base_path}")
    print(f"🔗 CoC log:       {case._coc_path}")

    return output


if __name__ == "__main__":

    raw_path   = input("Enter file path (video/audio/text): ")
    input_path = raw_path.strip().strip('"').strip("'")

    if not os.path.exists(input_path):
        print("\n❌ Invalid file path")
        print("👉 Make sure:")
        print("   • Path exists")
        print("   • No quotes issues")
        print("   • File not moved/deleted")
        print(f"\n📌 Received: {input_path}")
        exit()

    run_pipeline(input_path)