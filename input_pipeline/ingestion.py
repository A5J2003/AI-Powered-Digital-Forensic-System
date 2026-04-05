import os
import json

from input_pipeline.video_processing import (
    extract_audio,
    extract_frames,
    extract_subtitles
)
from input_pipeline.asr import transcribe_audio
from input_pipeline.ocr import extract_text_from_image
from input_pipeline.consistency import compute_similarity
from input_pipeline.metadata import extract_metadata
from input_pipeline.subtitle_utils import read_subtitle_file


def process_video(video_path, case):

    result = {
        "metadata": None,
        "audio_transcript": None,
        "ocr_text": None,
        "ocr_quality": None,
        "subtitle_text": None,
        "subtitle_density": None,
        "ocr_coverage_ratio": None,
        "asr_subtitle_similarity": None,
        "asr_ocr_consistency": None,

        "frames_path": None,
        "audio_path": None,
        "text": None,

        "errors": []
    }

    # ==========================================
    # CoC — ORIGINAL FILE RECEIVED
    # ==========================================
    case.log_coc(
        stage="ingestion",
        file_path=video_path,
        modality="video",
        action="received",
        notes="Original file submitted for analysis."
    )

    # ==========================================
    # 🔹 1️⃣ METADATA
    # ==========================================
    try:
        metadata = extract_metadata(video_path)
        result["metadata"] = metadata

        meta_path = os.path.join(case.get_path("metadata"), "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # CoC — metadata saved
        case.log_coc(
            stage="ingestion",
            file_path=meta_path,
            modality="metadata",
            action="extracted",
            notes="Structural metadata extracted from source video."
        )

    except Exception as e:
        result["errors"].append(f"Metadata extraction failed: {str(e)}")

    # ==========================================
    # 🔹 2️⃣ AUDIO
    # ==========================================
    try:
        audio_path = extract_audio(video_path, case.get_path("audio"))
        result["audio_path"] = audio_path

        # CoC — audio extracted
        case.log_coc(
            stage="ingestion",
            file_path=audio_path,
            modality="audio",
            action="extracted",
            notes="Audio track demuxed from source video via ffmpeg."
        )

    except Exception as e:
        result["errors"].append(f"Audio extraction failed: {str(e)}")
        audio_path = None

    # ==========================================
    # 🔹 3️⃣ FRAMES
    # ==========================================
    try:
        frames_dir = extract_frames(video_path, case.get_path("frames"))
        result["frames_path"] = frames_dir

        # CoC — log frame count as extra metadata (no single file to hash here)
        frame_files = sorted(os.listdir(frames_dir)) if frames_dir else []
        case.log_coc(
            stage="ingestion",
            file_path=frames_dir,   # directory — sha256 will be None (expected)
            modality="frames",
            action="extracted",
            notes="Video frames extracted at 1 fps via ffmpeg.",
            extra={"frame_count": len(frame_files)}
        )

    except Exception as e:
        result["errors"].append(f"Frame extraction failed: {str(e)}")
        frames_dir = None

    # ==========================================
    # 🔹 4️⃣ SUBTITLES
    # ==========================================
    try:
        subtitle_path = extract_subtitles(video_path, case.get_path("text"))
        subtitle_text = read_subtitle_file(subtitle_path)
        result["subtitle_text"] = subtitle_text

        if subtitle_path:
            case.log_coc(
                stage="ingestion",
                file_path=subtitle_path,
                modality="text",
                action="extracted",
                notes="Embedded subtitle stream extracted from video container."
            )

    except Exception as e:
        result["errors"].append(f"Subtitle extraction failed: {str(e)}")
        subtitle_text = None

    # ==========================================
    # 🔹 5️⃣ ASR
    # ==========================================
    transcript = None
    if audio_path:
        try:
            transcript = transcribe_audio(audio_path)
            result["audio_transcript"] = transcript
        except Exception as e:
            result["errors"].append(f"ASR failed: {str(e)}")

    # ==========================================
    # 🔥 6️⃣ OCR
    # ==========================================
    ocr_segments = []
    quality_scores = []
    previous_text = None
    text_frequency = {}

    total_frames = 0
    frames_with_text = 0

    if frames_dir:
        try:
            frame_files = sorted(os.listdir(frames_dir))
            total_frames = len(frame_files)

            for frame in frame_files:
                frame_path = os.path.join(frames_dir, frame)
                text, quality = extract_text_from_image(frame_path)

                if not text:
                    continue

                frames_with_text += 1

                if len(text.split()) < 3:
                    continue

                text_frequency[text] = text_frequency.get(text, 0) + 1

                if previous_text:
                    similarity = compute_similarity(text, previous_text)
                    if similarity > 0.85:
                        continue

                ocr_segments.append(text)
                quality_scores.append(quality)
                previous_text = text

        except Exception as e:
            result["errors"].append(f"OCR failed: {str(e)}")

    # ==========================================
    # 🔹 WATERMARK REMOVAL
    # ==========================================
    watermark_detected = False
    filtered_segments = []

    for text in ocr_segments:
        if total_frames > 0 and text_frequency.get(text, 0) > total_frames * 0.3:
            watermark_detected = True
            continue
        filtered_segments.append(text)

    combined_ocr = " ".join(filtered_segments).strip()
    result["ocr_text"] = combined_ocr

    # ==========================================
    # 🔹 OCR QUALITY
    # ==========================================
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    result["ocr_quality"] = round(avg_quality, 2)

    # ==========================================
    # 🔹 SUBTITLE DENSITY
    # ==========================================
    subtitle_density = (frames_with_text / total_frames) if total_frames else 0
    result["subtitle_density"] = round(subtitle_density, 2)

    # ==========================================
    # 🔹 COVERAGE
    # ==========================================
    asr_word_count = len(transcript.split()) if transcript else 0
    ocr_word_count = len(combined_ocr.split()) if combined_ocr else 0

    coverage_ratio = (ocr_word_count / asr_word_count) if asr_word_count else 0
    result["ocr_coverage_ratio"] = round(coverage_ratio, 2)

    # ==========================================
    # 🔹 SIMILARITY
    # ==========================================
    if transcript and subtitle_text:
        try:
            result["asr_subtitle_similarity"] = compute_similarity(
                transcript,
                subtitle_text
            )
        except Exception as e:
            result["errors"].append(f"ASR-Subtitle similarity failed: {str(e)}")

    similarity_score = None
    consistency_label = "Insufficient Data"

    if transcript and combined_ocr and avg_quality >= 0.5:
        try:
            similarity_score = compute_similarity(transcript, combined_ocr)

            if similarity_score < 0.3:
                consistency_label = "Low Match"
            elif similarity_score < 0.6:
                consistency_label = "Partial Match"
            else:
                consistency_label = "Strong Match"

        except Exception as e:
            result["errors"].append(f"ASR-OCR similarity failed: {str(e)}")

    result["asr_ocr_consistency"] = {
        "score": similarity_score,
        "label": consistency_label,
        "watermark_detected": watermark_detected
    }

    # ==========================================
    # 🔥 FINAL TEXT
    # ==========================================
    combined_text = " ".join(filter(None, [
        transcript,
        subtitle_text,
        combined_ocr
    ])).strip()

    result["text"] = combined_text if combined_text else None

    # ==========================================
    # 💾 SAVE FINAL RESULT
    # ==========================================
    save_path = os.path.join(case.get_path("results"), "ingestion_result.json")

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

    # CoC — ingestion result saved
    case.log_coc(
        stage="ingestion",
        file_path=save_path,
        modality="metadata",
        action="saved",
        notes="Ingestion result summary saved.",
        extra={
            "asr_words": asr_word_count,
            "ocr_words": ocr_word_count,
            "watermark_detected": watermark_detected,
            "consistency_label": consistency_label
        }
    )

    return result