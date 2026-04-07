import torch
import numpy as np
import librosa
import soundfile as sf
import os
import json

from models.audio.crnn_model import CRNN
from explainability.audio_explainer import run_audio_explainability


# ==========================================
# CONFIG (MATCHES TRAINING)
# ==========================================
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 128
TARGET_FRAMES = 400

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_logmel(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        y, sr = sf.read(audio_path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except RuntimeError:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    logmel = np.log(mel + 1e-9)

    T = logmel.shape[1]
    if T > TARGET_FRAMES:
        logmel = logmel[:, :TARGET_FRAMES]
    else:
        logmel = np.pad(logmel, ((0, 0), (0, TARGET_FRAMES - T)))

    return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ==========================================
# LOAD MODEL
# ==========================================
def load_audio_model(model_path):

    print("Loading AUDIO model (CRNN)...")

    model = CRNN().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ==========================================
# PREDICTION
# ==========================================
def predict_audio(audio_path, model):

    print("Running AUDIO pipeline (CRNN)...")

    x = extract_logmel(audio_path).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    fake_prob = probs[0][1].item()
    label = 1 if fake_prob > 0.5 else 0

    return {
        "label": label,
        "confidence": round(fake_prob if label == 1 else (1 - fake_prob), 4),
        "fake_prob": round(fake_prob, 4),
        "real_prob": round(1 - fake_prob, 4)
    }


# ==========================================
# MAIN PIPELINE
# ==========================================
def run_audio_pipeline(audio_path, model, case):

    # ==========================================
    # 🔥 AUDIO VALIDITY CHECK
    # ==========================================
    if audio_path is None or not os.path.exists(audio_path):
        print("[AudioPipeline] Skipping audio (no audio found)")

        result = {
            "label": None,
            "confidence": 0.0,
            "fake_prob": 0.0,
            "explainability": {
                "explanation": "No audio available for analysis."
            }
        }

        save_path = os.path.join(case.get_path("results"), "audio_result.json")
        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

        return result

    # ==========================================
    # CoC — AUDIO FILE RECEIVED FOR ANALYSIS
    # ==========================================
    case.log_coc(
        stage="audio",
        file_path=audio_path,
        modality="audio",
        action="analysed",
        notes="Audio file submitted to CRNN pipeline for deepfake detection."
    )

    # ==========================================
    # 🔥 LENGTH CHECK
    # ==========================================
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        audio_duration = len(y) / sr

        if len(y) < TARGET_SR * 1:
            print("[AudioPipeline] Skipping audio (too short)")

            result = {
                "label": None,
                "confidence": 0.0,
                "fake_prob": 0.0,
                "explainability": {
                    "explanation": "Audio too short for reliable analysis."
                }
            }

            save_path = os.path.join(case.get_path("results"), "audio_result.json")
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)

            return result

    except Exception as e:
        print(f"[Warning] Audio loading failed: {e}")
        audio_duration = None

    # ==========================================
    # 🔹 PREDICTION
    # ==========================================
    result = predict_audio(audio_path, model)

    # ==========================================
    # 🔥 EXPLAINABILITY
    # ==========================================
    explain_result = None

    try:
        explain_result = run_audio_explainability(audio_path, case)
    except Exception as e:
        print(f"[Warning] Audio explainability failed: {e}")
        explain_result = {"error": str(e)}

    # ==========================================
    # 📊 FINAL OUTPUT
    # ==========================================
    final_result = {
        **result,
        "explainability": explain_result
    }

    # ==========================================
    # 💾 SAVE RESULT
    # ==========================================
    save_path = os.path.join(case.get_path("results"), "audio_result.json")

    with open(save_path, "w") as f:
        json.dump(final_result, f, indent=4)

    # ==========================================
    # CoC — RESULT SAVED
    # ==========================================
    case.log_coc(
        stage="audio",
        file_path=save_path,
        modality="audio",
        action="saved",
        notes="Audio detection result saved.",
        extra={
            "label": result["label"],
            "fake_prob": result["fake_prob"],
            "confidence": result["confidence"],
            "duration_seconds": audio_duration
        }
    )

    return final_result