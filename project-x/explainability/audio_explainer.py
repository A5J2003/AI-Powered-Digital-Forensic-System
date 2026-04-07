import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import json

from models.audio.crnn_model import CRNN


# ==========================================
# CONFIG
# ==========================================
TARGET_SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 128
TARGET_FRAMES = 400

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# LOG-MEL
# ==========================================
def compute_logmel(audio_path):

    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

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

    return logmel


# ==========================================
# LOAD MODEL
# ==========================================
def load_audio_model(model_path):

    model = CRNN().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ==========================================
# GRAD-CAM (CRNN)
# ==========================================
activations = None
gradients = None


def save_act(m, i, o):
    global activations
    activations = o


def save_grad(m, gi, go):
    global gradients
    gradients = go[0]


def compute_gradcam(logmel, model):

    global activations, gradients

    x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    model.cnn[12].register_forward_hook(save_act)
    model.cnn[12].register_full_backward_hook(save_grad)

    was_training = model.training
    model.train()

    model.zero_grad()
    logits = model(x)

    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    one_hot = torch.zeros_like(logits)
    one_hot[0][pred_class] = 1

    logits.backward(gradient=one_hot)

    pooled = torch.mean(gradients, dim=[0, 2, 3])

    weighted = activations.clone()
    for i in range(weighted.shape[1]):
        weighted[:, i] *= pooled[i]

    heatmap = torch.mean(weighted, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-8

    heatmap = heatmap.cpu().detach().numpy()

    if not was_training:
        model.eval()

    return heatmap


# ==========================================
# ANALYSIS (FIXED FORENSIC LOGIC)
# ==========================================
def analyze_audio(logmel, heatmap):

    low_band = heatmap[:40, :]
    mid_band = heatmap[40:90, :]
    high_band = heatmap[90:, :]

    low_score = float(np.mean(low_band))
    mid_score = float(np.mean(mid_band))
    high_score = float(np.mean(high_band))

    band_scores = {
        "low": low_score,
        "mid": mid_score,
        "high": high_score
    }

    dominant_band = max(band_scores, key=band_scores.get)

    diff = np.abs(np.diff(heatmap, axis=1))
    spike_score = float(np.mean(diff))
    variation_score = float(np.std(heatmap))

    # ==========================================
    # SAFE TEMPORAL INTERPRETATION
    # ==========================================
    if spike_score > 0.5:
        temporal_comment = "High temporal instability observed."
    elif variation_score < 0.05:
        temporal_comment = "Low temporal variation observed (may indicate smooth regions)."
    else:
        temporal_comment = "Temporal dynamics appear consistent with natural speech."

    # ==========================================
    # FIXED PATTERN CLASSIFICATION (NO HARD CLAIMS)
    # ==========================================
    if spike_score > 0.5 and high_score > 0.4:
        pattern = "anomalous"
        pattern_reason = "Combined high-frequency emphasis and temporal instability."
    elif variation_score < 0.05:
        pattern = "low-variation"
        pattern_reason = "Reduced variation, possibly due to steady or quiet speech."
    else:
        pattern = "natural-like"
        pattern_reason = "Balanced spectral and temporal characteristics."

    # ==========================================
    # FIXED EXPLANATION (NON-CONTRADICTORY)
    # ==========================================
    explanation = (
        f"The model focuses primarily on the {dominant_band}-frequency region. "
        f"{temporal_comment} "
        f"Overall, the signal exhibits {pattern} characteristics due to {pattern_reason}. "
        f"These observations alone are not sufficient to conclude synthetic generation."
    )

    confidence = float(max(high_score, spike_score, variation_score))

    return {
        "band_scores": band_scores,
        "dominant_band": dominant_band,
        "temporal": {
            "spike_score": spike_score,
            "variation_score": variation_score,
            "comment": temporal_comment
        },
        "pattern": {
            "type": pattern,
            "reason": pattern_reason
        },
        "explanation": explanation.strip(),
        "explain_confidence": confidence
    }


# ==========================================
# SAVE OVERLAY
# ==========================================
def save_overlay(logmel, heatmap, save_path):

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        logmel,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel'
    )

    plt.imshow(
        heatmap,
        cmap='jet',
        alpha=0.4,
        aspect='auto',
        origin='lower'
    )

    plt.colorbar()
    plt.title("Audio Grad-CAM")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================
# MAIN
# ==========================================
def run_audio_explainability(audio_path, case):

    explain_dir = case.get_path("explain")
    os.makedirs(explain_dir, exist_ok=True)

    spec_path = os.path.join(explain_dir, "audio_logmel.png")
    gradcam_path = os.path.join(explain_dir, "audio_gradcam.png")
    analysis_path = os.path.join(explain_dir, "audio_explanation.json")

    logmel = compute_logmel(audio_path)

    # Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(logmel, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(spec_path)
    plt.close()

    # Model
    model_path = "models/audio/CRNN_aug_best_model.pth"
    model = load_audio_model(model_path)

    # Grad-CAM
    heatmap = compute_gradcam(logmel, model)
    save_overlay(logmel, heatmap, gradcam_path)

    # TIME SCORES
    time_scores = np.mean(heatmap, axis=0)
    time_scores = (time_scores - np.min(time_scores)) / (
        np.max(time_scores) - np.min(time_scores) + 1e-8
    )

    segment_scores = time_scores.tolist()

    # Duration
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    audio_duration = len(y) / sr

    # Analysis
    analysis = analyze_audio(logmel, heatmap)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=4)

    return {
        "spectrogram_path": spec_path,
        "gradcam_path": gradcam_path,
        "analysis_path": analysis_path,
        "explanation": analysis["explanation"],
        "details": {
            "segment_scores": segment_scores
        },
        "metadata": {
            "duration": audio_duration
        }
    }