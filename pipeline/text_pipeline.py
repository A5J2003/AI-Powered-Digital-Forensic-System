import torch
import torch.nn.functional as F
from transformers import (
    RobertaTokenizer,
    DebertaV2Tokenizer,
    RobertaForSequenceClassification,
    DebertaV2ForSequenceClassification
)

import os
import json

from explainability.text_explainer import explain_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD CONFIG
# =========================
def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "models", "text", "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


# =========================
# SAFE CHECKPOINT LOADER
# =========================
def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model


# =========================
# LOAD MODELS
# =========================
def load_text_models():

    config = load_config()

    roberta_path = config["roberta_model_path"]
    deberta_path = config["deberta_model_path"]

    w_r = config["ensemble_weights"]["roberta"]
    w_d = config["ensemble_weights"]["deberta"]

    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )
    roberta_model = load_checkpoint(roberta_model, roberta_path)
    roberta_model.to(DEVICE).eval()

    deberta_model = DebertaV2ForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2
    )
    deberta_model = load_checkpoint(deberta_model, deberta_path)
    deberta_model.to(DEVICE).eval()

    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

    return {
        "roberta_model": roberta_model,
        "deberta_model": deberta_model,
        "roberta_tokenizer": roberta_tokenizer,
        "deberta_tokenizer": deberta_tokenizer,
        "weights": {
            "roberta": w_r,
            "deberta": w_d
        },
        "config": config
    }


# =========================
# PREDICT
# =========================
def predict_text(text, models):

    roberta_model = models["roberta_model"]
    deberta_model = models["deberta_model"]
    roberta_tokenizer = models["roberta_tokenizer"]
    deberta_tokenizer = models["deberta_tokenizer"]

    w_r = models["weights"]["roberta"]
    w_d = models["weights"]["deberta"]

    max_len = models["config"].get("max_length", 512)

    enc_r = roberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(DEVICE)

    enc_d = deberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(DEVICE)

    with torch.no_grad():
        logits_r = roberta_model(**enc_r).logits
        logits_d = deberta_model(**enc_d).logits

        ensemble_logits = (w_r * logits_r) + (w_d * logits_d)
        probs = F.softmax(ensemble_logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return {
        "label": pred,
        "confidence": round(confidence, 4),
        "probabilities": probs.cpu().numpy().tolist()
    }


# =========================
# PIPELINE ENTRY
# =========================
def run_text_pipeline(text, models, case):

    print("Running TEXT pipeline...")

    # ==========================================
    # 🔥 TEXT VALIDITY CHECK
    # ==========================================
    if text is None or len(text.strip()) < 20 or len(text.split()) < 5:
        print("[TextPipeline] Skipping text (insufficient content)")

        result = {
            "label": None,
            "confidence": 0.0,
            "details": {},
            "explanation": {
                "explanation": "Insufficient text for reliable analysis."
            }
        }

        result_path = os.path.join(case.get_path("results"), "text_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

        return result

    # ==========================================
    # CoC — TEXT SUBMITTED FOR ANALYSIS
    # Text is a string (not a file), so we hash
    # the content directly and note its source.
    # ==========================================
    import hashlib
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    case.log_coc(
        stage="text",
        file_path=None,         # no physical file at this stage
        modality="text",
        action="analysed",
        notes="Extracted text submitted to RoBERTa+DeBERTa ensemble for AI-detection.",
        extra={
            "text_sha256": text_hash,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    )

    # ==========================================
    # 🔹 PREDICTION
    # ==========================================
    prediction = predict_text(text, models)

    explanation = None

    # ==========================================
    # 🔥 EXPLAINABILITY
    # ==========================================
    try:
        explanation = explain_text(
            text,
            models["roberta_model"],
            models["deberta_model"],
            models["roberta_tokenizer"],
            models["deberta_tokenizer"],
            case_path=case
        )
    except Exception as e:
        print(f"[Warning] Text explainability failed: {e}")
        explanation = {"error": str(e)}

    # ==========================================
    # 📊 FINAL RESULT
    # ==========================================
    result = {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "details": prediction,
        "explanation": explanation
    }

    # ==========================================
    # 💾 SAVE RESULTS
    # ==========================================
    result_path = os.path.join(case.get_path("results"), "text_result.json")

    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    # CoC — result file saved
    case.log_coc(
        stage="text",
        file_path=result_path,
        modality="text",
        action="saved",
        notes="Text detection result saved.",
        extra={
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "text_sha256": text_hash
        }
    )

    # ==========================================
    # 💾 SAVE EXPLANATION
    # ==========================================
    explain_path = os.path.join(case.get_path("explain"), "text_explanation.json")

    with open(explain_path, "w") as f:
        json.dump(explanation, f, indent=4)

    case.log_coc(
        stage="text",
        file_path=explain_path,
        modality="text",
        action="saved",
        notes="Text explainability (Integrated Gradients) output saved."
    )

    return result