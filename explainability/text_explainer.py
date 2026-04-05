# text_explainer.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from captum.attr import IntegratedGradients
from difflib import SequenceMatcher

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROBERTA_WEIGHT = 0.45
DEBERTA_WEIGHT = 0.55

LABELS = {0: "Human Written", 1: "AI Generated"}


# ==================================================
# VISUALIZATION FUNCTION
# ==================================================
def plot_token_attributions(words, scores, pred_class, pred_prob, save_path=None, top_n=20):

    if len(words) > top_n:
        pairs = sorted(zip(words, scores), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        words, scores = zip(*pairs)

    scores = np.array(scores)

    # 🔥 FIX: make color consistent with label
    if pred_class == 0:
        # REAL → positive = human (blue), negative = AI (red)
        colors = ["#3498db" if s > 0 else "#e74c3c" for s in scores]
        legend = "Blue = Human signal | Red = AI signal"
    else:
        # FAKE → reverse
        colors = ["#e74c3c" if s > 0 else "#3498db" for s in scores]
        legend = "Red = AI signal | Blue = Human signal"

    plt.figure(figsize=(12, max(4, len(words) * 0.4)))
    bars = plt.barh(range(len(words)), scores, color=colors)

    plt.yticks(range(len(words)), words)
    plt.axvline(0, linewidth=0.8)

    plt.title(
        f"Text Attribution → {LABELS[pred_class]} ({pred_prob*100:.1f}%)\n{legend}"
    )
    plt.xlabel("Attribution Score")

    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(
            score,
            i,
            f"{score:.3f}",
            va='center',
            ha='left' if score >= 0 else 'right',
            fontsize=8
        )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    try:
        if save_path is None:
            save_path = os.path.abspath("text_attribution.png")

        save_path = os.path.abspath(save_path)

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ [Text Explainability] Saved at: {save_path}")

    except Exception as e:
        print("❌ Error saving text attribution:", e)

    plt.close()


# ==================================================
# FORWARD FUNCTIONS
# ==================================================
def roberta_forward_embeds(inputs_embeds, attention_mask, model):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return F.softmax(outputs.logits, dim=1)


def deberta_forward_embeds(inputs_embeds, attention_mask, model):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return F.softmax(outputs.logits, dim=1)


# ==================================================
# SUBWORD → WORD
# ==================================================
def aggregate_subwords(tokens, attributions, prefix):
    words, scores = [], []
    current_word, current_score = "", 0.0

    special_tokens = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}

    for token, score in zip(tokens, attributions):
        if token in special_tokens:
            continue

        if token.startswith(prefix):
            if current_word:
                words.append(current_word)
                scores.append(current_score)
            current_word = token.replace(prefix, "")
            current_score = float(score)
        else:
            current_word += token
            current_score += float(score)

    if current_word:
        words.append(current_word)
        scores.append(current_score)

    return words, scores


# ==================================================
# ALIGN WORDS
# ==================================================
def align_words(words_a, scores_a, words_b, scores_b):
    matcher = SequenceMatcher(None, words_a, words_b)

    aligned_a, aligned_b = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                aligned_a.append(scores_a[i])
                aligned_b.append(scores_b[j])
        elif tag == "replace":
            for i in range(i1, i2):
                aligned_a.append(scores_a[i])
                aligned_b.append(0.0)
        elif tag == "insert":
            for j in range(j1, j2):
                aligned_a.append(0.0)
                aligned_b.append(scores_b[j])
        elif tag == "delete":
            for i in range(i1, i2):
                aligned_a.append(scores_a[i])
                aligned_b.append(0.0)

    final_words = words_a if len(words_a) >= len(words_b) else words_b
    return final_words, aligned_a, aligned_b


# ==================================================
# MAIN FUNCTION
# ==================================================
def explain_text(
    text,
    roberta_model,
    deberta_model,
    roberta_tokenizer,
    deberta_tokenizer,
    save_path=None,
    case_path=None
):

    if not text or len(text.strip()) == 0:
        return {
            "prediction": "N/A",
            "confidence": 0.0,
            "explanation": "No text available for analysis.",
            "details": {}
        }

    if save_path is None:
        if case_path:
            if hasattr(case_path, "get_path"):
                explain_dir = case_path.get_path("explain")
            else:
                explain_dir = os.path.join(case_path, "explainability")
        else:
            explain_dir = os.path.join(os.getcwd(), "explainability")

        os.makedirs(explain_dir, exist_ok=True)
        save_path = os.path.join(explain_dir, "text_attribution.png")

    print(f"📁 Saving text attribution to: {save_path}")

    enc_r = roberta_tokenizer(text, return_tensors="pt",
                              truncation=True, padding=True, max_length=256).to(DEVICE)

    enc_d = deberta_tokenizer(text, return_tensors="pt",
                              truncation=True, padding=True, max_length=256).to(DEVICE)

    with torch.no_grad():
        logits_r = roberta_model(**enc_r).logits
        logits_d = deberta_model(**enc_d).logits

        ensemble_logits = ROBERTA_WEIGHT * logits_r + DEBERTA_WEIGHT * logits_d
        probs = F.softmax(ensemble_logits, dim=1).squeeze()

        pred_class = torch.argmax(probs).item()
        pred_prob = probs[pred_class].item()

    embeds_r = roberta_model.roberta.embeddings(enc_r["input_ids"]).detach().requires_grad_(True)
    embeds_d = deberta_model.deberta.embeddings(enc_d["input_ids"]).detach().requires_grad_(True)

    baseline_r = roberta_model.roberta.embeddings(
        torch.full_like(enc_r["input_ids"], roberta_tokenizer.pad_token_id)
    )

    baseline_d = deberta_model.deberta.embeddings(
        torch.full_like(enc_d["input_ids"], deberta_tokenizer.pad_token_id)
    )

    ig_r = IntegratedGradients(lambda x, m: roberta_forward_embeds(x, m, roberta_model))
    ig_d = IntegratedGradients(lambda x, m: deberta_forward_embeds(x, m, deberta_model))

    attr_r = ig_r.attribute(
        embeds_r, baselines=baseline_r,
        additional_forward_args=(enc_r["attention_mask"],),
        target=pred_class, n_steps=20
    )

    attr_d = ig_d.attribute(
        embeds_d, baselines=baseline_d,
        additional_forward_args=(enc_d["attention_mask"],),
        target=pred_class, n_steps=20
    )

    attr_r = attr_r.sum(dim=-1).squeeze().detach().cpu().numpy()
    attr_d = attr_d.sum(dim=-1).squeeze().detach().cpu().numpy()

    tokens_r = roberta_tokenizer.convert_ids_to_tokens(enc_r["input_ids"].squeeze())
    tokens_d = deberta_tokenizer.convert_ids_to_tokens(enc_d["input_ids"].squeeze())

    words_r, scores_r = aggregate_subwords(tokens_r, attr_r, "Ġ")
    words_d, scores_d = aggregate_subwords(tokens_d, attr_d, "▁")

    words, r_aligned, d_aligned = align_words(words_r, scores_r, words_d, scores_d)

    final_scores = [
        ROBERTA_WEIGHT * r + DEBERTA_WEIGHT * d
        for r, d in zip(r_aligned, d_aligned)
    ]

    plot_token_attributions(words, final_scores, pred_class, pred_prob, save_path)

    # ==================================================
    # 🔥 FIXED EVIDENCE LOGIC
    # ==================================================
    word_scores = list(zip(words, final_scores))
    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)

    if pred_class == 0:  # REAL
        supporting = [(w, s) for w, s in word_scores if s > 0][:5]
        opposing = [(w, s) for w, s in word_scores if s < 0][:5]

        supporting_label = "Words supporting HUMAN classification"
        opposing_label = "Words indicating AI-like patterns"

    else:  # FAKE
        supporting = [(w, s) for w, s in word_scores if s < 0][:5]
        opposing = [(w, s) for w, s in word_scores if s > 0][:5]

        supporting_label = "Words supporting AI classification"
        opposing_label = "Words indicating HUMAN-like patterns"

    support_words = [w for w, _ in supporting]
    oppose_words = [w for w, _ in opposing]

    explanation_text = (
        f"Prediction: {LABELS[pred_class]}\n"
        f"Confidence: {pred_prob:.4f}\n\n"
        f"{supporting_label}: {support_words}\n"
        f"{opposing_label}: {oppose_words}"
    )

    return {
        "prediction": LABELS[pred_class],
        "confidence": pred_prob,
        "explanation": explanation_text,
        "details": {
            "supporting_words": support_words,
            "opposing_words": oppose_words,
            "supporting_label": supporting_label,
            "opposing_label": opposing_label
        }
    }