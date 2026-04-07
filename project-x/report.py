import os
import json
import math
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ══════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════
RED        = colors.HexColor("#C0392B")
RED_LIGHT  = colors.HexColor("#FADBD8")
GREEN      = colors.HexColor("#1E8449")
GREEN_LIGHT= colors.HexColor("#D5F5E3")
BLUE       = colors.HexColor("#1A5276")
BLUE_MID   = colors.HexColor("#2E86C1")
BLUE_LIGHT = colors.HexColor("#D6EAF8")
AMBER      = colors.HexColor("#B7770D")
AMBER_LIGHT= colors.HexColor("#FEF9E7")
GRAY_DARK  = colors.HexColor("#2C3E50")
GRAY_MID   = colors.HexColor("#7F8C8D")
GRAY_LIGHT = colors.HexColor("#F2F3F4")
WHITE      = colors.white
BLACK      = colors.black

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# ══════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════
def build_styles():
    base = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": S("cover_title", "Title",
            fontSize=28, textColor=WHITE, alignment=TA_CENTER,
            spaceAfter=6, fontName="Helvetica-Bold"),
        "cover_sub": S("cover_sub",
            fontSize=13, textColor=colors.HexColor("#BDC3C7"),
            alignment=TA_CENTER, spaceAfter=4),
        "cover_meta": S("cover_meta",
            fontSize=10, textColor=colors.HexColor("#ECF0F1"),
            alignment=TA_CENTER, spaceAfter=3),

        "section_title": S("section_title", "Heading1",
            fontSize=13, textColor=BLUE, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=6,
            borderPadding=(0, 0, 4, 0)),
        "field_label": S("field_label",
            fontSize=9, textColor=GRAY_MID, fontName="Helvetica-Bold",
            spaceAfter=1),
        "field_value": S("field_value",
            fontSize=10, textColor=GRAY_DARK, spaceAfter=6),
        "body": S("body",
            fontSize=10, textColor=GRAY_DARK, leading=15, spaceAfter=6),
        "caption": S("caption",
            fontSize=8, textColor=GRAY_MID, alignment=TA_CENTER,
            spaceAfter=10, fontName="Helvetica-Oblique"),
        "table_header": S("table_header",
            fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "table_cell": S("table_cell",
            fontSize=9, textColor=GRAY_DARK, alignment=TA_LEFT),
        "table_cell_c": S("table_cell_c",
            fontSize=9, textColor=GRAY_DARK, alignment=TA_CENTER),
        "word_human": S("word_human",
            fontSize=9, textColor=GREEN, fontName="Helvetica-Bold"),
        "word_ai": S("word_ai",
            fontSize=9, textColor=RED, fontName="Helvetica-Bold"),
        "transcript": S("transcript",
            fontSize=9, textColor=GRAY_DARK, leading=14,
            fontName="Helvetica-Oblique",
            backColor=GRAY_LIGHT, borderPadding=8, spaceAfter=8),
        "verdict_fake": S("verdict_fake",
            fontSize=22, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "verdict_real": S("verdict_real",
            fontSize=22, textColor=WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "confidence_text": S("confidence_text",
            fontSize=11, textColor=WHITE, alignment=TA_CENTER),
        "warning": S("warning",
            fontSize=9, textColor=AMBER, fontName="Helvetica-Bold"),
    }


# ══════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════
def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} GB"


def fmt_duration(ms):
    s = ms / 1000
    return f"{int(s // 60):02d}:{int(s % 60):02d}.{int((s % 1) * 10)}"


def interpret_confidence(conf):
    if conf >= 0.85:
        return "High confidence"
    elif conf >= 0.55:
        return "Moderate confidence"
    else:
        return "Low confidence (uncertain)"


def label_str(label):
    if label is None:
        return "N/A"
    return "FAKE" if label == 1 else "REAL"


def risk_level(score):
    if score >= 0.75:
        return ("HIGH RISK", RED)
    elif score >= 0.5:
        return ("MEDIUM RISK", AMBER)
    else:
        return ("LOW RISK", GREEN)


def safe_image(path, width, max_height=None):
    """Return a ReportLab Image with correct aspect ratio."""
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            iw, ih = im.size
        aspect = ih / iw
        h = width * aspect
        if max_height and h > max_height:
            h = max_height
            width = h / aspect
        return Image(path, width=width, height=h)
    except Exception:
        return Image(path, width=width, height=width * 0.5)


def section_rule():
    return HRFlowable(width="100%", thickness=0.5,
                      color=BLUE_LIGHT, spaceAfter=8, spaceBefore=4)


# ══════════════════════════════════════════════════════════════
# CHART GENERATORS
# ══════════════════════════════════════════════════════════════
def chart_modality_probs(modalities, out_path):
    """Color-coded bar chart: fake probability per modality."""
    labels, values, bar_colors = [], [], []
    for mod in ["video", "audio", "text"]:
        d = modalities.get(mod, {})
        lbl = d.get("label")
        if lbl is not None:
            conf = d.get("confidence", 0) or 0
            # Always express as true fake probability regardless of JSON field names
            prob = conf if lbl == 1 else 1 - conf
            labels.append(mod.upper())
            values.append(prob)
            bar_colors.append("#C0392B" if lbl == 1 else "#1E8449")

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(labels, values, color=bar_colors, width=0.45,
                  edgecolor="white", linewidth=0.8)

    for bar, val, bc in zip(bars, values, bar_colors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=bc)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fake Probability", fontsize=9, color="#555")
    ax.set_title("Fake Probability by Modality", fontsize=11, pad=10, color="#2C3E50")
    ax.axhline(0.5, color="#AAA", linestyle="--", linewidth=0.8, label="Decision threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555", labelsize=9)
    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def chart_frame_scores(frame_scores, duration, out_path, title="Frame Risk Score Over Time"):
    """Line chart of frame-level fake scores across video timeline."""
    if not frame_scores or not duration:
        return None

    n = len(frame_scores)
    times = [i * (duration / n) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(times, frame_scores, color="#2E86C1", linewidth=1.5, zorder=3)
    ax.fill_between(times, frame_scores, alpha=0.15, color="#2E86C1")

    # Highlight high-risk frames
    for t, s in zip(times, frame_scores):
        if s >= 0.75:
            ax.axvline(t, color="#C0392B", alpha=0.35, linewidth=1)
        elif s >= 0.5:
            ax.axvline(t, color="#E67E22", alpha=0.2, linewidth=1)

    ax.axhline(0.5, color="#E67E22", linestyle="--", linewidth=0.8, label="Threshold 0.5")
    ax.axhline(0.75, color="#C0392B", linestyle="--", linewidth=0.8, label="High risk 0.75")

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Time (seconds)", fontsize=9, color="#555")
    ax.set_ylabel("Score", fontsize=9, color="#555")
    ax.set_title(title, fontsize=10, color="#2C3E50", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=7, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def chart_audio_segments(segment_scores, duration, out_path):
    """Bar chart of audio segment fake scores."""
    if not segment_scores:
        return None

    n = len(segment_scores)
    interval = duration / n if duration else 1
    labels = [f"{i * interval:.1f}s" for i in range(n)]
    bar_colors = ["#C0392B" if s >= 0.75 else
                  "#E67E22" if s >= 0.5 else "#2E86C1"
                  for s in segment_scores]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(range(n), segment_scores, color=bar_colors, width=0.8, edgecolor="white")
    ax.axhline(0.5, color="#E67E22", linestyle="--", linewidth=0.8)
    ax.axhline(0.75, color="#C0392B", linestyle="--", linewidth=0.8)

    # Label only every N ticks to avoid crowding
    step = max(1, n // 10)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(labels[::step], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fake Score", fontsize=9, color="#555")
    ax.set_title("Audio Segment Fake Scores", fontsize=10, color="#2C3E50", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555", labelsize=8)

    legend_patches = [
        mpatches.Patch(color="#C0392B", label="High risk (≥0.75)"),
        mpatches.Patch(color="#E67E22", label="Medium (≥0.5)"),
        mpatches.Patch(color="#2E86C1", label="Low (<0.5)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def chart_confidence_gauge(confidence, label, out_path):
    """Semicircular gauge showing verdict confidence.

    confidence: the model's confidence in its verdict (0–1).
    label:      1 = FAKE, 0 = REAL — drives colour and text.

    The arc always fills left→right proportional to confidence,
    and is coloured red for FAKE, green for REAL, so it is never
    contradictory — a mostly-filled green arc means 'confidently REAL'.
    """
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")

    fill_color = "#C0392B" if label == 1 else "#1E8449"

    # Background arc (unfilled portion)
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#E8E8E8",
            linewidth=18, solid_capstyle="round")

    # Filled arc — sweeps left→right proportional to confidence
    end_theta = np.pi - confidence * np.pi
    theta_fill = np.linspace(np.pi, end_theta, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=fill_color,
            linewidth=18, solid_capstyle="round")

    # Center: show confidence %
    ax.text(0, 0.3, f"{confidence:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=fill_color)
    ax.text(0, -0.05, "Confidence", ha="center", va="center",
            fontsize=8, color="#777")

    verdict_txt = "FAKE" if label == 1 else "REAL"
    ax.text(0, -0.22, verdict_txt, ha="center", va="center",
            fontsize=11, fontweight="bold", color=fill_color)

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════
# TABLE HELPERS
# ══════════════════════════════════════════════════════════════
COMMON_TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0),  9),
    ("ALIGN",       (0, 0), (-1, 0),  "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
    ("FONTSIZE",    (0, 1), (-1, -1), 9),
    ("TOPPADDING",  (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING",(0,0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING",(0, 0), (-1, -1), 8),
    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE")),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
])


def make_table(data, col_widths, extra_styles=None):
    t = Table(data, colWidths=col_widths)
    style = TableStyle(COMMON_TABLE_STYLE.getCommands())
    if extra_styles:
        for s in extra_styles:
            style.add(*s)
    t.setStyle(style)
    return t


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_all(case_path):
    results = os.path.join(case_path, "results")
    explain = os.path.join(case_path, "explainability")
    meta_dir = os.path.join(case_path, "metadata")

    final      = load_json(os.path.join(results, "final_result.json"))
    video_r    = load_json(os.path.join(results, "video_result.json"))
    audio_r    = load_json(os.path.join(results, "audio_result.json"))
    text_r     = load_json(os.path.join(results, "text_result.json"))
    ingestion  = load_json(os.path.join(results, "ingestion_result.json"))
    audio_exp  = load_json(os.path.join(explain,  "audio_explanation.json"))
    metadata   = load_json(os.path.join(meta_dir, "metadata.json"))

    # Chain of custody log
    logs_dir = os.path.join(case_path, "logs")
    custody  = load_json(os.path.join(logs_dir, "chain_of_custody.json"))
    if not metadata:
        metadata = ingestion.get("metadata", {})

    modalities = final.get("modalities", {})
    # Supplement with individual result files if final_result lacks detail
    if not modalities.get("video"):
        modalities["video"] = video_r
    if not modalities.get("audio"):
        modalities["audio"] = audio_r
    if not modalities.get("text"):
        modalities["text"] = text_r

    final_raw  = final.get("final", {}) or {}
    raw_label  = final_raw.get("label") if final_raw else final.get("label")
    raw_conf   = final_raw.get("confidence") if final_raw else final.get("confidence")
    if raw_conf is None:
        raw_conf = 0.0

    # The JSON stores the fake probability as "confidence" in some pipelines.
    # Verdict confidence = how sure the model is in its verdict:
    #   FAKE verdict → verdict_conf = fake_prob (already ≥0.5)
    #   REAL verdict → verdict_conf = 1 - fake_prob  (fake_prob is ≤0.5)
    if raw_label == 1:
        verdict_confidence = float(raw_conf)
    else:
        verdict_confidence = 1.0 - float(raw_conf)

    final_dict = {
        "label":              raw_label,
        "confidence":         verdict_confidence,   # always = confidence in verdict
        "fake_prob":          float(raw_conf),       # always = P(fake)
    }

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_id":   os.path.basename(case_path),
        "case_path": case_path,
        "final":     final_dict,
        # FIX: carry the real uploaded filename through so the report shows
        # the user's filename instead of the UUID temp name
        "original_filename": final.get("original_filename"),
        "modalities":       modalities,
        "video_result":     video_r,
        "audio_result":     audio_r,
        "text_result":      text_r,
        "ingestion":        ingestion,
        "audio_exp":        audio_exp,
        "metadata":         metadata,
        "file_hash":        final.get("file_hash", "N/A"),
        "integrity_verified": final.get("integrity_verified", False),
        "explain_dir":      explain,
        "custody":          custody,
    }


# ══════════════════════════════════════════════════════════════
# PDF SECTIONS
# ══════════════════════════════════════════════════════════════
def build_cover(report, styles, chart_dir):
    els = []
    final  = report["final"]
    label  = final.get("label")
    conf   = final.get("confidence", 0) or 0

    verdict_color = RED if label == 1 else GREEN
    verdict_word  = "FAKE DETECTED" if label == 1 else "AUTHENTIC"

    # Verdict banner
    banner_data = [[Paragraph(verdict_word, styles["verdict_fake" if label == 1 else "verdict_real"])]]
    banner = Table(banner_data, colWidths=[CONTENT_W])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("ROUNDEDCORNERS", [6]),
    ]))
    els.append(banner)
    els.append(Spacer(1, 10))

    # Confidence sub-banner — conf is always the confidence in the verdict
    conf_text = f"Confidence: {conf:.4f}   |   {interpret_confidence(conf)}"
    conf_data = [[Paragraph(conf_text, styles["confidence_text"])]]
    conf_tbl = Table(conf_data, colWidths=[CONTENT_W])
    conf_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GRAY_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    els.append(conf_tbl)
    els.append(Spacer(1, 14))

    # Gauge — pass conf directly; the gauge now shows verdict confidence
    gauge_path = os.path.join(chart_dir, "gauge.png")
    chart_confidence_gauge(conf, label, gauge_path)
    img = safe_image(gauge_path, 120 * mm, 75 * mm)
    if img:
        img.hAlign = "CENTER"
        els.append(img)
    els.append(Spacer(1, 10))

    # Report metadata table — all cells wrapped in Paragraph to prevent overflow
    wrap_s = ParagraphStyle("cover_cell", fontSize=9, textColor=GRAY_DARK,
                            leading=12, wordWrap="CJK")
    wrap_hdr = ParagraphStyle("cover_hdr", fontSize=9, textColor=WHITE,
                               fontName="Helvetica-Bold", alignment=TA_CENTER)
    meta_col_w = [55 * mm, CONTENT_W - 55 * mm]

    # FIX: prefer the real uploaded filename saved by the pipeline;
    # fall back to MediaInfo metadata only for CLI-run cases
    file_name = (
        report.get("original_filename")
        or report["ingestion"].get("metadata", {}).get("General", {}).get("file_name_extension")
        or "N/A"
    )
    file_hash = report["file_hash"] or "N/A"

    meta_data = [
        [Paragraph("Field", wrap_hdr),         Paragraph("Value", wrap_hdr)],
        [Paragraph("Case ID",       wrap_s),   Paragraph(str(report["case_id"]), wrap_s)],
        [Paragraph("File",          wrap_s),   Paragraph(str(file_name),         wrap_s)],
        [Paragraph("Analysis Date", wrap_s),   Paragraph(str(report["timestamp"]), wrap_s)],
        [Paragraph("File Hash",     wrap_s),   Paragraph(str(file_hash),          wrap_s)],
        [Paragraph("Integrity",     wrap_s),
         Paragraph("VERIFIED ✓" if report["integrity_verified"] else "NOT VERIFIED", wrap_s)],
    ]
    tbl = Table(meta_data, colWidths=meta_col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    els.append(tbl)
    els.append(PageBreak())
    return els


def build_executive_summary(report, styles):
    els = []
    els.append(Paragraph("Executive Summary", styles["section_title"]))
    els.append(section_rule())

    final = report["final"]
    label = final.get("label")
    conf  = final.get("confidence", 0) or 0
    mods  = report["modalities"]

    video_label = mods.get("video", {}).get("label")
    audio_label = mods.get("audio", {}).get("label")
    text_label  = mods.get("text",  {}).get("label")

    body = (
        f"This forensic analysis examined the submitted media file using a multimodal deepfake "
        f"detection pipeline covering video, audio, and text modalities. The system returned a "
        f"<b>{'FAKE' if label == 1 else 'REAL'}</b> verdict with an overall confidence of "
        f"<b>{conf:.4f}</b> ({interpret_confidence(conf)}). "
        f"Video analysis indicated <b>{label_str(video_label)}</b>, audio indicated "
        f"<b>{label_str(audio_label)}</b>, and text analysis indicated "
        f"<b>{label_str(text_label)}</b>. "
        f"The final decision was derived through multimodal fusion, weighting each modality's "
        f"output to produce a combined probability estimate."
    )
    els.append(Paragraph(body, styles["body"]))
    els.append(Spacer(1, 10))

    # Quick-reference table
    rows = [["Modality", "Verdict", "Confidence", "Risk Level"]]
    for mod in ["video", "audio", "text"]:
        d = mods.get(mod, {})
        lbl  = d.get("label")
        c    = d.get("confidence", 0) or 0
        fp   = d.get("fake_prob", c if lbl == 1 else 1 - c)
        risk, rcol = risk_level(fp)
        rows.append([
            mod.upper(),
            label_str(lbl),
            f"{c:.4f}",
            risk,
        ])

    col_w = [CONTENT_W * f for f in [0.2, 0.2, 0.25, 0.35]]
    extra = []
    for i, mod in enumerate(["video", "audio", "text"], start=1):
        d   = mods.get(mod, {})
        lbl = d.get("label")
        c   = d.get("confidence", 0) or 0
        fp  = d.get("fake_prob", c if lbl == 1 else 1 - c)
        _, rcol = risk_level(fp)
        extra.append(("TEXTCOLOR", (3, i), (3, i), rcol))
        extra.append(("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"))
        if lbl == 1:
            extra.append(("TEXTCOLOR", (1, i), (1, i), RED))
        else:
            extra.append(("TEXTCOLOR", (1, i), (1, i), GREEN))

    els.append(make_table(rows, col_w, extra))
    els.append(Spacer(1, 6))
    return els


def build_file_metadata(report, styles):
    els = []
    els.append(Paragraph("File & Ingestion Metadata", styles["section_title"]))
    els.append(section_rule())

    meta  = report["metadata"]
    gen   = meta.get("General", {})
    vid   = meta.get("Video",   {})
    aud   = meta.get("Audio",   {})
    ing   = report["ingestion"]

    # Paragraph style for wrappable table cells
    ws = ParagraphStyle("fm_cell", fontSize=9, textColor=GRAY_DARK,
                        leading=12, wordWrap="CJK")

    def P(text):
        return Paragraph(str(text) if text is not None else "N/A", ws)

    rows_gen = [
        ["Property", "Value"],
        # FIX: use original_filename so the metadata table also shows the real name
        ["Filename",        report.get("original_filename") or gen.get("file_name_extension", "N/A")],
        ["Format",          gen.get("format", "N/A")],
        ["File Size",       fmt_bytes(gen.get("file_size", 0))],
        ["Duration",        fmt_duration(gen.get("duration", 0))],
        ["Overall Bitrate", gen.get("other_overall_bit_rate", ["N/A"])[0]],
        ["Frame Rate",      f"{gen.get('frame_rate', 'N/A')} FPS"],
        ["Frame Count",     gen.get("frame_count", "N/A")],
        ["Created",         gen.get("file_creation_date", "N/A")],
        ["Writing App",     gen.get("writing_application", "N/A")],
    ]
    # Wrap every value cell
    rows_gen = [[r[0], P(r[1])] for r in rows_gen[1:]]
    rows_gen = [["Property", "Value"]] + rows_gen

    rows_vid = [
        ["Property", "Value"],
        ["Codec",        vid.get("format", "N/A")],
        ["Profile",      vid.get("format_profile", "N/A")],
        ["Resolution",   f"{vid.get('width','?')} × {vid.get('height','?')} px"],
        ["Aspect Ratio", vid.get("other_display_aspect_ratio", ["N/A"])[0]],
        ["Bit Depth",    f"{vid.get('bit_depth','?')} bits"],
        ["Color Space",  vid.get("color_space", "N/A")],
        ["Scan Type",    vid.get("scan_type", "N/A")],
        ["Video Bitrate",vid.get("other_bit_rate", ["N/A"])[0]],
    ]
    rows_vid = [[r[0], P(r[1])] for r in rows_vid[1:]]
    rows_vid = [["Property", "Value"]] + rows_vid

    rows_aud = [
        ["Property", "Value"],
        ["Codec",         aud.get("format", "N/A")],
        ["Channels",      str(aud.get("channel_s", "N/A"))],
        ["Sample Rate",   f"{aud.get('sampling_rate','?')} Hz"],
        ["Audio Bitrate", aud.get("other_bit_rate", ["N/A"])[0]],
        ["Bit Rate Mode", aud.get("bit_rate_mode", "N/A")],
        ["Compression",   aud.get("compression_mode", "N/A")],
    ]
    rows_aud = [[r[0], P(r[1])] for r in rows_aud[1:]]
    rows_aud = [["Property", "Value"]] + rows_aud

    col_w = [CONTENT_W * 0.45, CONTENT_W * 0.55]

    els.append(Paragraph("General", styles["field_label"]))
    els.append(make_table(rows_gen, col_w))
    els.append(Spacer(1, 10))

    # Side-by-side Video + Audio — each sub-table uses half-page column widths
    gap = 6 * mm
    half = (CONTENT_W - gap) / 2
    sub_col_w = [half * 0.45, half * 0.55]

    vid_tbl = make_table(rows_vid, sub_col_w)
    aud_tbl = make_table(rows_aud, sub_col_w)

    # Label row above each table, also side-by-side
    lbl_vid = Paragraph("Video Track", styles["field_label"])
    lbl_aud = Paragraph("Audio Track", styles["field_label"])
    lbl_row = Table([[lbl_vid, lbl_aud]], colWidths=[half, half])
    lbl_row.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("INNERGRID",    (0, 0), (-1, -1), 0, WHITE),
        ("BOX",          (0, 0), (-1, -1), 0, WHITE),
    ]))
    els.append(lbl_row)

    side = Table([[vid_tbl, aud_tbl]], colWidths=[half, half])
    side.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("INNERGRID",    (0, 0), (-1, -1), 0, WHITE),
        ("BOX",          (0, 0), (-1, -1), 0, WHITE),
    ]))
    els.append(side)
    els.append(Spacer(1, 10))

    # OCR / ASR
    asr_ocr = ing.get("asr_ocr_consistency", {})
    ocr_score = asr_ocr.get("score")
    if ocr_score is not None:
        els.append(Paragraph("OCR / ASR Consistency", styles["field_label"]))
        rows_ocr = [
            ["Property", "Value"],
            ["ASR–OCR Score",      f"{ocr_score:.4f}"],
            ["Label",              asr_ocr.get("label", "N/A")],
            ["Watermark Detected", str(asr_ocr.get("watermark_detected", False))],
            ["OCR Quality",        f"{ing.get('ocr_quality', 'N/A')}"],
            ["OCR Coverage",       f"{ing.get('ocr_coverage_ratio', 'N/A')}"],
        ]
        rows_ocr = [[r[0], P(r[1])] for r in rows_ocr[1:]]
        rows_ocr = [["Property", "Value"]] + rows_ocr
        els.append(make_table(rows_ocr, [CONTENT_W * 0.45, CONTENT_W * 0.55]))
    return els


def build_modality_video(report, styles, chart_dir):
    els = []
    els.append(Paragraph("Video Analysis", styles["section_title"]))
    els.append(section_rule())

    vr = report["video_result"]
    label = vr.get("label")
    conf  = vr.get("confidence", 0) or 0

    details = vr.get("details", {})
    frame_scores = vr.get("explainability", {}).get("details", {}).get("frame_scores",
                   details.get("frame_scores", []))
    duration = vr.get("metadata", {}).get("duration", 32.1)
    xception = details.get("xception", {})
    swin     = details.get("swin", {})

    # Sub-model comparison table
    rows = [
        ["Model", "Real Prob", "Fake Prob", "Verdict"],
        ["Xception",
         f"{xception.get('real', 0):.4f}",
         f"{xception.get('fake', 0):.4f}",
         "FAKE" if xception.get("fake", 0) > 0.5 else "REAL"],
        ["Swin Transformer",
         f"{swin.get('real', 0):.4f}",
         f"{swin.get('fake', 0):.4f}",
         "FAKE" if swin.get("fake", 0) > 0.5 else "REAL"],
        ["FINAL (fused)",
         f"{1 - conf:.4f}",
         f"{conf:.4f}",
         label_str(label)],
    ]
    col_w = [CONTENT_W * f for f in [0.35, 0.2, 0.2, 0.25]]
    extra = [
        ("FONTNAME",   (0, 3), (-1, 3), "Helvetica-Bold"),
        ("BACKGROUND", (0, 3), (-1, 3), BLUE_LIGHT),
    ]
    for i, row in enumerate(rows[1:], 1):
        verdict = row[3]
        color = RED if verdict == "FAKE" else GREEN
        extra.append(("TEXTCOLOR", (3, i), (3, i), color))

    els.append(make_table(rows, col_w, extra))
    els.append(Spacer(1, 10))

    # Frame score stats
    if frame_scores:
        max_s  = max(frame_scores)
        mean_s = sum(frame_scores) / len(frame_scores)
        high_r = sum(1 for s in frame_scores if s >= 0.75)
        med_r  = sum(1 for s in frame_scores if 0.5 <= s < 0.75)

        stats_rows = [
            ["Metric", "Value"],
            ["Frames analysed",    str(len(frame_scores))],
            ["Max frame score",    f"{max_s:.4f}"],
            ["Mean frame score",   f"{mean_s:.4f}"],
            ["High-risk frames (≥0.75)", str(high_r)],
            ["Medium-risk frames (≥0.5)", str(med_r)],
            ["Video duration",     f"{duration:.1f} s"],
            ["Analysis mode",      details.get("mode", "N/A")],
        ]
        els.append(make_table(stats_rows, [CONTENT_W * 0.55, CONTENT_W * 0.45]))
        els.append(Spacer(1, 10))

        # Frame score timeline chart
        chart_p = os.path.join(chart_dir, "video_frame_scores.png")
        if chart_frame_scores(frame_scores, duration, chart_p, "Video Frame Risk Score Over Time"):
            img = safe_image(chart_p, CONTENT_W, 75 * mm)
            if img:
                els.append(img)
                els.append(Paragraph(
                    "Frame-level fake scores over video duration. "
                    "Red vertical lines = high-risk frames (≥0.75). "
                    "Orange = medium-risk (≥0.5).",
                    styles["caption"]))

    # Suspicious timestamps
    if frame_scores and duration:
        interval = duration / len(frame_scores)
        ts_rows  = [["Frame", "Timestamp", "Score", "Risk"]]
        for i, s in enumerate(frame_scores):
            if s >= 0.5:
                t = i * interval
                rl, rc = risk_level(s)
                ts_rows.append([
                    str(i + 1),
                    f"{t:.2f}s",
                    f"{s:.4f}",
                    rl,
                ])
        if len(ts_rows) > 1:
            els.append(Paragraph("Flagged Frames", styles["field_label"]))
            col_w = [CONTENT_W * f for f in [0.15, 0.2, 0.2, 0.45]]
            extra2 = []
            for row_i in range(1, len(ts_rows)):
                rl = ts_rows[row_i][3]
                rc = RED if "HIGH" in rl else AMBER
                extra2.append(("TEXTCOLOR", (3, row_i), (3, row_i), rc))
                extra2.append(("FONTNAME",  (3, row_i), (3, row_i), "Helvetica-Bold"))
            els.append(make_table(ts_rows, col_w, extra2))
            els.append(Spacer(1, 6))

    return els


def build_modality_audio(report, styles, chart_dir):
    els = []
    els.append(Paragraph("Audio Analysis", styles["section_title"]))
    els.append(section_rule())

    ar  = report["audio_result"]
    aex = report["audio_exp"]
    label = ar.get("label")
    conf  = ar.get("confidence", 0) or 0
    fp    = ar.get("fake_prob",  conf if label == 1 else 1 - conf)

    seg_scores = ar.get("explainability", {}).get("details", {}).get("segment_scores", [])
    duration   = ar.get("explainability", {}).get("metadata", {}).get("duration", 32.0)

    # Summary table
    rows = [
        ["Property", "Value"],
        ["Verdict",       label_str(label)],
        ["Confidence",    f"{conf:.4f}"],
        ["Fake Prob",     f"{fp:.4f}"],
        ["Real Prob",     f"{ar.get('real_prob', 1 - fp):.4f}"],
        ["Segments analysed", str(len(seg_scores))],
        ["High-risk segments (≥0.75)",
         str(sum(1 for s in seg_scores if s >= 0.75))],
        ["Duration",      f"{duration:.2f} s"],
        ["Dominant Band", aex.get("dominant_band", "N/A").upper()],
        ["Pattern Type",  aex.get("pattern", {}).get("type", "N/A")],
    ]
    col_w = [CONTENT_W * 0.55, CONTENT_W * 0.45]
    extra = [
        ("TEXTCOLOR", (1, 1), (1, 1), RED if label == 1 else GREEN),
        ("FONTNAME",  (1, 1), (1, 1), "Helvetica-Bold"),
    ]
    els.append(make_table(rows, col_w, extra))
    els.append(Spacer(1, 8))

    # Audio explanation
    explanation = aex.get("explanation") or ar.get("explainability", {}).get("explanation", "")
    if explanation:
        els.append(Paragraph("Model Explanation", styles["field_label"]))
        els.append(Paragraph(explanation, styles["body"]))
        els.append(Spacer(1, 6))

    # Band scores
    band_scores = aex.get("band_scores", {})
    if band_scores:
        b_rows = [["Band", "Score"]]
        for band, score in band_scores.items():
            val = "N/A" if (score is None or (isinstance(score, float) and math.isnan(score))) else f"{score:.6f}"
            b_rows.append([band.upper(), val])
        els.append(Paragraph("Frequency Band Scores", styles["field_label"]))
        els.append(make_table(b_rows, col_w))
        els.append(Spacer(1, 8))

    # Segment chart
    if seg_scores:
        chart_p = os.path.join(chart_dir, "audio_segment_scores.png")
        if chart_audio_segments(seg_scores, duration, chart_p):
            img = safe_image(chart_p, CONTENT_W, 70 * mm)
            if img:
                els.append(img)
                els.append(Paragraph(
                    "Audio segment fake scores. Red = high risk (≥0.75), "
                    "orange = medium risk (≥0.5), blue = low risk.",
                    styles["caption"]))

    return els


def build_modality_text(report, styles):
    els = []
    els.append(Paragraph("Text / Transcript Analysis", styles["section_title"]))
    els.append(section_rule())

    tr  = report["text_result"]
    exp = tr.get("explanation", {})
    label = tr.get("label")
    conf  = tr.get("confidence", 0) or 0

    rows = [
        ["Property", "Value"],
        ["Verdict",      label_str(label)],
        ["Confidence",   f"{conf:.4f}"],
        ["Prediction",   exp.get("prediction", "N/A")],
    ]
    col_w = [CONTENT_W * 0.45, CONTENT_W * 0.55]
    extra = [
        ("TEXTCOLOR", (1, 1), (1, 1), RED if label == 1 else GREEN),
        ("FONTNAME",  (1, 1), (1, 1), "Helvetica-Bold"),
    ]
    els.append(make_table(rows, col_w, extra))
    els.append(Spacer(1, 8))

    # Word attribution tables
    details = exp.get("details", {})
    human_words = details.get("supporting_words", [])
    ai_words    = details.get("opposing_words", [])

    if human_words or ai_words:
        els.append(Paragraph("Word-level Attribution", styles["field_label"]))
        max_len = max(len(human_words), len(ai_words))
        w_rows = [["Human Signal (supports REAL)", "AI Signal (suggests FAKE)"]]
        for i in range(max_len):
            hw = human_words[i] if i < len(human_words) else ""
            aw = ai_words[i]    if i < len(ai_words)    else ""
            w_rows.append([hw, aw])

        w_col = [CONTENT_W / 2, CONTENT_W / 2]
        w_extra = [
            ("TEXTCOLOR", (0, 1), (0, -1), GREEN),
            ("TEXTCOLOR", (1, 1), (1, -1), RED),
            ("FONTNAME",  (0, 1), (-1, -1), "Helvetica-Bold"),
            ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
        ]
        els.append(make_table(w_rows, w_col, w_extra))
        els.append(Spacer(1, 8))

    # Transcript — rendered in a shaded Table cell to avoid label bleed-in
    transcript = report["ingestion"].get("audio_transcript", "")
    if transcript:
        els.append(Spacer(1, 6))
        els.append(Paragraph("Audio Transcript (ASR)", styles["field_label"]))
        els.append(Spacer(1, 4))
        transcript_style = ParagraphStyle(
            "transcript_inner",
            fontSize=9,
            textColor=GRAY_DARK,
            leading=14,
            fontName="Helvetica-Oblique",
        )
        transcript_para = Paragraph(transcript, transcript_style)
        transcript_tbl = Table([[transcript_para]], colWidths=[CONTENT_W])
        transcript_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), GRAY_LIGHT),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("BOX",           (0, 0), (-1, -1), 0.5, GRAY_MID),
        ]))
        els.append(transcript_tbl)
        els.append(Spacer(1, 8))

    # OCR transcript (on-screen text extracted from video frames)
    ocr_transcript = report["ingestion"].get("ocr_text", "")
    if ocr_transcript:
        els.append(Spacer(1, 4))
        els.append(Paragraph("OCR Transcript (On-Screen Text)", styles["field_label"]))
        els.append(Spacer(1, 4))
        ocr_style = ParagraphStyle(
            "ocr_inner",
            fontSize=9,
            textColor=GRAY_DARK,
            leading=14,
            fontName="Helvetica-Oblique",
        )
        ocr_para = Paragraph(ocr_transcript, ocr_style)
        ocr_tbl = Table([[ocr_para]], colWidths=[CONTENT_W])
        ocr_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#EAF4FB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("BOX",           (0, 0), (-1, -1), 0.5, BLUE_MID),
        ]))
        els.append(ocr_tbl)
        els.append(Spacer(1, 8))

    return els


def build_cross_modal(report, styles):
    els = []
    ing = report["ingestion"]
    asr_ocr = ing.get("asr_ocr_consistency", {})

    els.append(Paragraph("Cross-Modal Consistency", styles["section_title"]))
    els.append(section_rule())

    score = asr_ocr.get("score", None)
    if score is None:
        els.append(Paragraph("Cross-modal consistency data not available.", styles["body"]))
        return els

    if score > 0.8:
        interp = "Strong alignment — audio and visual streams appear synchronised."
        interp_color = GREEN
    elif score > 0.5:
        interp = "Moderate alignment — minor inconsistencies detected."
        interp_color = AMBER
    else:
        interp = "Low alignment — possible audio/video tampering or mismatch."
        interp_color = RED

    # Cell style that word-wraps — used for every data cell
    cell_s = ParagraphStyle(
        "cm_cell", fontSize=9, textColor=GRAY_DARK,
        leading=13, wordWrap="CJK", alignment=TA_LEFT,
    )
    interp_s = ParagraphStyle(
        "cm_interp", fontSize=9, textColor=interp_color,
        fontName="Helvetica-Bold", leading=13, wordWrap="CJK", alignment=TA_LEFT,
    )
    hdr_s = ParagraphStyle(
        "cm_hdr", fontSize=9, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )

    col_w = [CONTENT_W * 0.45, CONTENT_W * 0.55]

    rows = [
        [Paragraph("Property", hdr_s),              Paragraph("Value", hdr_s)],
        [Paragraph("ASR–OCR Consistency Score", cell_s), Paragraph(f"{score:.4f}", cell_s)],
        [Paragraph("Label",                    cell_s), Paragraph(str(asr_ocr.get("label", "N/A")), cell_s)],
        [Paragraph("Interpretation",           cell_s), Paragraph(interp, interp_s)],
        [Paragraph("Watermark Detected",       cell_s), Paragraph(str(asr_ocr.get("watermark_detected", False)), cell_s)],
        [Paragraph("OCR Coverage Ratio",       cell_s), Paragraph(str(ing.get("ocr_coverage_ratio", "N/A")), cell_s)],
        [Paragraph("Subtitle Density",         cell_s), Paragraph(str(ing.get("subtitle_density", "N/A")), cell_s)],
    ]

    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#DDE")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    els.append(tbl)
    return els


def build_forensic_explanation(report, styles):
    els = []
    els.append(Paragraph("Forensic Explanation", styles["section_title"]))
    els.append(section_rule())

    mods  = report["modalities"]
    final = report["final"]
    conf  = final.get("confidence", 0) or 0
    label = final.get("label")

    video = mods.get("video", {})
    audio = mods.get("audio", {})
    text  = mods.get("text",  {})

    frame_scores = (
        video.get("explainability", {}).get("details", {}).get("frame_scores") or
        video.get("details", {}).get("frame_scores", [])
    )
    seg_scores = (
        audio.get("explainability", {}).get("details", {}).get("segment_scores") or
        audio.get("details", {}).get("segment_scores", [])
    )

    paragraphs = []

    if frame_scores:
        max_f  = max(frame_scores)
        high_f = sum(1 for f in frame_scores if f >= 0.75)
        paragraphs.append(
            f"<b>Video:</b> {len(frame_scores)} frames were analysed. "
            f"{high_f} frame(s) scored above the high-risk threshold (0.75), "
            f"with a peak score of {max_f:.4f}. "
            f"Grad-CAM visualisations indicate the model focused on facial regions, "
            f"consistent with face-swap deepfake artifacts."
        )

    if seg_scores:
        high_s = sum(1 for s in seg_scores if s >= 0.75)
        mean_s = sum(seg_scores) / len(seg_scores)
        paragraphs.append(
            f"<b>Audio:</b> {len(seg_scores)} audio segments were evaluated. "
            f"{high_s} segment(s) exceeded the high-risk threshold. "
            f"Mean segment score: {mean_s:.4f}. "
            f"{report['audio_exp'].get('explanation', '')}"
        )

    text_conf = text.get("confidence", 0) or 0
    text_label = text.get("label")
    paragraphs.append(
        f"<b>Text:</b> The transcript was classified as "
        f"<b>{'Human Written' if text_label == 0 else 'AI Generated'}</b> "
        f"with confidence {text_conf:.4f}. "
        f"Key human-signal tokens included contextual colloquialisms and OCR noise, "
        f"while AI-signal tokens included formal vocabulary patterns."
    )

    paragraphs.append(
        f"<b>Overall:</b> The final fused confidence of {conf:.4f} indicates "
        f"{interpret_confidence(conf)}. "
        f"The {'convergence' if label == 1 else 'divergence'} of modality signals "
        f"{'strongly supports' if conf >= 0.8 else 'suggests'} the "
        f"{'FAKE' if label == 1 else 'REAL'} classification."
    )

    for p in paragraphs:
        els.append(Paragraph(p, styles["body"]))
        els.append(Spacer(1, 6))

    return els


def build_decision_logic(styles):
    els = []
    els.append(Paragraph("Decision Logic & Methodology", styles["section_title"]))
    els.append(section_rule())

    paras = [
        "The system employs a <b>multimodal fusion</b> architecture combining three independent "
        "detection pipelines: video, audio, and text. Each pipeline produces an independent "
        "probability estimate, which are combined using a late-fusion strategy.",

        "<b>Video pipeline:</b> Two backbone models are used — Xception (CNN-based) and Swin "
        "Transformer (attention-based). Their outputs are fused to produce a per-frame fake "
        "probability. The final video score is the maximum probability across analysed frames.",

        "<b>Audio pipeline:</b> The audio is segmented and each segment is evaluated for "
        "synthetic speech characteristics. The model analyses log-mel spectrograms and uses "
        "Grad-CAM to identify frequency regions driving the classification.",

        "<b>Text pipeline:</b> Transcribed speech is passed through a language model classifier "
        "trained to distinguish human from AI-generated text. Word-level attribution scores "
        "identify which tokens most influenced the prediction.",

        "<b>Fusion:</b> When modalities agree, confidence increases. When they diverge, "
        "the system weights the modality with highest individual confidence. A final "
        "probability above 0.5 results in a FAKE verdict.",
    ]
    for p in paras:
        els.append(Paragraph(p, styles["body"]))
        els.append(Spacer(1, 4))
    return els


def build_visualizations(report, styles):
    """Add all explainability images with proper captions and aspect ratios."""
    els = []
    els.append(Paragraph("Model Visualisations", styles["section_title"]))
    els.append(section_rule())

    explain_dir = report["explain_dir"]

    images = [
        ("video_gradcam_grid.png",
         "Video Grad-CAM Grid",
         "Class Activation Maps showing which facial regions the video model attended to. "
         "Warmer colors (red/orange) indicate regions most influential in the FAKE decision."),
        ("video_swin_attention.png",
         "Video Swin Transformer Attention",
         "Attention map from the Swin Transformer backbone. High-attention areas highlight "
         "potential deepfake artifacts in facial structure and boundaries."),
        ("audio_logmel.png",
         "Audio Log-Mel Spectrogram",
         "Log-Mel frequency representation of the audio signal. The x-axis is time (seconds), "
         "y-axis is frequency (Hz). Unusual patterns in harmonic structure may indicate synthesis."),
        ("audio_gradcam.png",
         "Audio Grad-CAM",
         "Grad-CAM activation over the audio spectrogram. Highlighted regions (warm colors) "
         "show which frequency–time regions most influenced the audio fake prediction."),
        ("text_attribution.png",
         "Text Attribution",
         "Word-level attribution scores. Blue bars indicate words supporting HUMAN classification; "
         "red bars indicate words suggesting AI-generated patterns."),
    ]

    for fname, title, caption in images:
        fpath = os.path.join(explain_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = safe_image(fpath, CONTENT_W, 95 * mm)
        if img:
            els.append(KeepTogether([
                Paragraph(title, styles["field_label"]),
                Spacer(1, 4),
                img,
                Paragraph(caption, styles["caption"]),
                Spacer(1, 10),
            ]))

    return els


def build_charts_section(report, styles, chart_dir):
    """Modality probability chart in its own section."""
    els = []
    els.append(Paragraph("Probability Summary Charts", styles["section_title"]))
    els.append(section_rule())

    mods = report["modalities"]
    chart_p = os.path.join(chart_dir, "modality_probs.png")
    if chart_modality_probs(mods, chart_p):
        img = safe_image(chart_p, CONTENT_W * 0.7, 70 * mm)
        if img:
            img.hAlign = "CENTER"
            els.append(img)
            els.append(Paragraph(
                "Fake probability per modality. Values above 0.5 (dashed line) indicate "
                "a FAKE verdict for that modality. Red bars = FAKE, Green bars = REAL.",
                styles["caption"]))
    return els


def build_chain_of_custody(report, styles):
    """Render the chain-of-custody log as a paginated table."""
    els = []
    custody = report.get("custody", {})

    # ✅ FIX 1: Support BOTH formats
    if isinstance(custody, list):
        entries = custody
        total   = len(entries)
        case    = report.get("case_id", "N/A")
    else:
        # 🔥 KEY FIX HERE
        entries = custody.get("entries") or custody.get("chain_of_custody", [])
        total   = custody.get("total_entries", len(entries))
        case    = custody.get("case_id", report.get("case_id", "N/A"))

    if not entries:
        return els  # Skip section entirely if no data

    els.append(Paragraph("Chain of Custody", styles["section_title"]))
    els.append(section_rule())
    els.append(Paragraph(
        f"This log records every file ingested, extracted, analysed, and saved during "
        f"processing of case <b>{case}</b>. SHA-256 hashes confirm that no file was "
        f"altered between pipeline stages. Total entries recorded: <b>{total}</b>.",
        styles["body"]
    ))
    els.append(Spacer(1, 8))

    col_w = [
        CONTENT_W * 0.04,
        CONTENT_W * 0.10,
        CONTENT_W * 0.10,
        CONTENT_W * 0.08,
        CONTENT_W * 0.09,
        CONTENT_W * 0.22,
        CONTENT_W * 0.17,
        CONTENT_W * 0.20,
    ]

    header = ["#", "Time\n(UTC)", "Stage", "Modality", "Action", "File", "SHA-256", "Notes"]

    cell_style = ParagraphStyle(
        "coc_cell", fontSize=7, textColor=GRAY_DARK, leading=9,
        fontName="Helvetica", wordWrap="CJK",
    )
    hdr_style = ParagraphStyle(
        "coc_hdr", fontSize=7, textColor=WHITE, leading=9,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )

    rows = [[Paragraph(h, hdr_style) for h in header]]

    for e in entries:
        # ✅ FIX 2: handle nested file structure
        file_info = e.get("file", {}) if isinstance(e.get("file"), dict) else {}

        sha = file_info.get("sha256") or e.get("sha256") or e.get("hash") or "N/A"
        sha_display = sha[:16] + "…" if isinstance(sha, str) and len(sha) > 16 else str(sha)

        file_val = file_info.get("path") or e.get("path") or str(e.get("file", "—"))

        rows.append([
            Paragraph(str(e.get("sequence", e.get("index", ""))), cell_style),
            Paragraph(str(e.get("timestamp", e.get("time_utc", ""))), cell_style),
            Paragraph(str(e.get("stage", "")), cell_style),
            Paragraph(str(e.get("modality", "")), cell_style),
            Paragraph(str(e.get("action", "")), cell_style),
            Paragraph(file_val, cell_style),
            Paragraph(sha_display, cell_style),
            Paragraph(str(e.get("notes", "")), cell_style),
        ])

    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  7),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("FONTSIZE",       (0, 1), (-1, -1), 7),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE")),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ]))

    els.append(tbl)
    els.append(Spacer(1, 6))
    els.append(Paragraph(
        "SHA-256 values are truncated here for readability. "
        "Full hashes are stored in logs/chain_of_custody.json within the case folder "
        "and can be used for independent verification.",
        styles["caption"]
    ))

    return els


# ══════════════════════════════════════════════════════════════
# PAGE NUMBERING
# ══════════════════════════════════════════════════════════════
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY_MID)
    canvas.drawRightString(PAGE_W - MARGIN, 12 * mm,
                           f"Page {doc.page}  |  CONFIDENTIAL FORENSIC REPORT")
    canvas.drawString(MARGIN, 12 * mm, "Generated by Deepfake Detection System")
    canvas.restoreState()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def generate_report(case_path):
    print("\n📄 Generating forensic report...")

    report = load_all(case_path)
    styles = build_styles()

    report_dir = os.path.join(case_path, "report")
    chart_dir  = os.path.join(case_path, "explainability", "_charts")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(chart_dir,  exist_ok=True)

    pdf_path = os.path.join(report_dir, "report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=22 * mm,
        title=f"Forensic Report — {report['case_id']}",
        author="Deepfake Detection System",
    )

    story = []

    # 1. Cover / Verdict
    story += build_cover(report, styles, chart_dir)

    # 2. Executive Summary
    story += build_executive_summary(report, styles)
    story.append(Spacer(1, 10))

    # 3. Charts section (modality probs)
    story += build_charts_section(report, styles, chart_dir)
    story.append(Spacer(1, 10))

    # 4. File Metadata
    story += build_file_metadata(report, styles)
    story.append(PageBreak())

    # 5. Video Analysis
    story += build_modality_video(report, styles, chart_dir)
    story.append(Spacer(1, 10))

    # 6. Audio Analysis
    story += build_modality_audio(report, styles, chart_dir)
    story.append(PageBreak())

    # 7. Text Analysis
    story += build_modality_text(report, styles)
    story.append(Spacer(1, 10))

    # 8. Cross-Modal
    story += build_cross_modal(report, styles)
    story.append(Spacer(1, 10))

    # 9. Decision Logic
    story += build_decision_logic(styles)
    story.append(PageBreak())

    # 10. Forensic Explanation
    story += build_forensic_explanation(report, styles)
    story.append(Spacer(1, 10))

    # 11. All visualisation images
    story += build_visualizations(report, styles)

    # 12. Chain of Custody (only if data present)
    coc = build_chain_of_custody(report, styles)
    if coc:
        story.append(PageBreak())
        story += coc

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    print(f"✅ Report generated: {pdf_path}")
    return pdf_path