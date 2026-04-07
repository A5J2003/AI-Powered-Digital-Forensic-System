import easyocr
import cv2
import re
import torch

# -------------------------------------------------
# 🔹 Automatically detect GPU
# -------------------------------------------------
USE_GPU = torch.cuda.is_available()

print(f"[OCR] Using GPU: {USE_GPU}")

# Initialize reader ONCE (important)
reader = easyocr.Reader(['en'], gpu=USE_GPU)


# -------------------------------------------------
# 🔹 Text Cleaning
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------------------------------------------------
# 🔹 OCR Extraction
# -------------------------------------------------
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return "", 0.0

    # Convert BGR → RGB (EasyOCR expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run EasyOCR
    results = reader.readtext(img_rgb)

    extracted_texts = []
    confidence_scores = []

    for (bbox, text, confidence) in results:
        if confidence > 0.4:  # filter weak detections
            extracted_texts.append(text)
            confidence_scores.append(confidence)

    if not extracted_texts:
        return "", 0.0

    combined_text = " ".join(extracted_texts)
    cleaned = clean_text(combined_text)

    # Average confidence as quality score
    avg_conf = sum(confidence_scores) / len(confidence_scores)

    return cleaned, round(float(avg_conf), 2)