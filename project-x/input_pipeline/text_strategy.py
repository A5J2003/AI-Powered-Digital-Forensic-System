def select_text_strategy(transcript,
                         ocr_text,
                         similarity_score,
                         ocr_quality,
                         coverage_ratio):
    """
    Decide how to use ASR and OCR for text AI detection.
    """

    # No speech detected
    if not transcript and ocr_text:
        return {
            "strategy": "OCR_PRIMARY",
            "cross_modal_flag": False
        }

    # Speech exists but no OCR
    if transcript and not ocr_text:
        return {
            "strategy": "ASR_ONLY",
            "cross_modal_flag": False
        }

    # Both exist
    if similarity_score is not None:

        # High similarity → normal subtitle video
        if similarity_score >= 0.6 and coverage_ratio >= 0.25 and ocr_quality >= 0.7:
            return {
                "strategy": "ASR_ONLY",
                "cross_modal_flag": False
            }

        # Medium similarity → partial mismatch
        if 0.3 <= similarity_score < 0.6:
            return {
                "strategy": "DUAL_FUSION",
                "cross_modal_flag": True
            }

        # Low similarity → suspicious
        if similarity_score < 0.3:
            return {
                "strategy": "DUAL_FUSION",
                "cross_modal_flag": True
            }

    # Default fallback
    return {
        "strategy": "ASR_ONLY",
        "cross_modal_flag": False
    }