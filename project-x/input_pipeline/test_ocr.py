from input_pipeline.ocr import extract_text_from_image

image_path = "20260221_110359.jpg"  # use an image with clear English text
text = extract_text_from_image(image_path)

print("OCR Output:")
print(text)


#  python -m input_pipeline.test_ocr