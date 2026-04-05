import re


def read_subtitle_file(srt_path):
    if not srt_path:
        return ""

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Remove timestamps and numbers
    content = re.sub(r'\d+\n', '', content)
    content = re.sub(r'\d{2}:\d{2}:\d{2}.*?\n', '', content)

    return content.strip()