import os
import subprocess


# ==========================================
# 🔊 AUDIO EXTRACTION
# ==========================================
def extract_audio(video_path, output_dir):
    """
    Extracts audio track from video and saves as WAV.
    Returns path to extracted audio or None if failed.
    """

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.wav")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        output_path,
        "-y"
    ]

    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        print("❌ Audio extraction failed.")
        return None

    return output_path


# ==========================================
# 🎞 FRAME EXTRACTION
# ==========================================
def extract_frames(
    video_path,
    output_dir,
    mode="fps",
    fps=1
):
    """
    Extracts frames from video.

    mode="fps"   → Extract frames at fixed FPS (default).
    mode="scene" → Extract frames only when scene changes.

    Returns directory containing extracted frames or None if failed.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Choose extraction strategy
    if mode == "fps":
        filter_arg = f"fps={fps}"
    elif mode == "scene":
        filter_arg = "select=gt(scene\\,0.4)"
    else:
        filter_arg = f"fps={fps}"

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", filter_arg,
        os.path.join(output_dir, "frame_%04d.png"),
        "-vsync", "vfr",
        "-y"
    ]

    result = subprocess.run(command, capture_output=True)

    if result.returncode != 0:
        print("❌ Frame extraction failed.")
        return None

    return output_dir


# ==========================================
# 💬 SUBTITLE EXTRACTION
# ==========================================
def extract_subtitles(video_path, output_dir):
    """
    Attempts to extract embedded subtitle stream.
    Returns path to .srt file if found, otherwise None.
    """

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.srt")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-map", "0:s:0",
        output_path,
        "-y"
    ]

    result = subprocess.run(command, capture_output=True)

    # If no subtitle stream exists, ffmpeg will fail — that's okay
    if result.returncode != 0 or not os.path.exists(output_path):
        return None

    return output_path