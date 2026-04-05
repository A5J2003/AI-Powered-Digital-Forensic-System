from utils.case_manager import CaseManager
from input_pipeline.ingestion import process_video
import shutil
import os


def test_ingestion(video_path):

    case = CaseManager()

    print("\n🚀 Running ingestion pipeline...\n")

    # Copy input into case folder
    input_copy = os.path.join(case.get_path("input"), os.path.basename(video_path))
    shutil.copy(video_path, input_copy)

    result = process_video(input_copy, case)

    print("\n✅ Ingestion complete")
    print("📁 Case folder:", case.base_path)


if __name__ == "__main__":
    video_path = ""  # put test video path here
    test_ingestion(video_path)