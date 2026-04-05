import cv2
import numpy as np
import torch
import os
import json
from mtcnn import MTCNN

from models.video.xception.inference import XceptionDetector
from models.video.swin.inference import SwinDetector
from explainability.video_explainer import run_explainability


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VideoPipeline:
    def __init__(self, xception_path, swin_path, case, device=DEVICE):
        self.device = device
        self.case = case

        print("Loading video models...")
        self.xception = XceptionDetector(xception_path, device)
        self.swin = SwinDetector(swin_path, device)

        print("Initializing face detector...")
        self.detector = MTCNN()

        self.frames_dir = case.get_path("frames")
        self.explain_dir = case.get_path("explain")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.explain_dir, exist_ok=True)

    def clear_frames(self):
        for f in os.listdir(self.frames_dir):
            os.remove(os.path.join(self.frames_dir, f))

    def save_frame(self, frame, idx):
        cv2.imwrite(
            os.path.join(self.frames_dir, f"frame_{idx:04d}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

    def extract_face(self, frame, padding=0.3):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb)

            if not detections:
                return None

            valid = [d for d in detections if d['confidence'] > 0.9]
            if len(valid) == 0:
                return None

            best = max(valid, key=lambda d: d['confidence'])
            x, y, w, h = best['box']

            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                return None

            return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"[Warning] Face detection failed: {e}")
            return None

    def extract_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total_frames - 1, 1), num_frames, dtype=int)

        frames = []
        save_idx = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            face = self.extract_face(frame)

            if face is not None:
                frames.append(face)
                self.save_frame(face, save_idx)
                save_idx += 1

        cap.release()
        return frames if len(frames) > 0 else None

    def extract_full_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total_frames - 1, 1), num_frames, dtype=int)

        frames = []
        save_idx = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            self.save_frame(frame, save_idx)
            save_idx += 1

        cap.release()
        return frames if len(frames) > 0 else None

    def analyze(self, video_path):

        print("\n[VideoPipeline] Extracting frames...")
        self.clear_frames()

        # ==========================================
        # CoC — VIDEO FILE RECEIVED FOR ANALYSIS
        # ==========================================
        self.case.log_coc(
            stage="video",
            file_path=video_path,
            modality="video",
            action="analysed",
            notes="Video file submitted to Xception + Swin pipeline for deepfake detection."
        )

        # ==========================================
        # 🔥 GET VIDEO DURATION
        # ==========================================
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        frames = self.extract_frames(video_path)
        mode = "face"

        if frames is None or len(frames) == 0:
            print("[Fallback] Using full frames")
            self.clear_frames()
            frames = self.extract_full_frames(video_path)
            mode = "full_frame"

            if frames is None:
                raise ValueError("No frames extracted")

        if len(frames) < 8:
            while len(frames) < 8:
                frames.append(frames[-1])

        # CoC — frames extracted
        self.case.log_coc(
            stage="video",
            file_path=self.frames_dir,
            modality="frames",
            action="extracted",
            notes=f"Frames extracted for video analysis (mode: {mode}).",
            extra={
                "frame_count": len(frames),
                "mode": mode,
                "duration_seconds": round(duration, 2)
            }
        )

        # ==========================================
        # XCEPTION (FRAME-LEVEL)
        # ==========================================
        x_preds = [self.xception.predict_frame(f) for f in frames]
        x_preds = np.array(x_preds)

        x_real, x_fake = x_preds.mean(axis=0)

        # ==========================================
        # SWIN (VIDEO-LEVEL)
        # ==========================================
        s_real, s_fake = self.swin.predict_video(frames)

        # ==========================================
        # 🔥 COMBINED FRAME SCORES
        # ==========================================
        frame_scores = []

        for i in range(len(x_preds)):
            x_fake_i = x_preds[i][1]
            combined = 0.7 * x_fake_i + 0.3 * s_fake
            frame_scores.append(float(combined))

        # ==========================================
        # FINAL DECISION
        # ==========================================
        final_fake = max(x_fake, s_fake)
        final_real = min(x_real, s_real)

        final_label = 1 if final_fake > final_real else 0
        confidence = float(final_fake if final_label == 1 else final_real)

        # ==========================================
        # EXPLAINABILITY
        # ==========================================
        try:
            explain_result = run_explainability(
                self.frames_dir,
                self.case,
                device=self.device
            )
        except Exception as e:
            print(f"[Warning] Explainability failed: {e}")
            explain_result = {"error": str(e)}

        # ==========================================
        # FINAL RESULT
        # ==========================================
        result = {
            "label": final_label,
            "confidence": float(confidence),

            "metadata": {
                "duration": duration
            },

            "details": {
                "mode": mode,
                "frames_used": len(frames),
                "frame_scores": frame_scores,

                "xception": {
                    "real": float(x_real),
                    "fake": float(x_fake)
                },

                "swin": {
                    "real": float(s_real),
                    "fake": float(s_fake)
                },

                "final_scores": {
                    "real": float(final_real),
                    "fake": float(final_fake)
                }
            },

            "explainability": explain_result
        }

        save_path = os.path.join(self.case.get_path("results"), "video_result.json")

        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)

        # ==========================================
        # CoC — RESULT SAVED
        # ==========================================
        self.case.log_coc(
            stage="video",
            file_path=save_path,
            modality="video",
            action="saved",
            notes="Video detection result saved.",
            extra={
                "label": final_label,
                "confidence": round(confidence, 4),
                "fake_prob": round(final_fake, 4),
                "frames_used": len(frames),
                "mode": mode,
                "duration_seconds": round(duration, 2)
            }
        )

        return result


_video_pipeline_instance = None


def load_video_pipeline(xception_path, swin_path, case):
    global _video_pipeline_instance

    if _video_pipeline_instance is None:
        _video_pipeline_instance = VideoPipeline(
            xception_path,
            swin_path,
            case
        )

    return _video_pipeline_instance


def run_video_pipeline(video_path, pipeline):

    print("Running VIDEO pipeline...")

    return pipeline.analyze(video_path)