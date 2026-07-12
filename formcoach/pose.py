"""Shared MediaPipe pose utilities.

Every stage of the pipeline represents a body pose as a flat vector of
33 landmarks x (x, y, z) = 99 features, in MediaPipe landmark order.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

NUM_LANDMARKS = 33

# Official Google-hosted PoseLandmarker model (heavy variant, float16).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
DEFAULT_TASK_PATH = Path("models") / "pose_landmarker_heavy.task"

# Form-quality classes and the keys used to label them during collection.
CLASS_KEY_MAP = {
    "u": "Up",
    "d": "Down",
    "o": "Optimal",
    "s": "subOptimal",
    "x": "Dangerous",
}

FEATURE_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

CSV_HEADERS = ["label"] + FEATURE_COLUMNS


def ensure_task_model(path: str | Path = DEFAULT_TASK_PATH) -> Path:
    """Return the path to the PoseLandmarker .task model, downloading it if missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PoseLandmarker model to {path} (~30 MB, one-time)...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete.")
    return path


def create_landmarker(task_path: str | Path = DEFAULT_TASK_PATH):
    """Create a synchronous (VIDEO-mode) MediaPipe PoseLandmarker.

    Use as a context manager:
        with create_landmarker() as landmarker: ...
    """
    import mediapipe as mp

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ensure_task_model(task_path))
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms: int):
    """Run pose detection on a BGR OpenCV frame."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def result_to_row(result) -> list[float] | None:
    """Flatten the first detected pose into a 99-value feature row, or None."""
    if not result.pose_landmarks:
        return None
    row: list[float] = []
    for lm in result.pose_landmarks[0]:
        row.extend([lm.x, lm.y, lm.z])
    return row


def draw_landmarks(frame_bgr, result) -> None:
    """Draw detected landmarks as green dots on the frame (in place)."""
    import cv2

    if not result.pose_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for lm in result.pose_landmarks[0]:
        cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
