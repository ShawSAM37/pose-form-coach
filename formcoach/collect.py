"""Collect labeled pose samples from a webcam.

While the preview window is focused, press a class key to save the current
pose as one labeled sample:

    u = Up    d = Down    o = Optimal    s = subOptimal    x = Dangerous
    q = quit

Each exercise gets its own CSV; the filename (without extension) becomes the
exercise name used by train/run.

Usage:
    python -m formcoach.collect --out data/squat.csv
    python -m formcoach.collect --out data/squat.csv --append --camera 1
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from formcoach import pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output CSV path, e.g. data/squat.csv")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV instead of overwriting")
    parser.add_argument("--task-model", default=str(pose.DEFAULT_TASK_PATH), help="Path to PoseLandmarker .task model")
    args = parser.parse_args()

    import cv2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append or not out_path.exists():
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerow(pose.CSV_HEADERS)
        print(f"Created {out_path}")
    else:
        print(f"Appending to {out_path}")

    key_mapping = {ord(k): label for k, label in pose.CLASS_KEY_MAP.items()}
    print("-" * 40)
    for k, label in pose.CLASS_KEY_MAP.items():
        print(f"  Press '{k}' -> {label}")
    print("  Press 'q' to quit.")
    print("-" * 40)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    saved = 0
    with pose.create_landmarker(args.task_model) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pose.detect(landmarker, frame, int(time.time() * 1000))
            pose.draw_landmarks(frame, result)

            cv2.putText(frame, "u:Up d:Down o:Opt s:Sub x:Dang q:quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"saved: {saved}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("formcoach - data collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in key_mapping:
                row = pose.result_to_row(result)
                if row is None:
                    print("No pose detected, sample ignored.")
                else:
                    label = key_mapping[key]
                    with open(out_path, "a", newline="") as f:
                        csv.writer(f).writerow([label] + row)
                    saved += 1
                    print(f"Saved sample #{saved}: {label}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved} samples saved to {out_path}")


if __name__ == "__main__":
    main()
