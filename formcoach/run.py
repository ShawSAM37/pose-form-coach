"""Real-time form feedback: webcam -> pose -> classifier -> Arduino.

Streams webcam frames through MediaPipe pose estimation, classifies form
quality with the trained model for the active exercise, and sends the
predicted class over serial to the Arduino, which drives the RGB LED,
buzzer, and servo.

Serial protocol: lowercase class name + newline (e.g. "dangerous\\n"),
sent only when the prediction changes and confidence >= --threshold.

Keys: 1-9 switch exercise, q quits.

Usage:
    python -m formcoach.run --port COM7
    python -m formcoach.run --no-serial            # software-only, no Arduino
    python -m formcoach.run --camera 1 --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from formcoach import pose


def open_serial(port: str, baud: int):
    import serial

    print(f"Connecting to Arduino on {port}...")
    connection = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # opening the port reboots the Arduino; let it come up
    print("Arduino connected.")
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="models/exercise_models.pkl", help="Trained model bundle from train.py")
    parser.add_argument("--features", default="models/feature_columns.json", help="Feature column list from train.py")
    parser.add_argument("--task-model", default=str(pose.DEFAULT_TASK_PATH), help="PoseLandmarker .task model path")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--port", default="COM7", help="Arduino serial port (default COM7)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum prediction confidence to actuate hardware (default 0.6)")
    parser.add_argument("--no-serial", action="store_true", help="Run without an Arduino connected")
    args = parser.parse_args()

    import cv2

    master_payload = joblib.load(args.models)
    exercises = list(master_payload.keys())
    if not exercises:
        raise SystemExit(f"No exercises found in {args.models}. Run `python -m formcoach.train` first.")

    with open(args.features) as f:
        feature_columns = json.load(f)
    expected_len = len(feature_columns)

    current_exercise = exercises[0]
    print(f"Exercises: {exercises} (keys 1-{len(exercises)} to switch). Active: {current_exercise}")

    arduino = None
    if not args.no_serial:
        try:
            arduino = open_serial(args.port, args.baud)
        except Exception as e:  # noqa: BLE001 - any serial failure degrades to software-only
            print(f"Serial connection failed ({e}). Continuing without hardware output.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    last_sent = None
    with pose.create_landmarker(args.task_model) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = pose.detect(landmarker, frame, int(time.time() * 1000))
            prediction, confidence = "No pose detected", 0.0

            row = pose.result_to_row(result)
            if row is not None and len(row) == expected_len:
                model = master_payload[current_exercise]["model"]
                probs = model.predict_proba(np.array(row).reshape(1, -1))[0]
                idx = int(np.argmax(probs))
                prediction = str(model.classes_[idx])
                confidence = float(probs[idx])

                # Actuate only on confident, changed predictions to avoid
                # flooding the serial link and chattering the hardware.
                if arduino is not None and confidence >= args.threshold and prediction != last_sent:
                    try:
                        arduino.write(f"{prediction.lower()}\n".encode("utf-8"))
                        last_sent = prediction
                        print(f"-> Arduino: {prediction.lower()} ({confidence:.2f})")
                    except Exception as e:  # noqa: BLE001
                        print(f"Serial write failed: {e}")

                pose.draw_landmarks(frame, result)

            cv2.rectangle(frame, (0, 0), (420, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"EXERCISE: {current_exercise}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Form: {prediction}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "1-9: switch exercise | q: quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("formcoach - live form feedback", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("1") <= key <= ord("9"):
                index = key - ord("1")
                if index < len(exercises):
                    current_exercise = exercises[index]
                    last_sent = None  # re-send state for the new exercise context
                    print(f"Switched to: {current_exercise}")

    cap.release()
    cv2.destroyAllWindows()
    if arduino is not None:
        arduino.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
