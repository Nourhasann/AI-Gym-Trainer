"""
Collects labeled joint-angle data for training the exercise classifier.

Usage:
    python collect_data.py --label bicep_curl
    python collect_data.py --label squat --out_dir data/raw

While running: press 'r' to toggle recording on/off, 'q' to save and quit.
Only frames captured while "RECORDING" is on get saved -- this lets you get
into position before you start capturing samples.
"""
import argparse
import csv
import os
from datetime import datetime

import cv2
import mediapipe as mp

from core.pose_estimator import PoseEstimator
from core.classifier import extract_features, FEATURE_NAMES

mp_pose = mp.solutions.pose


def main():
    parser = argparse.ArgumentParser(description="Collect labeled pose data for exercise classification.")
    parser.add_argument("--label", required=True, help="Exercise label for this session, e.g. bicep_curl")
    parser.add_argument("--out_dir", default="data/raw", help="Where to save the CSV")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"{args.label}_{timestamp}.csv")

    cap = cv2.VideoCapture(args.camera)
    estimator = PoseEstimator()

    rows = []
    recording = False

    print("Press 'r' to toggle recording, 'q' to save and quit.")

    # try/finally: if you interrupt this from a Jupyter cell instead of
    # pressing 'q', the camera still gets released properly -- otherwise
    # it can stay locked and break the next cell that tries to open it.
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, landmarks, pose_landmarks_raw = estimator.process(frame)
            image = estimator.draw(image, pose_landmarks_raw)

            if landmarks is not None:
                features = extract_features(landmarks, mp_pose)
                if features is not None and recording:
                    rows.append(features + [args.label])

            status_text = "RECORDING" if recording else "PAUSED"
            status_color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(image, f"{status_text} | label={args.label} | samples={len(rows)}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

            cv2.imshow("Collect Data", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('r'):
                recording = not recording
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES + ["label"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} samples to {out_path}")
    else:
        print("No samples recorded, nothing saved.")


if __name__ == "__main__":
    main()
