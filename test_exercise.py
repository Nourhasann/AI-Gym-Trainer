"""
Standalone test harness for a single exercise -- bypasses the classifier
entirely, so you can verify an exercise's rep-counting and form-checking
logic works correctly even before it has any training data collected.

Usage:
    python test_exercise.py --exercise lateral_raise
    python test_exercise.py --exercise bicep_curl --arm left

Press 'q' to quit.
"""
import argparse

import cv2
import mediapipe as mp

from core.pose_estimator import PoseEstimator
from exercises import EXERCISES

mp_pose = mp.solutions.pose


def draw_hud(image, state):
    cv2.rectangle(image, (0, 0), (340, 110), (245, 117, 16), -1)

    cv2.putText(image, state.get("exercise", "?").upper(), (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, f"Reps: {state.get('counter', 0)}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, f"Correct: {state.get('correct', 0)}  Incorrect: {state.get('incorrect', 0)}",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, state.get("feedback", ""), (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Test a single exercise without the classifier.")
    parser.add_argument("--exercise", required=True, choices=list(EXERCISES.keys()),
                         help="Which exercise to force, e.g. lateral_raise")
    parser.add_argument("--arm", default="both", choices=["left", "right", "both"],
                         help="Only used for bicep_curl")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    exercise_cls = EXERCISES[args.exercise]
    if args.exercise == "bicep_curl":
        exercise = exercise_cls(arm_selection=args.arm)
    else:
        exercise = exercise_cls()

    cap = cv2.VideoCapture(args.camera)
    estimator = PoseEstimator()

    print(f"Testing '{args.exercise}'. Watch this terminal for debug prints from the exercise itself.")
    print("Press 'q' in the video window to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, landmarks, pose_landmarks_raw = estimator.process(frame)

            if landmarks is not None:
                try:
                    exercise.update(landmarks, mp_pose)
                except Exception as e:
                    print(f"Update error: {e}")

            image = estimator.draw(image, pose_landmarks_raw)
            draw_hud(image, exercise.get_state())

            cv2.imshow(f"Testing: {args.exercise}", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    state = exercise.get_state()
    print("\n--- Final state ---")
    print(f"Reps: {state['counter']}  Correct: {state['correct']}  Incorrect: {state['incorrect']}")


if __name__ == "__main__":
    main()