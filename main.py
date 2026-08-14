"""
Console version of the workout loop (no GUI) -- opens the webcam, auto-detects
the exercise being performed, tracks reps/form, and saves a report when you
press 'q'.

Usage:
    python main.py
"""
import cv2
import mediapipe as mp

from core.pose_estimator import PoseEstimator
from core.classifier import ExercisePredictor
from exercises import EXERCISES
from core.report import generate_report, save_report

mp_pose = mp.solutions.pose


def draw_hud(image, state):
    cv2.rectangle(image, (0, 0), (320, 110), (245, 117, 16), -1)

    cv2.putText(image, state.get("exercise", "detecting...").upper(), (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, f"Reps: {state.get('counter', 0)}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, f"Correct: {state.get('correct', 0)}  Incorrect: {state.get('incorrect', 0)}",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(image, state.get("feedback", ""), (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def run_workout(camera_index=0, model_path="models/exercise_classifier.pkl", use_classifier=True, forced_exercise=None):
    """
    use_classifier=False lets you test a single exercise before you've
    collected any data or trained a model -- pass forced_exercise as one of
    the keys in exercises.EXERCISES (e.g. "bicep_curl") to skip auto-detection
    entirely.
    """
    cap = cv2.VideoCapture(camera_index)
    estimator = PoseEstimator()
    predictor = ExercisePredictor(model_path=model_path) if use_classifier else None

    active_label = None
    active_exercise = None
    completed_exercises = []

    if not use_classifier:
        active_label = forced_exercise
        exercise_cls = EXERCISES.get(forced_exercise)
        active_exercise = exercise_cls() if exercise_cls else None

    # try/finally matters most when running this from Jupyter: if you hit
    # the notebook's "interrupt" (stop) button instead of pressing 'q',
    # this still guarantees the camera gets released and the window closes.
    # Without it, an interrupted cell can leave the webcam locked so the
    # next cell that opens it will hang or fail.
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, landmarks, pose_landmarks_raw = estimator.process(frame)

            if landmarks is not None:
                if use_classifier:
                    predicted_label = predictor.predict(landmarks, mp_pose)

                    # Switch active exercise only when the stable (majority-vote) prediction changes.
                    if predicted_label is not None and predicted_label != active_label:
                        if active_exercise is not None:
                            completed_exercises.append(active_exercise)

                        active_label = predicted_label
                        exercise_cls = EXERCISES.get(active_label)
                        active_exercise = exercise_cls() if exercise_cls else None

                if active_exercise is not None:
                    try:
                        active_exercise.update(landmarks, mp_pose)
                    except Exception as e:
                        # Don't let one bad frame crash the whole session.
                        print(f"Update error: {e}")

            image = estimator.draw(image, pose_landmarks_raw)

            state = active_exercise.get_state() if active_exercise else {
                "exercise": "detecting...", "counter": 0, "correct": 0, "incorrect": 0, "feedback": ""
            }
            draw_hud(image, state)

            cv2.imshow("AI Gym Trainer", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        if active_exercise is not None and active_exercise not in completed_exercises:
            completed_exercises.append(active_exercise)

        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    report = generate_report(completed_exercises)
    save_report(report)
    print("Workout finished. Report saved to workout_report.json")
    return report


if __name__ == "__main__":
    # TEMPORARY: forced to bicep_curl with no classifier, since no model has
    # been trained yet. Once you've run collect_data.py + train_model.py,
    # switch this back to just: run_workout()
    run_workout(use_classifier=False, forced_exercise="bicep_curl")
