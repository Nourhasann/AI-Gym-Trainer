"""
Tkinter GUI for the AI Gym Trainer: Start Workout / Finish Workout buttons,
live camera feed with skeleton overlay, live rep/feedback display, and a
final report shown in a text box after you finish.

Usage:
    python app.py
"""
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
from PIL import Image, ImageTk
import mediapipe as mp

from core.pose_estimator import PoseEstimator
from core.classifier import ExercisePredictor
from exercises import EXERCISES
from core.report import generate_report, save_report

mp_pose = mp.solutions.pose


class GymTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Gym Trainer")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.info_label = tk.Label(root, text="Press Start Workout to begin", font=("Arial", 14), justify="left")
        self.info_label.pack(pady=5)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.start_button = tk.Button(button_frame, text="Start Workout", command=self.start_workout)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Finish Workout", command=self.stop_workout, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.report_box = scrolledtext.ScrolledText(root, width=55, height=12)
        self.report_box.pack(pady=10)

        self.running = False
        self.thread = None

        self.cap = None
        self.estimator = None
        self.predictor = None
        self.active_label = None
        self.active_exercise = None
        self.completed_exercises = []

    def start_workout(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.report_box.delete("1.0", tk.END)

        self.cap = cv2.VideoCapture(0)
        self.estimator = PoseEstimator()
        self.predictor = ExercisePredictor()
        self.active_label = None
        self.active_exercise = None
        self.completed_exercises = []

        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image, landmarks, pose_landmarks_raw = self.estimator.process(frame)

            if landmarks is not None:
                predicted_label = self.predictor.predict(landmarks, mp_pose)

                if predicted_label is not None and predicted_label != self.active_label:
                    if self.active_exercise is not None:
                        self.completed_exercises.append(self.active_exercise)
                    self.active_label = predicted_label
                    exercise_cls = EXERCISES.get(predicted_label)
                    self.active_exercise = exercise_cls() if exercise_cls else None

                if self.active_exercise is not None:
                    try:
                        self.active_exercise.update(landmarks, mp_pose)
                    except Exception as e:
                        print(f"Update error: {e}")

            image = self.estimator.draw(image, pose_landmarks_raw)

            # Push the frame into the Tkinter video label.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update the live info text.
            if self.active_exercise is not None:
                state = self.active_exercise.get_state()
                info = (f"{state['exercise'].upper()}  |  Reps: {state['counter']}  "
                        f"Correct: {state['correct']}  Incorrect: {state['incorrect']}\n"
                        f"{state['feedback']}")
            else:
                info = "Detecting exercise..."
            self.info_label.config(text=info)

            time.sleep(0.01)

    def stop_workout(self):
        self.running = False
        time.sleep(0.2)

        if self.active_exercise is not None:
            self.completed_exercises.append(self.active_exercise)

        if self.cap is not None:
            self.cap.release()
        if self.estimator is not None:
            self.estimator.close()

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        report = generate_report(self.completed_exercises)
        save_report(report)
        self.display_report(report)

    def display_report(self, report):
        self.report_box.delete("1.0", tk.END)
        self.report_box.insert(tk.END, "Workout Report\n")
        self.report_box.insert(tk.END, f"Total reps: {report['total_reps']}\n")
        self.report_box.insert(tk.END, f"Correct: {report['total_correct']}  Incorrect: {report['total_incorrect']}\n")
        self.report_box.insert(tk.END, f"Overall form score: {report['overall_form_score']}%\n\n")
        for ex in report["exercises"]:
            self.report_box.insert(
                tk.END,
                f"- {ex['name']}: {ex['total_reps']} reps "
                f"({ex['correct_reps']} correct / {ex['incorrect_reps']} incorrect), "
                f"form score {ex['form_score']}%\n"
            )
        if report["common_mistakes"]:
            self.report_box.insert(tk.END, "\nCommon mistakes:\n")
            for mistake, count in report["common_mistakes"].items():
                self.report_box.insert(tk.END, f"- {mistake}: {count}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GymTrainerApp(root)
    root.mainloop()
