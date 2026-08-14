import time
from abc import ABC, abstractmethod


class Exercise(ABC):
    """
    Base class every exercise (BicepCurl, Squat, ShoulderPress, LateralRaise...)
    must implement. main.py / app.py only ever talk to this interface -- they
    never need to know the specific joint-angle logic for a given exercise.
    """

    name = "generic_exercise"

    def __init__(self):
        self.counter = 0
        self.correct_counter = 0
        self.incorrect_counter = 0
        self.stage = None
        self.feedback = ""
        self.log = []  # list of dicts, one per completed rep

    @abstractmethod
    def update(self, landmarks, mp_pose):
        """
        Called every frame with the current MediaPipe landmarks.
        Must update self.stage / self.counter / self.feedback, and call
        self.log_rep(...) whenever a repetition completes.
        """
        raise NotImplementedError

    def log_rep(self, correct: bool, details: dict):
        """Records one completed repetition for the final report."""
        if correct:
            self.correct_counter += 1
        else:
            self.incorrect_counter += 1

        self.log.append({
            "exercise": self.name,
            "rep_number": self.counter,
            "correct": correct,
            "timestamp": time.time(),
            **details,
        })

    def get_state(self):
        """What the GUI reads every frame to display counters/feedback."""
        return {
            "exercise": self.name,
            "counter": self.counter,
            "correct": self.correct_counter,
            "incorrect": self.incorrect_counter,
            "stage": self.stage,
            "feedback": self.feedback,
        }

    def get_form_score(self):
        """Simple 0-100 form score based on correct vs total reps."""
        if self.counter == 0:
            return 100.0
        return round((self.correct_counter / self.counter) * 100, 1)
