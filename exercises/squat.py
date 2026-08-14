from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class Squat(Exercise):
    """
    STUB -- not fully implemented yet.
    Follow the BicepCurl pattern once you're ready to build this out:
      1. Compute hip-knee-ankle angle (and knee-hip-shoulder for torso lean).
      2. down/up thresholds based on knee angle (e.g. > 160 = standing, < 100 = squatting).
      3. Form checks: knees not caving inward, torso not leaning too far forward.
    """
    name = "squat"

    def update(self, landmarks, mp_pose):
        self.feedback = "Squat detection not implemented yet"
