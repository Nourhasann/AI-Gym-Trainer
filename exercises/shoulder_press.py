from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class ShoulderPress(Exercise):
    """
    STUB -- not fully implemented yet.
    Follow the BicepCurl pattern: elbow angle at shoulder height (start) vs
    arm extended overhead (top), with a check that wrists stay above elbows.
    """
    name = "shoulder_press"

    def update(self, landmarks, mp_pose):
        self.feedback = "Shoulder press detection not implemented yet"
