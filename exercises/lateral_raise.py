from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class LateralRaise(Exercise):
    """
    STUB -- not fully implemented yet.
    Follow the BicepCurl pattern: shoulder-hip-elbow angle low (arms down)
    vs near 90 degrees (arms raised to the side), with a check that arms
    stay roughly level with each other.
    """
    name = "lateral_raise"

    def update(self, landmarks, mp_pose):
        self.feedback = "Lateral raise detection not implemented yet"
