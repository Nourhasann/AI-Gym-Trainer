from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class BicepCurl(Exercise):
    name = "bicep_curl"

    DOWN_ANGLE = 160          # arm considered fully extended
    UP_ANGLE = 30             # arm considered fully curled
    ELBOW_DRIFT_THRESHOLD = 0.06  # normalized coords (~6% of frame width) of allowed elbow sway

    def __init__(self):
        super().__init__()
        # Track left/right arms independently, but report through one shared counter.
        self.sides = {
            "left": self._new_side_state(),
            "right": self._new_side_state(),
        }

    @staticmethod
    def _new_side_state():
        return {
            "stage": None,
            "elbow_anchor_x": None,
            "max_drift": 0.0,
        }

    def update(self, landmarks, mp_pose):
        feedback_messages = []

        for side in ("left", "right"):
            prefix = side.upper()
            shoulder = get_landmark_xy(landmarks, f"{prefix}_SHOULDER", mp_pose)
            elbow = get_landmark_xy(landmarks, f"{prefix}_ELBOW", mp_pose)
            wrist = get_landmark_xy(landmarks, f"{prefix}_WRIST", mp_pose)

            angle = calculate_angle(shoulder, elbow, wrist)
            state = self.sides[side]

            if angle > self.DOWN_ANGLE:
                state["stage"] = "down"
                # Reset the elbow anchor at the bottom of each rep so drift
                # is measured fresh for the upcoming rep.
                state["elbow_anchor_x"] = elbow[0]
                state["max_drift"] = 0.0

            elif state["stage"] == "down":
                # Mid-rep: track how far the elbow strays horizontally
                # from where it was at the bottom of the movement.
                if state["elbow_anchor_x"] is not None:
                    drift = abs(elbow[0] - state["elbow_anchor_x"])
                    state["max_drift"] = max(state["max_drift"], drift)

            if angle < self.UP_ANGLE and state["stage"] == "down":
                state["stage"] = "up"
                self.counter += 1
                self.stage = "up"

                elbow_stable = state["max_drift"] <= self.ELBOW_DRIFT_THRESHOLD
                correct = elbow_stable
                mistake = None if elbow_stable else "elbow_unstable"

                if not elbow_stable:
                    feedback_messages.append(f"{side.title()} elbow: keep it stable")

                self.log_rep(correct, {
                    "side": side,
                    "angle_at_top": round(angle, 1),
                    "max_elbow_drift": round(state["max_drift"], 3),
                    "mistake": mistake,
                })

        self.feedback = " | ".join(feedback_messages) if feedback_messages else "Good form"
