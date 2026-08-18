from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class LateralRaise(Exercise):
    name = "lateral_raise"

    DOWN_ANGLE = 25   # arm hanging by your side
    UP_ANGLE = 65     # arm raised out to roughly shoulder height

    ELBOW_STRAIGHT_ANGLE = 130   # elbow angle AT THE TOP must stay above this
                                  # (i.e. fairly straight) -- checked as a single
                                  # snapshot at rep completion, not tracked across
                                  # the whole movement, since a single noisy
                                  # tracking frame mid-raise shouldn't ruin the rep

    OVERRAISE_MARGIN = 0.10      # how far above shoulder height (normalized
                                  # coords) the wrist is allowed to go before
                                  # it's flagged as raised too high

    def __init__(self):
        super().__init__()

        self.sides = {
            "left": self._new_side_state(),
            "right": self._new_side_state(),
        }

        # A rep only counts once BOTH arms have reached the top together --
        # same "both arms" pattern as bicep_curl.py and shoulder_press.py.
        self.both_reached = {"left": False, "right": False}
        self.both_rep_info = {}

    @staticmethod
    def _new_side_state():
        return {
            "stage": None,
            "max_wrist_overraise": 0.0,
        }

    def update(self, landmarks, mp_pose):
        feedback_messages = []

        for side in ["left", "right"]:
            prefix = side.upper()
            shoulder = get_landmark_xy(landmarks, f"{prefix}_SHOULDER", mp_pose)
            elbow = get_landmark_xy(landmarks, f"{prefix}_ELBOW", mp_pose)
            wrist = get_landmark_xy(landmarks, f"{prefix}_WRIST", mp_pose)
            hip = get_landmark_xy(landmarks, f"{prefix}_HIP", mp_pose)

            # Abduction angle: how far the arm has lifted away from the torso.
            raise_angle = calculate_angle(elbow, shoulder, hip)

            # Elbow straightness -- this frame's reading only, used as a
            # snapshot at the top, not accumulated across the whole rep.
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            state = self.sides[side]

            # ---- arm is down: reset tracking for the next rep ----
            if raise_angle < self.DOWN_ANGLE:
                state["stage"] = "down"
                state["max_wrist_overraise"] = 0.0

            # ---- arm is rising: track overraise only ----
            elif state["stage"] == "down":
                # y grows downward, so a positive value here means the wrist
                # has risen above the shoulder line.
                overraise = shoulder[1] - wrist[1]
                if overraise > 0:
                    state["max_wrist_overraise"] = max(state["max_wrist_overraise"], overraise)

            # ---- arm reached the top (raised to roughly shoulder height) ----
            if raise_angle > self.UP_ANGLE and state["stage"] == "down":
                state["stage"] = "up"

                elbow_stable = elbow_angle >= self.ELBOW_STRAIGHT_ANGLE
                height_stable = state["max_wrist_overraise"] <= self.OVERRAISE_MARGIN

                # DEBUG: prints real measured values so you can tune both
                # thresholds for your own camera distance/setup.
                print(f"[{side}] raise_angle={raise_angle:.1f} | elbow_angle_at_top={elbow_angle:.1f} "
                      f"(th={self.ELBOW_STRAIGHT_ANGLE}) | max_overraise={state['max_wrist_overraise']:.3f} "
                      f"(th={self.OVERRAISE_MARGIN}) | {'OK' if (elbow_stable and height_stable) else 'FLAGGED'}")

                self.both_reached[side] = True
                self.both_rep_info[side] = {
                    "elbow_stable": elbow_stable,
                    "height_stable": height_stable,
                    "raise_angle": round(raise_angle, 1),
                    "elbow_angle_at_top": round(elbow_angle, 1),
                    "max_wrist_overraise": round(state["max_wrist_overraise"], 3),
                }

        # ======================================================
        # Only count once BOTH arms have reached the top
        # ======================================================
        if self.both_reached["left"] and self.both_reached["right"]:
            left_info = self.both_rep_info["left"]
            right_info = self.both_rep_info["right"]

            both_stable = (
                left_info["elbow_stable"] and right_info["elbow_stable"]
                and left_info["height_stable"] and right_info["height_stable"]
            )

            mistakes = []
            if not left_info["elbow_stable"]:
                feedback_messages.append("Left elbow: keep your arm straighter")
                mistakes.append("left_elbow_bent")
            if not right_info["elbow_stable"]:
                feedback_messages.append("Right elbow: keep your arm straighter")
                mistakes.append("right_elbow_bent")
            if not left_info["height_stable"]:
                feedback_messages.append("Left arm: don't raise above shoulder height")
                mistakes.append("left_overraise")
            if not right_info["height_stable"]:
                feedback_messages.append("Right arm: don't raise above shoulder height")
                mistakes.append("right_overraise")

            self.counter += 1
            self.stage = "up"
            self.log_rep(both_stable, {
                "side": "both",
                "left_raise_angle": left_info["raise_angle"],
                "right_raise_angle": right_info["raise_angle"],
                "left_elbow_angle_at_top": left_info["elbow_angle_at_top"],
                "right_elbow_angle_at_top": right_info["elbow_angle_at_top"],
                "mistake": mistakes[0] if mistakes else None,
                "all_mistakes": mistakes,
            })

            self.both_reached = {"left": False, "right": False}
            self.both_rep_info = {}

        # Reset if the user stops mid-rep without both arms syncing up.
        left_down = self.sides["left"]["stage"] == "down"
        right_down = self.sides["right"]["stage"] == "down"
        if left_down and right_down:
            self.both_reached = {"left": False, "right": False}
            self.both_rep_info = {}

        self.feedback = " | ".join(feedback_messages) if feedback_messages else "Good form"