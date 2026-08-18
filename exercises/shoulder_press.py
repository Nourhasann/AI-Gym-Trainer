from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy






class ShoulderPress(Exercise):
    name = "shoulder_press"

    DOWN_ANGLE = 100  # elbow bent, wrist ~shoulder height (start position)
    UP_ANGLE = 130    # arm extended overhead (top of the press)
    WRIST_DROP_THRESHOLD = 0.15  # normalized coords -- confirmed working from your data

    MIN_WRIST_HEIGHT_ABOVE_SHOULDER = 0.05  # wrist must be at least this much higher
                                             # (smaller y) than the shoulder to count as
                                             # "overhead" -- this is what stops a sideways
                                             # arm stretch (elbow also straight) from being
                                             # mistaken for a completed press

    def __init__(self):
        super().__init__()

        # Keep separate movement states for each arm (always tracked together,
        # unlike bicep curl there's no left/right/both selection here).
        self.sides = {
            "left": self._new_side_state(),
            "right": self._new_side_state(),
        }

        # ------------------------------------------------------
        # A rep only counts once BOTH arms have reached the top
        # of the press (not just one). Same "both arms" pattern as
        # bicep_curl.py's "both" mode.
        # ------------------------------------------------------
        self.both_reached = {"left": False, "right": False}
        self.both_rep_info = {}

    @staticmethod
    def _new_side_state():
        return {
            "stage": None,
            "max_wrist_drop": 0.0,
        }

    def update(self, landmarks, mp_pose):
        feedback_messages = []

        for side in ["left", "right"]:
            prefix = side.upper()
            shoulder = get_landmark_xy(landmarks, f"{prefix}_SHOULDER", mp_pose)
            elbow = get_landmark_xy(landmarks, f"{prefix}_ELBOW", mp_pose)
            wrist = get_landmark_xy(landmarks, f"{prefix}_WRIST", mp_pose)

            angle = calculate_angle(shoulder, elbow, wrist)
            state = self.sides[side]

            # ---- arm is down (bent, start position): reset drop tracking ----
            if angle < self.DOWN_ANGLE:
                state["stage"] = "down"
                state["max_wrist_drop"] = 0.0

            # ---- arm is pressing up: track how far the wrist sags below the elbow ----
            elif state["stage"] == "down":
                # y grows downward in normalized image coords, so a positive
                # value here means the wrist is below (lower than) the elbow.
                drop = wrist[1] - elbow[1]
                if drop > 0:
                    state["max_wrist_drop"] = max(state["max_wrist_drop"], drop)

            # ---- arm reached the top (fully extended AND actually overhead) ----
            # y grows downward, so "wrist above shoulder" means wrist[1] is
            # SMALLER than shoulder[1] -- this is what rejects a sideways
            # T-pose stretch, since there the wrist stays level with the
            # shoulder instead of rising above it.
            wrist_overhead = wrist[1] < (shoulder[1] - self.MIN_WRIST_HEIGHT_ABOVE_SHOULDER)

            if angle > self.UP_ANGLE and state["stage"] == "down" and wrist_overhead:
                state["stage"] = "up"

                wrist_stable = state["max_wrist_drop"] <= self.WRIST_DROP_THRESHOLD

                # DEBUG: prints the real measured drop so you can tune
                # WRIST_DROP_THRESHOLD for your own camera distance/setup.
                print(f"[{side}] angle={angle:.1f} | max_wrist_drop={state['max_wrist_drop']:.3f} | "
                      f"threshold={self.WRIST_DROP_THRESHOLD} | wrist_overhead={wrist_overhead} | "
                      f"{'OK' if wrist_stable else 'FLAGGED'}")

                # Record that this arm reached the top, but don't count the
                # rep until the OTHER arm also has.
                self.both_reached[side] = True
                self.both_rep_info[side] = {
                    "wrist_stable": wrist_stable,
                    "angle_at_top": round(angle, 1),
                    "max_wrist_drop": round(state["max_wrist_drop"], 3),
                }

        # ======================================================
        # Only count once BOTH arms have reached the top
        # ======================================================
        if self.both_reached["left"] and self.both_reached["right"]:
            left_info = self.both_rep_info["left"]
            right_info = self.both_rep_info["right"]

            both_stable = left_info["wrist_stable"] and right_info["wrist_stable"]

            if not left_info["wrist_stable"]:
                feedback_messages.append("Left wrist: press straight up, don't let it drop")
            if not right_info["wrist_stable"]:
                feedback_messages.append("Right wrist: press straight up, don't let it drop")

            self.counter += 1
            self.stage = "up"
            self.log_rep(both_stable, {
                "side": "both",
                "left_angle_at_top": left_info["angle_at_top"],
                "right_angle_at_top": right_info["angle_at_top"],
                "left_max_wrist_drop": left_info["max_wrist_drop"],
                "right_max_wrist_drop": right_info["max_wrist_drop"],
                "mistake": None if both_stable else "wrist_drop",
            })

            # Reset for the next cycle -- both arms must reach the top again.
            self.both_reached = {"left": False, "right": False}
            self.both_rep_info = {}

        # Also reset the "reached" flags if the user gives up mid-rep and
        # both arms drop back down without ever completing a synced rep,
        # so a stale single-arm flag doesn't carry over incorrectly.
        left_down = self.sides["left"]["stage"] == "down"
        right_down = self.sides["right"]["stage"] == "down"
        if left_down and right_down:
            self.both_reached = {"left": False, "right": False}
            self.both_rep_info = {}

        self.feedback = " | ".join(feedback_messages) if feedback_messages else "Good form"