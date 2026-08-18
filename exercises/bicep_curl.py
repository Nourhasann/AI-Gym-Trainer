from core.exercise_base import Exercise
from core.angles import calculate_angle, get_landmark_xy


class BicepCurl(Exercise):
    name = "bicep_curl"

    DOWN_ANGLE = 160
    UP_ANGLE = 30
    ELBOW_DRIFT_THRESHOLD = 0.03  # normalized coords (~6% of frame width) of allowed elbow sway

    def __init__(self, arm_selection="both"):
        super().__init__()

        # "left", "right", or "both"
        self.arm_selection = arm_selection

        # Keep separate movement states for each arm
        self.sides = {
            "left": self._new_side_state(),
            "right": self._new_side_state(),
        }

        # ------------------------------------------------------
        # FOR "BOTH" MODE ONLY
        #
        # A rep only counts once BOTH arms have reached the top
        # of the curl (not just one). We track each arm's
        # "reached the top this cycle" flag + its form data here,
        # and only log the rep once both flags are True. The
        # flags reset once both arms return to the DOWN position.
        # ------------------------------------------------------
        self.both_reached = {"left": False, "right": False}
        self.both_rep_info = {}

    @staticmethod
    def _new_side_state():
        return {
            "stage": None,
            "elbow_anchor_x": None,
            "max_drift": 0.0,
        }

    def update(self, landmarks, mp_pose):
        feedback_messages = []

        active_sides = (
            ["left"] if self.arm_selection == "left"
            else ["right"] if self.arm_selection == "right"
            else ["left", "right"]
        )

        for side in active_sides:
            prefix = side.upper()
            shoulder = get_landmark_xy(landmarks, f"{prefix}_SHOULDER", mp_pose)
            elbow = get_landmark_xy(landmarks, f"{prefix}_ELBOW", mp_pose)
            wrist = get_landmark_xy(landmarks, f"{prefix}_WRIST", mp_pose)

            angle = calculate_angle(shoulder, elbow, wrist)
            state = self.sides[side]

            # ---- arm is down: reset drift tracking for the next rep ----
            if angle > self.DOWN_ANGLE:
                state["stage"] = "down"
                state["elbow_anchor_x"] = elbow[0]
                state["max_drift"] = 0.0

            # ---- arm is moving: track elbow sway from the anchor ----
            elif state["stage"] == "down":
                if state["elbow_anchor_x"] is not None:
                    drift = abs(elbow[0] - state["elbow_anchor_x"])
                    state["max_drift"] = max(state["max_drift"], drift)

            # ---- arm reached the top ----
            if angle < self.UP_ANGLE and state["stage"] == "down":
                state["stage"] = "up"

                elbow_stable = state["max_drift"] <= self.ELBOW_DRIFT_THRESHOLD

                # DEBUG: prints the real measured drift so you can tune
                # ELBOW_DRIFT_THRESHOLD for your own camera distance/setup.
                # Remove this print once you're happy with the threshold.
                print(f"[{side}] angle={angle:.1f} | max_drift={state['max_drift']:.3f} | "
                      f"threshold={self.ELBOW_DRIFT_THRESHOLD} | {'OK' if elbow_stable else 'FLAGGED'}")

                if self.arm_selection in ("left", "right"):
                    # Single-arm mode: this arm alone completes the rep.
                    mistake = None if elbow_stable else "elbow_unstable"
                    if not elbow_stable:
                        feedback_messages.append(f"{side.title()} elbow: keep it stable")

                    self.counter += 1
                    self.stage = "up"
                    self.log_rep(elbow_stable, {
                        "side": side,
                        "angle_at_top": round(angle, 1),
                        "max_elbow_drift": round(state["max_drift"], 3),
                        "mistake": mistake,
                    })

                else:
                    # "both" mode: record that this arm reached the top,
                    # but don't count the rep until the OTHER arm also has.
                    self.both_reached[side] = True
                    self.both_rep_info[side] = {
                        "elbow_stable": elbow_stable,
                        "angle_at_top": round(angle, 1),
                        "max_elbow_drift": round(state["max_drift"], 3),
                    }

        # ======================================================
        # "BOTH" MODE: only count once BOTH arms have reached the top
        # ======================================================
        if self.arm_selection == "both" and self.both_reached["left"] and self.both_reached["right"]:
            left_info = self.both_rep_info["left"]
            right_info = self.both_rep_info["right"]

            both_stable = left_info["elbow_stable"] and right_info["elbow_stable"]

            if not left_info["elbow_stable"]:
                feedback_messages.append("Left elbow: keep it stable")
            if not right_info["elbow_stable"]:
                feedback_messages.append("Right elbow: keep it stable")

            self.counter += 1
            self.stage = "up"
            self.log_rep(both_stable, {
                "side": "both",
                "left_angle_at_top": left_info["angle_at_top"],
                "right_angle_at_top": right_info["angle_at_top"],
                "left_max_elbow_drift": left_info["max_elbow_drift"],
                "right_max_elbow_drift": right_info["max_elbow_drift"],
                "mistake": None if both_stable else "elbow_unstable",
            })

            # Reset for the next cycle -- both arms must reach the top again.
            self.both_reached = {"left": False, "right": False}
            self.both_rep_info = {}

        # Also reset the "reached" flags if the user gives up mid-rep and
        # both arms drop back down without ever completing a synced rep,
        # so a stale single-arm flag doesn't carry over incorrectly.
        if self.arm_selection == "both":
            left_down = self.sides["left"]["stage"] == "down"
            right_down = self.sides["right"]["stage"] == "down"
            if left_down and right_down:
                self.both_reached = {"left": False, "right": False}
                self.both_rep_info = {}

        self.feedback = " | ".join(feedback_messages) if feedback_messages else "Good form"