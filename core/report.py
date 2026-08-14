import json
import time
from collections import defaultdict


def generate_report(exercises_completed):
    """
    exercises_completed: list of Exercise instances (already finished).
    Returns a plain dict summarizing the whole workout session.
    """
    report = {
        "generated_at": time.time(),
        "total_exercises": len(exercises_completed),
        "total_reps": 0,
        "total_correct": 0,
        "total_incorrect": 0,
        "exercises": [],
    }

    mistake_counts = defaultdict(int)

    for ex in exercises_completed:
        state = ex.get_state()
        report["total_reps"] += state["counter"]
        report["total_correct"] += state["correct"]
        report["total_incorrect"] += state["incorrect"]

        for rep in ex.log:
            if not rep.get("correct", True) and rep.get("mistake"):
                mistake_counts[rep["mistake"]] += 1

        report["exercises"].append({
            "name": state["exercise"],
            "total_reps": state["counter"],
            "correct_reps": state["correct"],
            "incorrect_reps": state["incorrect"],
            "form_score": ex.get_form_score(),
        })

    if report["total_reps"] > 0:
        report["overall_form_score"] = round(
            (report["total_correct"] / report["total_reps"]) * 100, 1
        )
    else:
        report["overall_form_score"] = 0.0

    report["common_mistakes"] = dict(mistake_counts)

    return report


def save_report(report, path="workout_report.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
