from exercises.bicep_curl import BicepCurl
from exercises.squat import Squat
from exercises.shoulder_press import ShoulderPress
from exercises.lateral_raise import LateralRaise

# Maps the string label your classifier predicts (must match the labels
# used when collecting/training data) to the Exercise class to instantiate.
EXERCISES = {
    "bicep_curl": BicepCurl,
    "squat": Squat,
    "shoulder_press": ShoulderPress,
    "lateral_raise": LateralRaise,
}
