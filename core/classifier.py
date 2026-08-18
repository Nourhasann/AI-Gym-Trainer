import pickle
import numpy as np
import pandas as pd

from core.angles import calculate_angle, get_landmark_xy

# The exact order of angles that make up a "feature vector".
# collect_data.py, train_model.py, and live prediction must all agree on this
# order -- if you change it here, you must re-collect data and retrain.
FEATURE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),      # left elbow angle
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),   # right elbow angle
    ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),        # left shoulder angle
    ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),     # right shoulder angle
    ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),         # left hip angle
    ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),      # right hip angle
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),            # left knee angle
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),         # right knee angle
]

FEATURE_NAMES = [
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]


def extract_features(landmarks, mp_pose):
    """
    Turns raw MediaPipe landmarks into a fixed-length feature vector of
    joint angles. Returns a python list of floats, or None if landmarks
    are missing/incomplete.
    """
    if landmarks is None:
        return None

    try:
        features = []
        for a_name, b_name, c_name in FEATURE_JOINTS:
            a = get_landmark_xy(landmarks, a_name, mp_pose)
            b = get_landmark_xy(landmarks, b_name, mp_pose)
            c = get_landmark_xy(landmarks, c_name, mp_pose)
            features.append(calculate_angle(a, b, c))
        return features
    except (IndexError, KeyError, AttributeError):
        return None


def load_model(path="models/exercise_classifier.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


class ExercisePredictor:
    """
    Wraps the trained model plus a small rolling-vote buffer so the
    predicted exercise doesn't flicker frame-to-frame.
    """

    def __init__(self, model_path="models/exercise_classifier.pkl", buffer_size=25, min_confidence_votes=0.75):
        self.model = load_model(model_path)
        self.buffer = []
        self.buffer_size = buffer_size
        self.min_confidence_votes = min_confidence_votes
        self.current_label = None

    def predict(self, landmarks, mp_pose):
        features = extract_features(landmarks, mp_pose)
        if features is None:
            return self.current_label

        # Wrap in a DataFrame with the same column names used in
        # train_model.py -- passing a plain list triggers a sklearn
        # warning on every single frame because the model was fitted
        # on named columns, not raw arrays.
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        prediction = self.model.predict(features_df)[0]
        self.buffer.append(prediction)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Only switch the active exercise if there's a clear majority vote
        # in the recent buffer -- prevents flicker between predictions.
        if len(self.buffer) == self.buffer_size:
            values, counts = np.unique(self.buffer, return_counts=True)
            top_label = values[np.argmax(counts)]
            top_ratio = counts.max() / self.buffer_size
            if top_ratio >= self.min_confidence_votes:
                self.current_label = top_label

        return self.current_label