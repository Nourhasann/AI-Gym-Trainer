import numpy as np


def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at point b, formed by points a-b-c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_landmark_xy(landmarks, part, mp_pose):
    """Extract [x, y] normalized coordinates for a given landmark name, e.g. 'LEFT_ELBOW'."""
    lm = landmarks[mp_pose.PoseLandmark[part].value]
    return [lm.x, lm.y]


def landmark_distance(a, b):
    """Euclidean distance between two [x, y] points."""
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))
