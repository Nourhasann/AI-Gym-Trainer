import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PoseEstimator:
    """Wraps MediaPipe Pose so the rest of the app never touches MediaPipe directly."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Takes a BGR frame (as read by cv2.VideoCapture), runs pose detection,
        and returns (image_bgr, landmarks_or_none, pose_landmarks_raw).

        pose_landmarks_raw is the raw MediaPipe object, kept around so it can
        be passed to self.draw() for visualization.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        landmarks = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        return image_bgr, landmarks, results.pose_landmarks

    def draw(self, image, pose_landmarks_raw):
        """Draws the skeleton overlay on the given image, in place."""
        if pose_landmarks_raw is None:
            return image
        mp_drawing.draw_landmarks(
            image,
            pose_landmarks_raw,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
        return image

    def close(self):
        self.pose.close()
