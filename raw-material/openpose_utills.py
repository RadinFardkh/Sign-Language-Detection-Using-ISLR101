import cv2
import numpy as np

# OpenPose BODY_18 (COCO) keypoint pairs for drawing skeleton connections.
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4),
    (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]


class OpenPoseEstimator:
    """Small wrapper around OpenCV DNN OpenPose model inference."""

    def __init__(self, proto_file, weights_file, input_size=(368, 368), threshold=0.1):
        self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
        self.input_size = input_size
        self.threshold = threshold

    def extract_keypoints(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()

        points = []
        for i in range(output.shape[1]):
            prob_map = output[0, i, :, :]
            _, confidence, _, point = cv2.minMaxLoc(prob_map)

            x = int((w * point[0]) / output.shape[3])
            y = int((h * point[1]) / output.shape[2])

            if confidence > self.threshold:
                points.append((x, y, confidence))
            else:
                points.append(None)

        return points


def points_to_feature_vector(points):
    """Convert BODY_18 points to a fixed-size normalized feature vector."""
    valid_points = [(x, y) for p in points if p is not None for x, y, _ in [p]]
    if not valid_points:
        return None

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)

    features = []
    for p in points[:18]:
        if p is None:
            features.extend([0.0, 0.0])
        else:
            x, y, _ = p
            features.extend([(x - min_x) / width, (y - min_y) / height])

    return np.array(features, dtype=np.float32)


def draw_skeleton(frame, points):
    for pair in POSE_PAIRS:
        part_a, part_b = pair
        if part_a >= len(points) or part_b >= len(points):
            continue
        if points[part_a] is None or points[part_b] is None:
            continue
        x1, y1, _ = points[part_a]
        x2, y2, _ = points[part_b]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)
