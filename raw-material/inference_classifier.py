import argparse
import pickle

import cv2
import numpy as np

from openpose_utils import OpenPoseEstimator, draw_skeleton, points_to_feature_vector


def parse_args():
    parser = argparse.ArgumentParser(description='Realtime sign inference using OpenPose keypoints.')
    parser.add_argument('--proto', required=True, help='Path to OpenPose .prototxt file.')
    parser.add_argument('--weights', required=True, help='Path to OpenPose .caffemodel weights file.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    return parser.parse_args()


def main():
    args = parse_args()

    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(args.camera_index)
    estimator = OpenPoseEstimator(
        proto_file=args.proto,
        weights_file=args.weights,
        input_size=(args.input_width, args.input_height),
        threshold=args.threshold,
    )

    labels_dict = {0: 'A', 1: 'B', 2: 'L'}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        points = estimator.extract_keypoints(frame)
        feature_vector = points_to_feature_vector(points)

        if feature_vector is not None:
            prediction = model.predict([np.asarray(feature_vector)])
            predicted_character = labels_dict.get(int(prediction[0]), str(prediction[0]))

            draw_skeleton(frame, points)

            valid_points = [(x, y) for p in points if p is not None for x, y, _ in [p]]
            if valid_points:
                xs = [p[0] for p in valid_points]
                ys = [p[1] for p in valid_points]
                x1, y1 = min(xs) - 10, min(ys) - 10
                x2, y2 = max(xs) + 10, max(ys) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
