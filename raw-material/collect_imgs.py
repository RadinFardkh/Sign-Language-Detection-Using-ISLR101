import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Collect RGB images for OpenPose-based training.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--data-dir', default='./data')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    cap = cv2.VideoCapture(args.camera_index)

    for class_idx in range(args.classes):
        class_dir = os.path.join(args.data_dir, str(class_idx))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {class_idx}')

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.putText(
                frame,
                'Ready? Press "Q" to start capture',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        counter = 0
        while counter < args.samples_per_class:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
