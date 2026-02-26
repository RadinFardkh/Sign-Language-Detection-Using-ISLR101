-import os
-import pickle
-
-import mediapipe as mp
-import cv2
-import matplotlib.pyplot as plt
-
-
-mp_hands = mp.solutions.hands
-mp_drawing = mp.solutions.drawing_utils
-mp_drawing_styles = mp.solutions.drawing_styles
-
-hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
-
-DATA_DIR = './data'
-
-data = []
-labels = []
-for dir_ in os.listdir(DATA_DIR):
-    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
-        data_aux = []
-
-        x_ = []
-        y_ = []
-
-        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
-        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
-
-        results = hands.process(img_rgb)
-        if results.multi_hand_landmarks:
-            for hand_landmarks in results.multi_hand_landmarks:
-                for i in range(len(hand_landmarks.landmark)):
-                    x = hand_landmarks.landmark[i].x
-                    y = hand_landmarks.landmark[i].y
-
-                    x_.append(x)
-                    y_.append(y)
-
-                for i in range(len(hand_landmarks.landmark)):
-                    x = hand_landmarks.landmark[i].x
-                    y = hand_landmarks.landmark[i].y
-                    data_aux.append(x - min(x_))
-                    data_aux.append(y - min(y_))
-
-            data.append(data_aux)
-            labels.append(dir_)
-
-f = open('data.pickle', 'wb')
-pickle.dump({'data': data, 'labels': labels}, f)
-f.close()
+import argparse
+import os
+import pickle
+
+import cv2
+
+from openpose_utils import OpenPoseEstimator, points_to_feature_vector
+
+
+DATA_DIR = './data'
+
+
+def parse_args():
+    parser = argparse.ArgumentParser(description='Build a dataset using OpenPose BODY_18 keypoints.')
+    parser.add_argument('--proto', required=True, help='Path to OpenPose .prototxt file.')
+    parser.add_argument('--weights', required=True, help='Path to OpenPose .caffemodel weights file.')
+    parser.add_argument('--threshold', type=float, default=0.1, help='Keypoint confidence threshold.')
+    parser.add_argument('--input-width', type=int, default=368)
+    parser.add_argument('--input-height', type=int, default=368)
+    return parser.parse_args()
+
+
+def main():
+    args = parse_args()
+    estimator = OpenPoseEstimator(
+        proto_file=args.proto,
+        weights_file=args.weights,
+        input_size=(args.input_width, args.input_height),
+        threshold=args.threshold,
+    )
+
+    data = []
+    labels = []
+
+    for dir_ in sorted(os.listdir(DATA_DIR)):
+        class_dir = os.path.join(DATA_DIR, dir_)
+        if not os.path.isdir(class_dir):
+            continue
+
+        for img_path in sorted(os.listdir(class_dir)):
+            image_path = os.path.join(class_dir, img_path)
+            img = cv2.imread(image_path)
+            if img is None:
+                continue
+
+            points = estimator.extract_keypoints(img)
+            feature_vector = points_to_feature_vector(points)
+            if feature_vector is None:
+                continue
+
+            data.append(feature_vector)
+            labels.append(dir_)
+
+    with open('data.pickle', 'wb') as f:
+        pickle.dump({'data': data, 'labels': labels}, f)
+
+    print(f'Saved {len(data)} samples to data.pickle')
+
+
+if __name__ == '__main__':
+    main()
