Sign Language Detector (OpenPose + OpenCV)

This version of sign-language-detector-python uses OpenPose BODY_18 keypoints via OpenCV DNN instead of MediaPipe.

The pipeline is simple:

Camera → OpenPose keypoints → ML classifier → realtime prediction

Requirements

Python 3.8+

OpenCV (with DNN support)

NumPy, scikit-learn

A webcam

Install dependencies (example):

pip install opencv-python numpy scikit-learn
Setup
Download OpenPose model files

You must download the COCO OpenPose model manually:

pose_deploy_linevec.prototxt

pose_iter_440000.caffemodel

Place them anywhere you want (just remember the paths).

Workflow
1. Collect training images

Capture labeled images for each sign class using your webcam:

python collect_imgs.py \
  --camera-index 0 \
  --classes 3 \
  --samples-per-class 100

Arguments

--camera-index → webcam ID (usually 0)

--classes → number of sign classes

--samples-per-class → images per class

Images will be saved automatically per class.

2. Create keypoint dataset (OpenPose)

Convert images into BODY_18 keypoints and serialize them into a dataset:

python create_dataset.py \
  --proto /path/to/pose_deploy_linevec.prototxt \
  --weights /path/to/pose_iter_440000.caffemodel

Output

data.pickle containing extracted keypoints + labels

3. Train the classifier

Train a simple ML model on the keypoint dataset:

python train_classifier.py

Output

model.p (trained classifier)

4. Realtime inference

Run live sign recognition using your webcam:

python inference_classifier.py \
  --proto /path/to/pose_deploy_linevec.prototxt \
  --weights /path/to/pose_iter_440000.caffemodel \
  --camera-index 0

Press q to quit.

Notes

This implementation uses BODY_18 keypoints only (hands are inferred from body pose).

Lighting and camera angle matter a lot—keep them consistent.

For better accuracy, collect more samples per class.

Why OpenPose?

No MediaPipe dependency

Works fully offline

Easy to extend to other pose-based tasks
