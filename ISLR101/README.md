# Sign Language Detector — OpenPose + OpenCV

A real-time sign language recognition system that uses OpenPose BODY_18 keypoints extracted via OpenCV DNN as input features to a trained ML classifier.

> **Note:** This is a drop-in replacement for the MediaPipe-based variant. No MediaPipe dependency is required.

## Pipeline

```
Webcam → OpenPose keypoint extraction → ML classifier → Real-time prediction
```

## Requirements

- Python 3.8+
- OpenCV with DNN support
- NumPy, scikit-learn
- A webcam

Install dependencies:

```bash
pip install opencv-python numpy scikit-learn
```

## Setup

### Download OpenPose Model Files

The COCO OpenPose model must be downloaded manually:

- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`

Place these files in any directory and note the paths — they are required by `create_dataset.py` and `inference_classifier.py`.

## Workflow

### 1. Collect Training Images

Capture labeled images for each sign class using your webcam:

```bash
python collect_imgs.py \
  --camera-index 0 \
  --classes 3 \
  --samples-per-class 100
```

| Argument | Description |
|---|---|
| `--camera-index` | Webcam device ID (usually `0`) |
| `--classes` | Number of sign classes to collect |
| `--samples-per-class` | Number of images to capture per class |

Images are saved automatically, organized by class.

### 2. Create Keypoint Dataset

Extract BODY_18 keypoints from the collected images and serialize them into a dataset file:

```bash
python create_dataset.py \
  --proto /path/to/pose_deploy_linevec.prototxt \
  --weights /path/to/pose_iter_440000.caffemodel
```

**Output:** `data.pickle` — extracted keypoints with corresponding labels.

### 3. Train the Classifier

Train an ML model on the keypoint dataset:

```bash
python train_classifier.py
```

**Output:** `model.p` — the serialized trained classifier.

### 4. Run Real-Time Inference

Start live sign recognition using your webcam:

```bash
python inference_classifier.py \
  --proto /path/to/pose_deploy_linevec.prototxt \
  --weights /path/to/pose_iter_440000.caffemodel \
  --camera-index 0
```

Press `q` to quit.

## Notes

- This implementation uses **BODY_18 keypoints only**. Hand pose is inferred from body keypoints rather than dedicated hand tracking.
- Consistent **lighting and camera angle** significantly affect accuracy — keep both stable across data collection and inference.
- To improve model accuracy, increase `--samples-per-class` during data collection.

## Why OpenPose?

| Advantage | Details |
|---|---|
| No MediaPipe dependency | Eliminates an extra library requirement |
| Fully offline | No network calls or cloud APIs needed |
| Extensible | Easily adapted to other pose-based classification tasks |
