# Dataset and method notes

## Task

The DataFountain 552 task concerns object detection in driving-scene images, with human and vehicle-related fine-grained categories. Consult the official competition page for the authoritative task definition, class list, data policy, and submission requirements.

## Data processing implemented here

`scripts/prepare_dataset.py` currently:
1. Extracts the train and test archives into a temporary working directory.
2. Reads `train.json` from the training archive.
3. Creates a deterministic image-level train/validation split (default validation ratio 0.15, seed 42).
4. Converts bounding boxes from pixel coordinates to normalized YOLO `class x_center y_center width height` rows.
5. Skips missing, non-numeric, non-finite, non-positive-area, or fully out-of-image boxes and records audit counters.
6. Writes class mapping, a YOLO-style data configuration, test image list, and audit report.

The script clips partially out-of-image boxes to image boundaries. Review the generated audit report and visually inspect labels before training.

## Model direction

The project is intended to evaluate a YOLO-family detector using the OneFlow ecosystem as required by the competition. The upstream implementation and its installation instructions are maintained at [OneFlow-Inc/one-yolov5](https://github.com/Oneflow-Inc/one-yolov5). Confirm current platform and CUDA compatibility before installation.

## Important limitations

- No model has been trained or benchmarked as part of this repository setup.
- No mAP, precision, recall, inference speed, or leaderboard score is claimed.
- Class mapping is generated from the order of first appearance in the training annotation file; preserve the generated `classes.json` for all later stages.
- Confirm that the generated submission JSON matches the current official evaluator schema before uploading it.
