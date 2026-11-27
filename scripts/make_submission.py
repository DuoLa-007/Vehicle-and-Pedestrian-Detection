import argparse
import json
import math
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO text predictions to the DataFountain 552 JSON layout."
    )
    parser.add_argument("--pred_dir", required=True, help="YOLO prediction TXT directory")
    parser.add_argument("--image_dir", required=True, help="Directory containing test JPG images")
    parser.add_argument("--classes", default="dataset/classes.json")
    parser.add_argument("--out", default="submission.json")
    parser.add_argument("--min_conf", type=float, default=0.001)
    args = parser.parse_args()

    if not 0 <= args.min_conf <= 1:
        parser.error("--min_conf must be between 0 and 1")

    class_info = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    class_names = class_info["names"]
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))
    results = []

    for image_path in image_files:
        prediction_file = pred_dir / f"{image_path.stem}.txt"
        if not prediction_file.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        for line_number, line in enumerate(
            prediction_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(
                    f"{prediction_file}:{line_number}: expected 6 columns "
                    "(class xc yc w h confidence)"
                )

            class_value, xc, yc, box_w, box_h, confidence = map(float, parts)
            if not all(math.isfinite(value) for value in (
                class_value, xc, yc, box_w, box_h, confidence
            )):
                continue

            class_id = int(class_value)
            if class_id < 0 or class_id >= len(class_names):
                continue
            if confidence < args.min_conf or not 0 <= confidence <= 1:
                continue
            if box_w <= 0 or box_h <= 0:
                continue

            x1 = max(0.0, min(width, (xc - box_w / 2) * width))
            y1 = max(0.0, min(height, (yc - box_h / 2) * height))
            x2 = max(0.0, min(width, (xc + box_w / 2) * width))
            y2 = max(0.0, min(height, (yc + box_h / 2) * height))
            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "filename": f"test_images\\{image_path.name}",
                "box": {
                    "xmin": round(x1, 2),
                    "ymin": round(y1, 2),
                    "xmax": round(x2, 2),
                    "ymax": round(y2, 2),
                },
                "label": class_names[class_id],
                "conf": round(confidence, 6),
            })

    output = {"annotations": results}
    Path(args.out).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.out}: {len(results)} detections "
        f"from {len(image_files)} test images."
    )


if __name__ == "__main__":
    main()
