import argparse
import json
import math
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

SEED = 42


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract an archive while rejecting paths that escape the destination."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        root = dest.resolve()
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe zip path: {member.filename}")
        archive.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DataFountain 552 data in YOLO detection format."
    )
    parser.add_argument("--raw_dir", default="raw")
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio must be between 0 and 1")

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    train_zip = raw / "train_dataset.zip"
    test_zip = raw / "test_dataset.zip"
    if not train_zip.exists() or not test_zip.exists():
        raise FileNotFoundError(
            "Place train_dataset.zip and test_dataset.zip in the raw directory."
        )

    work = out / "_extracted"
    if out.exists():
        shutil.rmtree(out)
    work.mkdir(parents=True)

    try:
        safe_extract(train_zip, work / "train")
        safe_extract(test_zip, work / "test")

        train_root = work / "train" / "train_images"
        test_root = work / "test" / "test_images"
        ann_path = work / "train" / "train.json"
        if not ann_path.exists():
            raise FileNotFoundError("train.json was not found in the training archive.")

        raw_ann = json.loads(ann_path.read_text(encoding="utf-8"))
        annotations = raw_ann["annotations"]
        image_paths = sorted(train_root.glob("*.jpg"))
        test_paths = sorted(test_root.glob("*.jpg"))
        if not image_paths or not test_paths:
            raise RuntimeError("No training or test JPG images were found.")

        class_names = list(dict.fromkeys(item["label"] for item in annotations))
        class_to_id = {name: index for index, name in enumerate(class_names)}
        grouped = {path.name: [] for path in image_paths}
        audit = Counter()
        bad_examples = []

        image_sizes = {}
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image_sizes[image_path.name] = image.size

        for item in annotations:
            filename = str(item.get("filename", "")).replace("\\", "/").split("/")[-1]
            if filename not in grouped:
                audit["unknown_image"] += 1
                continue

            box = item.get("box") or {}
            values = [box.get(key) for key in ("xmin", "ymin", "xmax", "ymax")]
            if any(value is None for value in values):
                audit["missing_box"] += 1
                bad_examples.append({
                    "filename": filename,
                    "label": item.get("label"),
                    "reason": "missing_box",
                })
                continue

            try:
                x1, y1, x2, y2 = map(float, values)
            except (TypeError, ValueError):
                audit["non_numeric_box"] += 1
                continue

            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                audit["non_finite_box"] += 1
                continue
            if x2 <= x1 or y2 <= y1:
                audit["non_positive_area"] += 1
                continue

            width, height = image_sizes[filename]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                audit["outside_image"] += 1
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            grouped[filename].append((class_to_id[item["label"]], xc, yc, bw, bh))
            audit["valid_box"] += 1

        names = sorted(grouped)
        random.Random(SEED).shuffle(names)
        val_count = max(1, int(round(len(names) * args.val_ratio)))
        val_names = set(names[:val_count])
        train_names = set(names[val_count:])

        for split in ("train", "val", "test"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        def copy_split(name_set, split):
            for name in sorted(name_set):
                shutil.copy2(train_root / name, out / "images" / split / name)
                rows = grouped[name]
                label_path = out / "labels" / split / (Path(name).stem + ".txt")
                label_path.write_text(
                    "".join(
                        f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n"
                        for class_id, xc, yc, bw, bh in rows
                    ),
                    encoding="utf-8",
                )

        copy_split(train_names, "train")
        copy_split(val_names, "val")
        for path in test_paths:
            shutil.copy2(path, out / "images" / "test" / path.name)

        (out / "classes.json").write_text(
            json.dumps(
                {"names": class_names, "class_to_id": class_to_id},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_config = {
            "path": str(out.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }
        (out / "data.yaml").write_text(
            json.dumps(data_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "test_images.txt").write_text(
            "".join(f"test_images/{path.name}\n" for path in test_paths),
            encoding="utf-8",
        )

        report = {
            "train_images": len(train_names),
            "val_images": len(val_names),
            "test_images": len(test_paths),
            "raw_annotations": len(annotations),
            "class_count": len(class_names),
            "class_names": class_names,
            "class_annotation_counts": dict(Counter(
                item["label"] for item in annotations if item.get("label") is not None
            )),
            "audit_counts": dict(audit),
            "bad_examples": bad_examples,
            "split_seed": SEED,
            "val_ratio_requested": args.val_ratio,
        }
        (out / "audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
