import os
import json
import csv
import random
import math
from collections import defaultdict
from PIL import Image

# ==================== 配置区域 ====================
RAW_DATA_DIR = "./raw_data"
IMAGE_DIR = os.path.join(RAW_DATA_DIR, "train_images")
JSON_PATH = os.path.join(RAW_DATA_DIR, "train.json")
OUTPUT_DIR = "./processed_data"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# True 表示过滤对应目标，False 表示保留
filter_truncated = False
filter_occluded = False
filter_ignore = True

EXPECTED_CLASS_COUNT = 22
# =================================================


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ["1", "true", "yes", "是"]
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(IMAGE_DIR):
        print("错误：找不到图片目录：", IMAGE_DIR)
        return
    if not os.path.isfile(JSON_PATH):
        print("错误：找不到标注文件：", JSON_PATH)
        return

    # 读取统一的 train.json
    try:
        with open(JSON_PATH, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        print("JSON文件读取失败：", error)
        return

    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        print("错误：train.json 顶层应包含 annotations 列表。")
        return

    annotations = data["annotations"]
    print("=" * 60)
    print("原始数据统计")
    print("JSON标注记录数：", len(annotations))

    # 数据集成：扫描图片，并按文件名建立索引。
    image_files = {}
    duplicate_image_names = 0
    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            key = filename.lower()
            if key in image_files:
                duplicate_image_names += 1
            else:
                image_files[key] = filename

    annotations_by_image = defaultdict(list)
    all_class_names = set()
    malformed_annotation_count = 0
    duplicate_annotation_count = 0
    seen_annotations = set()

    for item in annotations:
        if not isinstance(item, dict):
            malformed_annotation_count += 1
            continue

        filename = item.get("filename")
        label = item.get("label")
        box = item.get("box")

        if not filename or not label or not isinstance(box, dict):
            malformed_annotation_count += 1
            continue

        # JSON使用反斜杠路径，统一转换成文件名，兼容Windows和Linux。
        image_name = str(filename).replace("\\", "/").split("/")[-1]
        label = str(label).strip()
        if not image_name or not label:
            malformed_annotation_count += 1
            continue

        duplicate_key = (
            image_name.lower(), label,
            str(box.get("xmin")), str(box.get("ymin")),
            str(box.get("xmax")), str(box.get("ymax")),
            str(item.get("truncated", 0)), str(item.get("occluded", 0)),
            str(item.get("ignore", item.get("Ignore", 0)))
        )
        if duplicate_key in seen_annotations:
            duplicate_annotation_count += 1
            continue
        seen_annotations.add(duplicate_key)

        annotations_by_image[image_name.lower()].append(item)
        all_class_names.add(label)

    if len(all_class_names) != EXPECTED_CLASS_COUNT:
        print("类别数量检查失败。期望：", EXPECTED_CLASS_COUNT,
              "实际：", len(all_class_names))
        print("识别到的类别：", sorted(all_class_names))
        print("请核对数据或修改 EXPECTED_CLASS_COUNT。")
        return

    class_names = sorted(all_class_names)
    class_to_id = {}
    for index, name in enumerate(class_names, start=1):
        class_to_id[name] = index

    print("\n22类 class_id 映射：")
    for name in class_names:
        print(class_to_id[name], "->", name)

    annotation_image_names = set(annotations_by_image.keys())
    image_file_names = set(image_files.keys())
    matched_names = sorted(annotation_image_names & image_file_names)
    annotations_without_image = annotation_image_names - image_file_names
    images_without_annotation = image_file_names - annotation_image_names

    print("\n数据集成结果")
    print("图片文件数：", len(image_files))
    print("JSON中涉及的图片数：", len(annotation_image_names))
    print("成功匹配图片数：", len(matched_names))
    print("有标注但找不到图片的数量：", len(annotations_without_image))
    print("没有标注记录的图片数量：", len(images_without_annotation))

    # 数据清洗：检查图片、标注字段和bbox。
    samples = []
    invalid_bbox_count = 0
    ignored_count = 0
    filtered_truncated_count = 0
    filtered_occluded_count = 0
    unreadable_image_count = 0
    empty_after_clean_count = 0
    cleaned_annotation_count = 0

    for image_name in matched_names:
        actual_filename = image_files[image_name]
        image_path = os.path.join(IMAGE_DIR, actual_filename)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError):
            unreadable_image_count += 1
            continue

        valid_objects = []
        for item in annotations_by_image[image_name]:
            ignore_flag = to_bool(item.get("ignore", item.get("Ignore", 0)))
            truncated_flag = to_bool(item.get("truncated", item.get("Truancated", 0)))
            occluded_flag = to_bool(item.get("occluded", item.get("Occluded", 0)))

            if filter_ignore and ignore_flag:
                ignored_count += 1
                continue
            if filter_truncated and truncated_flag:
                filtered_truncated_count += 1
                continue
            if filter_occluded and occluded_flag:
                filtered_occluded_count += 1
                continue

            box = item.get("box", {})
            try:
                x_min = float(box["xmin"])
                y_min = float(box["ymin"])
                x_max = float(box["xmax"])
                y_max = float(box["ymax"])
            except (KeyError, TypeError, ValueError):
                invalid_bbox_count += 1
                continue

            values = [x_min, y_min, x_max, y_max]
            if (not all(math.isfinite(value) for value in values) or
                    x_min < 0 or y_min < 0 or
                    x_max > width or y_max > height or
                    x_max <= x_min or y_max <= y_min):
                invalid_bbox_count += 1
                continue

            valid_objects.append({
                "category_id": class_to_id[str(item["label"]).strip()],
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "truncated": int(truncated_flag),
                "occluded": int(occluded_flag)
            })

        if not valid_objects:
            empty_after_clean_count += 1

        cleaned_annotation_count += len(valid_objects)
        samples.append({
            "filename": actual_filename,
            "width": width,
            "height": height,
            "objects": valid_objects
        })

    # 数据归约：按图片去重后进行固定随机种子的8:2划分。
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    train_count = int(len(samples) * TRAIN_RATIO)
    if len(samples) > 1:
        train_count = max(1, min(train_count, len(samples) - 1))
    train_samples = samples[:train_count]
    val_samples = samples[train_count:]

    def make_coco(sample_list):
        result = {
            "info": {"description": "DataFountain Human Vehicle Detection Dataset", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        for name in class_names:
            result["categories"].append({
                "id": class_to_id[name], "name": name, "supercategory": "object"
            })

        annotation_id = 1
        for image_id, sample in enumerate(sample_list, start=1):
            result["images"].append({
                "id": image_id,
                "file_name": "train_images/" + sample["filename"],
                "width": sample["width"],
                "height": sample["height"]
            })

            for obj in sample["objects"]:
                result["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj["category_id"],
                    "bbox": obj["bbox"],
                    "area": obj["area"],
                    "iscrowd": 0,
                    "truncated": obj["truncated"],
                    "occluded": obj["occluded"]
                })
                annotation_id += 1
        return result

    train_coco = make_coco(train_samples)
    val_coco = make_coco(val_samples)

    with open(os.path.join(OUTPUT_DIR, "train_coco.json"), "w", encoding="utf-8") as file:
        json.dump(train_coco, file, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val_coco.json"), "w", encoding="utf-8") as file:
        json.dump(val_coco, file, ensure_ascii=False, indent=2)

    # 按类别统计目标数量，以及包含该类别的图片数量。
    train_instances = {class_to_id[name]: 0 for name in class_names}
    val_instances = {class_to_id[name]: 0 for name in class_names}
    train_images = {class_to_id[name]: 0 for name in class_names}
    val_images = {class_to_id[name]: 0 for name in class_names}

    for sample in train_samples:
        present = set()
        for obj in sample["objects"]:
            cid = obj["category_id"]
            train_instances[cid] += 1
            present.add(cid)
        for cid in present:
            train_images[cid] += 1

    for sample in val_samples:
        present = set()
        for obj in sample["objects"]:
            cid = obj["category_id"]
            val_instances[cid] += 1
            present.add(cid)
        for cid in present:
            val_images[cid] += 1

    csv_path = os.path.join(OUTPUT_DIR, "dataset_statistics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class_id", "class_name", "train_instances", "val_instances",
            "total_instances", "train_images", "val_images", "total_images"
        ])
        for name in class_names:
            cid = class_to_id[name]
            writer.writerow([
                cid, name, train_instances[cid], val_instances[cid],
                train_instances[cid] + val_instances[cid],
                train_images[cid], val_images[cid], train_images[cid] + val_images[cid]
            ])

    print("\n" + "=" * 60)
    print("四项数据预处理统计")
    print("[数据清洗]")
    print("格式异常标注记录：", malformed_annotation_count)
    print("非法bbox：", invalid_bbox_count)
    print("无法读取的图片：", unreadable_image_count)
    print("过滤Ignore目标：", ignored_count)
    print("过滤截断目标：", filtered_truncated_count)
    print("过滤遮挡目标：", filtered_occluded_count)
    print("清洗后无目标图片：", empty_after_clean_count)
    print("[数据集成]")
    print("成功匹配图片：", len(matched_names))
    print("重复图片文件名：", duplicate_image_names)
    print("重复标注记录：", duplicate_annotation_count)
    print("[数据变换]")
    print("COCO格式转换目标数：", cleaned_annotation_count)
    print("类别映射数量：", len(class_to_id))
    print("[数据归约与划分]")
    print("清洗后图片数：", len(samples))
    print("训练集图片数：", len(train_samples))
    print("验证集图片数：", len(val_samples))
    print("训练集目标数：", len(train_coco["annotations"]))
    print("验证集目标数：", len(val_coco["annotations"]))
    print("随机种子：", RANDOM_SEED)
    print("过滤配置：filter_ignore=", filter_ignore,
          ", filter_truncated=", filter_truncated,
          ", filter_occluded=", filter_occluded, sep="")
    print("\n输出文件：")
    print(os.path.join(OUTPUT_DIR, "train_coco.json"))
    print(os.path.join(OUTPUT_DIR, "val_coco.json"))
    print(csv_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
