"""Convert AnyLabeling's per-image JSON labels into a train/valid COCO split.

AnyLabeling's own "Export Annotations -> COCO" produced 0 annotations for this
project's circle-shape labels (a bug in its exporter), so this converts the raw
per-image labelme-style JSON sidecars directly. Supports circle, rectangle, and
polygon shapes by reducing each to its bounding box, since RF-DETR (detection, not
segmentation) only needs boxes here.

The train/valid split is taken from `annotation_subset` in
`outputs/hole_pitting_annotations/annotation_candidates.csv`, the same split
recorded when the candidate images were collected -- not re-derived here.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

CATEGORIES = [
    {"id": 1, "name": "shot_hole", "supercategory": "none"},
    {"id": 2, "name": "pitting", "supercategory": "none"},
]
CATEGORY_IDS = {c["name"]: c["id"] for c in CATEGORIES}


def shape_bbox(shape):
    points = shape["points"]
    if shape["shape_type"] == "circle":
        (cx, cy), (ex, ey) = points
        r = math.hypot(ex - cx, ey - cy)
        return cx - r, cy - r, 2 * r, 2 * r
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def build_split(rows, images_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1

    for image_id, row in enumerate(rows, start=1):
        filename = row["filename"]
        src_image = Path(images_dir) / filename
        shutil.copy2(src_image, out_dir / filename)

        label_path = Path(images_dir) / (Path(filename).stem + ".json")
        if label_path.exists():
            label = json.load(open(label_path))
            width, height = label["imageWidth"], label["imageHeight"]
            shapes = label["shapes"]
        else:
            from PIL import Image
            with Image.open(src_image) as img:
                width, height = img.size
            shapes = []

        coco["images"].append({
            "id": image_id, "file_name": filename,
            "width": width, "height": height, "license": 1, "date_captured": "",
        })

        for shape in shapes:
            if shape["label"] not in CATEGORY_IDS:
                continue
            x, y, w, h = shape_bbox(shape)
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id,
                "category_id": CATEGORY_IDS[shape["label"]],
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1

    with open(out_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    n_labeled = sum(1 for img in coco["images"]
                    if any(a["image_id"] == img["id"] for a in coco["annotations"]))
    print(f"{out_dir}: {len(coco['images'])} images, {len(coco['annotations'])} boxes "
          f"({n_labeled} images with at least one box)")


def main():
    parser = argparse.ArgumentParser(description="Convert AnyLabeling JSON to train/valid COCO")
    parser.add_argument("--config", type=str, default="configs/labelme_to_coco.json")
    args, _ = parser.parse_known_args()
    config = json.load(open(args.config))

    manifest = pd.read_csv(config["candidates_csv"])
    images_dir = config["images_dir"]

    for subset, out_key in [("train", "train_dir"), ("eval", "valid_dir")]:
        rows = manifest[manifest["annotation_subset"] == subset].to_dict("records")
        build_split(rows, images_dir, config[out_key])


if __name__ == "__main__":
    main()
