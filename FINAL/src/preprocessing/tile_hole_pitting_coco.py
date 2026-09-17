"""Tile the full-frame hole/pitting COCO export into crops sized for RF-DETR.

The raw field photos are ~4000x3000 px; RF-DETR resizes inputs to 384x384 before
training, which shrinks an ~18x18 px hole/pitting box to under 2 px -- invisible to
the model. This produced exactly 0.0 mAP for all 20 epochs. Mirrors the same lesson
already documented for the whole-image DINOv3 baseline in
`analyses/FINAL_PROJECT_RESULTS.md` section 4.

For each image, nearby boxes are clustered and one tile is cropped per cluster
(padded so damage occupies a meaningful fraction of the tile), with box coordinates
remapped into tile-local space. Images with no annotations contribute one random
negative tile each, for background negatives.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

TILE_SIZE = 640
PADDING = 200
CLUSTER_DISTANCE = 500


def cluster_boxes(boxes):
    """Union-find clustering of boxes whose centers are within CLUSTER_DISTANCE."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    centers = [(b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2) for b in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 < CLUSTER_DISTANCE:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(boxes[i])
    return list(clusters.values())


def cluster_tile_rect(cluster, img_w, img_h):
    xs_min = min(b["bbox"][0] for b in cluster)
    ys_min = min(b["bbox"][1] for b in cluster)
    xs_max = max(b["bbox"][0] + b["bbox"][2] for b in cluster)
    ys_max = max(b["bbox"][1] + b["bbox"][3] for b in cluster)

    cx, cy = (xs_min + xs_max) / 2, (ys_min + ys_max) / 2
    half = max(TILE_SIZE / 2, (xs_max - xs_min) / 2 + PADDING, (ys_max - ys_min) / 2 + PADDING)

    x0 = max(0, min(img_w - 2 * half, cx - half))
    y0 = max(0, min(img_h - 2 * half, cy - half))
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(img_w, x0 + 2 * half)
    y1 = min(img_h, y0 + 2 * half)
    return x0, y0, x1, y1


def boxes_in_tile(all_boxes, x0, y0, x1, y1):
    tiled = []
    for b in all_boxes:
        bx, by, bw, bh = b["bbox"]
        cx, cy = bx + bw / 2, by + bh / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            nx0 = max(bx, x0) - x0
            ny0 = max(by, y0) - y0
            nx1 = min(bx + bw, x1) - x0
            ny1 = min(by + bh, y1) - y0
            tiled.append({"category_id": b["category_id"], "bbox": [nx0, ny0, nx1 - nx0, ny1 - ny0]})
    return tiled


def tile_coco(coco_path, images_dir, out_dir, seed):
    coco = json.load(open(coco_path))
    images_by_id = {im["id"]: im for im in coco["images"]}
    boxes_by_image = {}
    for a in coco["annotations"]:
        boxes_by_image.setdefault(a["image_id"], []).append(a)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    out_images, out_annotations = [], []
    tile_image_id, tile_ann_id = 1, 1

    for image_id, image_info in images_by_id.items():
        src = Image.open(Path(images_dir) / image_info["file_name"])
        img_w, img_h = image_info["width"], image_info["height"]
        boxes = boxes_by_image.get(image_id, [])

        if boxes:
            clusters = cluster_boxes(boxes)
            rects = [cluster_tile_rect(c, img_w, img_h) for c in clusters]
        else:
            half = TILE_SIZE / 2
            cx = rng.uniform(half, img_w - half) if img_w > TILE_SIZE else img_w / 2
            cy = rng.uniform(half, img_h - half) if img_h > TILE_SIZE else img_h / 2
            rects = [(max(0, cx - half), max(0, cy - half),
                      min(img_w, cx + half), min(img_h, cy + half))]

        for x0, y0, x1, y1 in rects:
            tile_img = src.crop((int(x0), int(y0), int(x1), int(y1)))
            tile_name = f"{Path(image_info['file_name']).stem}_{tile_image_id}.jpg"
            tile_img.save(out_dir / tile_name)

            out_images.append({
                "id": tile_image_id, "file_name": tile_name,
                "width": tile_img.width, "height": tile_img.height,
                "license": 1, "date_captured": "",
            })
            for b in boxes_in_tile(boxes, x0, y0, x1, y1):
                out_annotations.append({
                    "id": tile_ann_id, "image_id": tile_image_id,
                    "category_id": b["category_id"], "bbox": b["bbox"],
                    "area": b["bbox"][2] * b["bbox"][3], "iscrowd": 0, "segmentation": [],
                })
                tile_ann_id += 1
            tile_image_id += 1

    out_coco = {"images": out_images, "annotations": out_annotations, "categories": coco["categories"]}
    with open(out_dir / "_annotations.coco.json", "w") as f:
        json.dump(out_coco, f)

    print(f"{out_dir}: {len(out_images)} tiles, {len(out_annotations)} boxes "
          f"(from {len(images_by_id)} source images, {len(coco['annotations'])} source boxes)")


def main():
    parser = argparse.ArgumentParser(description="Tile hole/pitting COCO export for RF-DETR")
    parser.add_argument("--config", type=str, default="configs/tile_hole_pitting_coco.json")
    args, _ = parser.parse_known_args()
    config = json.load(open(args.config))

    for split in ["train", "valid"]:
        tile_coco(
            coco_path=f"{config['coco_export_dir']}/{split}/_annotations.coco.json",
            images_dir=f"{config['coco_export_dir']}/{split}",
            out_dir=f"{config['out_dir']}/{split}",
            seed=config["seed"],
        )


if __name__ == "__main__":
    main()
