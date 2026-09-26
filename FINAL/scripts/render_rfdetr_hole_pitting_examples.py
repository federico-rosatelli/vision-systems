"""
Render side-by-side ground-truth vs. predicted-box figures for the RF-DETR
hole/pitting detector, color-coded per class (shot_hole vs pitting) so the
two lesion types can actually be told apart in the figure. See
analyses/HOLE_PITTING_ANNOTATION_PLAN.md section 5 for the EXIF-orientation
bug fix and the resulting weak-positive mAP@50 ~3.47% this illustrates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parents[1]

# BGR colors, consistent between GT and predictions so a color always means the same class.
CLASS_COLORS = {
    "shot_hole": (0, 128, 255),   # orange
    "pitting": (255, 0, 200),     # magenta
}
UNKNOWN_COLOR = (0, 200, 0)  # fallback if a category can't be resolved


def _draw_legend(image, categories_used):
    x, y = 10, image.shape[0] - 14
    for name in categories_used:
        color = CLASS_COLORS.get(name, UNKNOWN_COLOR)
        cv2.rectangle(image, (x, y - 12), (x + 18, y), color, -1)
        cv2.putText(image, name, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        x += 24 + 10 * len(name) + 30


def render(dataset_dir, checkpoint, out_dir, filenames, threshold, max_width):
    with open(dataset_dir / "_annotations.coco.json") as f:
        coco = json.load(f)
    images_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image_id = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)
    category_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    # RF-DETR predicts 0-indexed, contiguous class ids (sorted by COCO category id),
    # not the original 1-indexed COCO category ids used in the annotation file.
    category_names_by_zero_index = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]

    model = RFDETRNano(pretrain_weights=str(checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in filenames:
        img_meta = images_by_name[fname]
        image_bgr = cv2.imread(str(dataset_dir / fname))

        gt_vis = image_bgr.copy()
        gt_anns = anns_by_image_id.get(img_meta["id"], [])
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            class_name = category_name_by_id.get(ann["category_id"], "unknown")
            color = CLASS_COLORS.get(class_name, UNKNOWN_COLOR)
            cv2.rectangle(gt_vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        _draw_legend(gt_vis, sorted(category_name_by_id.values()))

        detections = model.predict(image_bgr, threshold=threshold)
        pred_vis = image_bgr.copy()
        pred_class_ids = getattr(detections, "class_id", None)
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box.astype(int)
            cid = int(pred_class_ids[i]) if pred_class_ids is not None else -1
            class_name = (
                category_names_by_zero_index[cid] if 0 <= cid < len(category_names_by_zero_index) else "unknown"
            )
            color = CLASS_COLORS.get(class_name, UNKNOWN_COLOR)
            cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, 1)
        _draw_legend(pred_vis, sorted(category_name_by_id.values()))

        gap = np.full((gt_vis.shape[0], 12, 3), 255, dtype=np.uint8)
        combined = np.hstack([gt_vis, gap, pred_vis])
        cv2.putText(combined, f"GT: {len(gt_anns)} boxes", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(combined, f"Pred (thr={threshold}): {len(detections.xyxy)} boxes",
                    (gt_vis.shape[1] + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        h, w = combined.shape[:2]
        if w > max_width:
            scale = max_width / w
            combined = cv2.resize(combined, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        out_path = out_dir / f"{Path(fname).stem}_gt_vs_pred.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{fname}: GT={len(gt_anns)} Pred={len(detections.xyxy)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render RF-DETR GT-vs-prediction example figures")
    parser.add_argument("--dataset-dir", default="outputs/hole_pitting_annotations/coco_export_tiled/valid")
    parser.add_argument("--checkpoint", default="outputs/rfdetr_hole_pitting/checkpoint_best_total.pth")
    parser.add_argument("--out-dir", default="outputs/rfdetr_hole_pitting/example_figures")
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=[
            "20251021_122655_11.jpg",
            "20251021_151409_9.jpg",
            "20251021_132633_1.jpg",
            "20251021_122353_12.jpg",
        ],
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-width", type=int, default=1600)
    args = parser.parse_args()

    render(
        ROOT / args.dataset_dir,
        ROOT / args.checkpoint,
        ROOT / args.out_dir,
        args.filenames,
        args.threshold,
        args.max_width,
    )


if __name__ == "__main__":
    main()
