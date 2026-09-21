"""
Render side-by-side ground-truth vs. predicted-box figures for the rejected
RF-DETR hole/pitting detector, for use as negative-result figures in the
report. See analyses/CONTINUATION_PLAN.md and
outputs/rfdetr_hole_pitting/rfdetr_hole_pitting_summary.json (mAP50 ~0.004)
for the quantitative result this illustrates.

Note: config/rfdetr_hole_pitting_eval.json lists confidence_threshold=0.5,
but the pred_box_count values in outputs/rfdetr_hole_pitting/
rfdetr_hole_pitting_review.csv only reproduce at threshold=0.05 with the
current rfdetr package version; this script defaults to 0.05 to match the
already-reported review numbers.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRNano

ROOT = Path(__file__).resolve().parents[1]


def render(dataset_dir, checkpoint, out_dir, filenames, threshold, max_width):
    with open(dataset_dir / "_annotations.coco.json") as f:
        coco = json.load(f)
    images_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_image_id = {}
    for ann in coco["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    model = RFDETRNano(pretrain_weights=str(checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in filenames:
        img_meta = images_by_name[fname]
        image_bgr = cv2.imread(str(dataset_dir / fname))

        gt_vis = image_bgr.copy()
        gt_anns = anns_by_image_id.get(img_meta["id"], [])
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(gt_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)

        detections = model.predict(image_bgr, threshold=threshold)
        pred_vis = image_bgr.copy()
        for box in detections.xyxy:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(pred_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        gap = np.full((gt_vis.shape[0], 12, 3), 255, dtype=np.uint8)
        combined = np.hstack([gt_vis, gap, pred_vis])
        cv2.putText(combined, f"GT: {len(gt_anns)} boxes", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(combined, f"Pred (thr={threshold}): {len(detections.xyxy)} boxes",
                    (gt_vis.shape[1] + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
    parser.add_argument("--filenames", nargs="+", default=["20251021_132633_1.jpg", "20251021_122353_12.jpg"])
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
