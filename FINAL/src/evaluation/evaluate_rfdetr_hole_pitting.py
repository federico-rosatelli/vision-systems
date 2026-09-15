"""Evaluate the fine-tuned RF-DETR hole/pitting detector on the held-out eval split.

Computes detection mAP against the manually annotated eval COCO subset and writes a
per-image review CSV (same shape as `direct_damage_audit.csv`) for the manual pass/fail
call described in `analyses/HOLE_PITTING_ANNOTATION_PLAN.md`, step 4.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import supervision as sv
from supervision.metrics import MeanAveragePrecision
from rfdetr import RFDETRNano


def load_config():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR hole/pitting detector")
    parser.add_argument("--config", type=str, default="configs/rfdetr_hole_pitting_eval.json")
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        return json.load(f)


def main():
    config = load_config()
    eval_dir = Path(config["dataset_dir"]) / "valid"
    ann_path = eval_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No eval annotations at {ann_path}; export COCO with a valid/ split first.")

    model = RFDETRNano(pretrain_weights=config["checkpoint"])

    ground_truth = sv.DetectionDataset.from_coco(
        images_directory_path=str(eval_dir),
        annotations_path=str(ann_path),
    )

    predictions = []
    targets = []
    rows = []
    for image_path, image, annotations in ground_truth:
        detections = model.predict(image, threshold=config.get("confidence_threshold", 0.5))
        predictions.append(detections)
        targets.append(annotations)
        rows.append({
            "filename": Path(image_path).name,
            "gt_box_count": len(annotations),
            "pred_box_count": len(detections),
        })

    map_result = MeanAveragePrecision().update(predictions, targets).compute()

    review_df = pd.DataFrame(rows)
    review_df["review_status"] = ""
    review_df["usable"] = ""
    review_df["review_notes"] = ""

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    review_path = out_dir / "rfdetr_hole_pitting_review.csv"
    review_df.to_csv(review_path, index=False)

    summary = {"mAP50": map_result.map50, "mAP50_95": map_result.map50_95}
    with open(out_dir / "rfdetr_hole_pitting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"mAP@50: {map_result.map50:.4f}  mAP@50:95: {map_result.map50_95:.4f}")
    print(f"Per-image review table written to {review_path} -- fill in usable/review_notes manually.")


if __name__ == "__main__":
    main()
