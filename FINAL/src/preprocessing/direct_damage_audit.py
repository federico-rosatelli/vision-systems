import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.biological_features import analyze_plant_biology
from src.preprocessing.frame_crop import (
    crop_frame_interior,
    detect_frame,
    read_image_oriented,
    select_audit_rows,
)
from src.preprocessing.plant_regions import extract_plant_regions
from src.preprocessing.plant_weight_audit import _write_contact_sheet


REVIEW_COLUMNS = [
    "review_status", "leaf_mask_usable", "holes_usable", "pitting_usable",
    "soil_false_damage", "damage_percentage_believable", "review_notes",
]
REVIEW_STATUSES = {"pass", "partial", "fail"}


def combined_damage_percentage(leaf_area, hole_area, pitting_area):
    return 100.0 * (hole_area + pitting_area) / leaf_area if leaf_area > 0 else 0.0


def analyze_frame_damage(frame, plant_parameters=None, hsv_bounds=None,
                         hole_classification="brightness", soil_lab_distance=35.0):
    plant_parameters = plant_parameters or {}
    _, regions = extract_plant_regions(frame, **plant_parameters)
    leaf_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hole_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pitting_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    totals = {"hole_count": 0, "pitting_count": 0}

    for region in regions:
        x1, y1, x2, y2 = region.patch_box
        patch = frame[y1:y2, x1:x2]
        metrics, pitting_contours, hole_contours, patch_leaf_mask = analyze_plant_biology(
            patch, hsv_bounds=hsv_bounds, hole_classification=hole_classification,
            soil_lab_distance=soil_lab_distance,
        )
        target_leaf = leaf_mask[y1:y2, x1:x2]
        cv2.bitwise_or(target_leaf, patch_leaf_mask, dst=target_leaf)
        translated_pitting = [contour + np.array([[[x1, y1]]]) for contour in pitting_contours]
        translated_holes = [contour + np.array([[[x1, y1]]]) for contour in hole_contours]
        cv2.drawContours(pitting_mask, translated_pitting, -1, 255, -1)
        cv2.drawContours(hole_mask, translated_holes, -1, 255, -1)
        totals["pitting_count"] += metrics["pitting_count"]
        totals["hole_count"] += metrics["hole_count"]

    leaf_area = int(np.count_nonzero(leaf_mask))
    hole_area = int(np.count_nonzero(hole_mask))
    pitting_area = int(np.count_nonzero(pitting_mask))
    metrics = {
        "region_count": len(regions),
        "leaf_area": leaf_area,
        "hole_area": hole_area,
        "hole_count": totals["hole_count"],
        "pitting_area": pitting_area,
        "pitting_count": totals["pitting_count"],
        "damaged_area": hole_area + pitting_area,
        "damage_percentage": combined_damage_percentage(leaf_area, hole_area, pitting_area),
    }
    return metrics, leaf_mask, hole_mask, pitting_mask


def draw_damage_overlay(frame, leaf_mask, hole_mask, pitting_mask, title):
    overlay = frame.copy()
    leaf = leaf_mask > 0
    overlay[leaf] = (
        0.6 * overlay[leaf].astype(np.float32)
        + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    overlay[pitting_mask > 0] = (0, 0, 255)
    overlay[hole_mask > 0] = (255, 0, 0)
    cv2.putText(overlay, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 255, 255), 3)
    cv2.putText(overlay, "GREEN=leaf BLUE=hole RED=pitting", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    width = min(1400, overlay.shape[1])
    height = round(overlay.shape[0] * width / overlay.shape[1])
    return cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)


def _preserve_reviews(audit, audit_path):
    if not audit_path.is_file():
        return audit
    previous = pd.read_csv(audit_path, keep_default_na=False)
    if "filename" not in previous or not set(REVIEW_COLUMNS).issubset(previous.columns):
        return audit
    reviews = previous[["filename", *REVIEW_COLUMNS]].drop_duplicates("filename")
    audit = audit.drop(columns=REVIEW_COLUMNS).merge(reviews, on="filename", how="left")
    audit[REVIEW_COLUMNS] = audit[REVIEW_COLUMNS].fillna("")
    return audit


def generate_direct_damage_audit(manifest, output_dir, sample_size=40, seed=73,
                                 working_width=1000, interior_inset_fraction=0.025,
                                 plant_parameters=None, hsv_bounds=None,
                                 hole_classification="brightness", soil_lab_distance=35.0):
    selected = select_audit_rows(manifest, sample_size=sample_size, seed=seed)
    if set(selected["split"]) - {"train", "val"}:
        raise ValueError("Direct-damage audit must not contain test images")
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records, overlay_paths = [], []

    for _, row in selected.iterrows():
        image = read_image_oriented(row["image_path"])
        detection = detect_frame(image, working_width=working_width)
        if detection.status != "detected":
            continue
        frame, _ = crop_frame_interior(
            image, detection.corners, inset_fraction=interior_inset_fraction
        )
        metrics, leaf_mask, hole_mask, pitting_mask = analyze_frame_damage(
            frame, plant_parameters=plant_parameters, hsv_bounds=hsv_bounds,
            hole_classification=hole_classification,
            soil_lab_distance=soil_lab_distance,
        )
        stem = Path(row["filename"]).stem
        overlay_path = overlays_dir / f"{stem}_damage.jpg"
        combined_mask = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        combined_mask[leaf_mask > 0] = (0, 255, 0)
        combined_mask[pitting_mask > 0] = (0, 0, 255)
        combined_mask[hole_mask > 0] = (255, 0, 0)
        mask_path = masks_dir / f"{stem}_damage_mask.png"
        cv2.imwrite(str(mask_path), combined_mask)
        title = (
            f"{row['filename']} {row['split']} expert={row['mean_score']:.2f} "
            f"direct={metrics['damage_percentage']:.3f}%"
        )
        cv2.imwrite(str(overlay_path), draw_damage_overlay(
            frame, leaf_mask, hole_mask, pitting_mask, title
        ), [cv2.IMWRITE_JPEG_QUALITY, 92])
        overlay_paths.append(overlay_path)
        records.append({
            "filename": row["filename"], "split": row["split"],
            "expert_score": row["mean_score"], "score_bin": str(row["score_bin"]),
            "frame_confidence": detection.confidence, **metrics,
            "overlay_path": str(overlay_path), "mask_path": str(mask_path),
            **{column: "" for column in REVIEW_COLUMNS},
        })

    audit_path = output_dir / "direct_damage_audit.csv"
    audit = _preserve_reviews(pd.DataFrame(records), audit_path)
    audit.to_csv(audit_path, index=False)
    _write_contact_sheet(overlay_paths, output_dir / "direct_damage_contact_sheet.jpg")
    summary = {
        "schema_version": 1, "manifest": str(Path(manifest).resolve()),
        "sample_size": len(audit), "seed": seed,
        "split_counts": audit["split"].value_counts().sort_index().to_dict(),
        "score_bin_counts": audit["score_bin"].value_counts().sort_index().to_dict(),
        "test_images": int((audit["split"] == "test").sum()),
        "manual_review_complete": False,
        "acceptance_rule": "At least 90% usable and no unresolved systematic error.",
        "known_limitation": "Missing leaf-edge damage is not estimated.",
    }
    (output_dir / "direct_damage_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit, summary


def summarize_manual_review(output_dir):
    output_dir = Path(output_dir)
    audit = pd.read_csv(output_dir / "direct_damage_audit.csv", keep_default_na=False)
    if (audit["review_status"] == "").any() or set(audit["review_status"]) - REVIEW_STATUSES:
        raise ValueError("Every review_status must be pass, partial, or fail")
    required = ["leaf_mask_usable", "holes_usable", "pitting_usable",
                "damage_percentage_believable"]
    usable = audit["review_status"].isin({"pass", "partial"})
    for column in required:
        usable &= audit[column].astype(str).str.lower() == "true"
    summary_path = output_dir / "direct_damage_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "manual_review_complete": True,
        "review_status_counts": audit["review_status"].value_counts().to_dict(),
        "usable_count": int(usable.sum()), "usable_rate": float(usable.mean()),
        "exit_criterion_met": bool(usable.mean() >= 0.90),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit direct leaf-damage measurements")
    parser.add_argument("--config", default="configs/direct_damage_audit.json")
    parser.add_argument("--summarize-review", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.summarize_review:
        summary = summarize_manual_review(config["output_dir"])
    else:
        _, summary = generate_direct_damage_audit(**config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
