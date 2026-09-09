import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.frame_crop import (
    crop_frame_interior,
    detect_frame,
    read_image_oriented,
    select_audit_rows,
)
from src.preprocessing.plant_regions import extract_plant_regions


REVIEW_STATUSES = {"pass", "partial", "fail"}


def normalized_area_weights(regions):
    areas = np.asarray([region.green_area for region in regions], dtype=np.float64)
    total = float(areas.sum())
    if total <= 0:
        return np.zeros(len(regions), dtype=np.float64)
    return areas / total


def _draw_overlay(image, mask, regions, weights, output_path, title=""):
    overlay = image.copy()
    selected = mask > 0
    overlay[selected] = (
        0.35 * overlay[selected].astype(np.float32)
        + 0.65 * np.array([255, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    thickness = max(2, image.shape[1] // 600)
    for region, weight in zip(regions, weights):
        x1, y1, x2, y2 = region.patch_box
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), thickness)
        label = f"P{region.region_id} A={region.green_area} w={weight:.3f}"
        cv2.putText(overlay, label, (x1 + 4, max(26, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    total = sum(region.green_area for region in regions)
    label = f"{title} regions={len(regions)} plant_pixels={total} weight_sum={weights.sum():.3f}"
    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 3)
    width = min(1400, image.shape[1])
    height = round(image.shape[0] * width / image.shape[1])
    preview = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _write_contact_sheet(paths, output_path, columns=4, tile_width=420):
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    tiles = []
    tile_height = round(tile_width * 0.78)
    for image in images:
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        y = (tile_height - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def generate_plant_weight_audit(manifest, output_dir, sample_size=40, seed=42,
                                working_width=1000, interior_inset_fraction=0.025,
                                **plant_parameters):
    selected = select_audit_rows(manifest, sample_size=sample_size, seed=seed)
    if set(selected["split"]) - {"train", "val"}:
        raise ValueError("Plant-weight audit must not contain test images")

    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []
    overlay_paths = []
    for _, row in selected.iterrows():
        image = read_image_oriented(row["image_path"])
        detection = detect_frame(image, working_width=working_width)
        if detection.status == "detected":
            crop, _ = crop_frame_interior(
                image, detection.corners, inset_fraction=interior_inset_fraction
            )
            mask, regions = extract_plant_regions(crop, **plant_parameters)
        else:
            crop = image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            regions = []
        weights = normalized_area_weights(regions)
        stem = Path(row["filename"]).stem
        overlay_path = overlays_dir / f"{stem}_weights.jpg"
        mask_path = masks_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        title = f"{row['filename']} {row['split']} score={row['mean_score']:.2f}"
        _draw_overlay(crop, mask, regions, weights, overlay_path, title=title)
        overlay_paths.append(overlay_path)
        region_data = [
            {
                "region_id": region.region_id,
                "mask_pixels": region.green_area,
                "weight": float(weight),
                "patch_box": list(region.patch_box),
            }
            for region, weight in zip(regions, weights)
        ]
        records.append({
            "filename": row["filename"],
            "split": row["split"],
            "mean_score": row["mean_score"],
            "score_bin": str(row["score_bin"]),
            "frame_status": detection.status,
            "frame_confidence": detection.confidence,
            "region_count": len(regions),
            "plant_pixels": int(sum(region.green_area for region in regions)),
            "weight_sum": float(weights.sum()),
            "regions_json": json.dumps(region_data),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "review_status": "",
            "missed_plant": "",
            "false_region": "",
            "split_or_merged_plant": "",
            "damaged_tissue_missing": "",
            "area_weight_usable": "",
            "review_notes": "",
        })

    audit = pd.DataFrame(records)
    audit_path = output_dir / "plant_weight_audit.csv"
    review_columns = [
        "review_status", "missed_plant", "false_region",
        "split_or_merged_plant", "damaged_tissue_missing",
        "area_weight_usable", "review_notes",
    ]
    if audit_path.is_file():
        previous = pd.read_csv(audit_path, keep_default_na=False)
        if "filename" in previous and set(review_columns).issubset(previous.columns):
            previous_review = previous[["filename", *review_columns]].drop_duplicates("filename")
            audit = audit.drop(columns=review_columns).merge(
                previous_review, on="filename", how="left"
            )
            audit[review_columns] = audit[review_columns].fillna("")
    audit.to_csv(audit_path, index=False)
    _write_contact_sheet(overlay_paths, output_dir / "plant_weight_contact_sheet.jpg")
    valid_weights = (audit["region_count"] > 0) & np.isclose(audit["weight_sum"], 1.0)
    summary = {
        "schema_version": 1,
        "manifest": str(Path(manifest).resolve()),
        "sample_size": len(audit),
        "seed": seed,
        "test_images": int((audit["split"] == "test").sum()),
        "split_counts": audit["split"].value_counts().sort_index().to_dict(),
        "score_bin_counts": audit["score_bin"].value_counts().sort_index().to_dict(),
        "frame_status_counts": audit["frame_status"].value_counts().to_dict(),
        "images_with_valid_weights": int(valid_weights.sum()),
        "manual_review_complete": False,
        "acceptance_rule": "At least 90% pass or acceptable partial; no unresolved systematic area bias.",
        "plant_parameters": plant_parameters,
    }
    with (output_dir / "plant_weight_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit, summary


def summarize_manual_review(output_dir):
    output_dir = Path(output_dir)
    audit_path = output_dir / "plant_weight_audit.csv"
    audit = pd.read_csv(audit_path, keep_default_na=False)
    invalid = set(audit["review_status"]) - REVIEW_STATUSES
    if invalid or (audit["review_status"] == "").any():
        raise ValueError("Every review_status must be pass, partial, or fail")
    usable = audit["review_status"].isin({"pass", "partial"}) & (
        audit["area_weight_usable"].astype(str).str.lower() == "true"
    )
    summary_path = output_dir / "plant_weight_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "manual_review_complete": True,
        "review_status_counts": audit["review_status"].value_counts().to_dict(),
        "usable_count": int(usable.sum()),
        "usable_rate": float(usable.mean()),
        "exit_criterion_met": bool(usable.mean() >= 0.90),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit plant-mask area weights")
    parser.add_argument("--config", default="configs/plant_weight_audit.json")
    parser.add_argument("--summarize-review", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.summarize_review:
        summary = summarize_manual_review(config["output_dir"])
    else:
        _, summary = generate_plant_weight_audit(**config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
