import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class PlantRegion:
    region_id: int
    bounding_box: tuple[int, int, int, int]
    green_area: int
    patch_box: tuple[int, int, int, int]


def vegetation_mask(image, hue_min=25, hue_max=100, saturation_min=35,
                    value_min=45, excess_green_min=5, minimum_green_area=30):
    """Return a conservative green-vegetation mask for an inside-frame BGR crop."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image.astype(np.int16))
    excess_green = green - np.maximum(red, blue)
    mask = (
        (hsv[:, :, 0] >= hue_min)
        & (hsv[:, :, 0] <= hue_max)
        & (hsv[:, :, 1] >= saturation_min)
        & (hsv[:, :, 2] >= value_min)
        & (excess_green >= excess_green_min)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_green_area:
            cleaned[labels == label] = 255
    return cleaned


def _odd_kernel_size(image, fraction):
    size = max(3, round(min(image.shape[:2]) * fraction))
    return size if size % 2 else size + 1


def find_plant_regions(image, mask, grouping_kernel_fraction=0.018,
                       minimum_group_area=200, minimum_region_green_area=200,
                       patch_padding_fraction=0.025):
    """Group nearby green fragments into plant/seedling-cluster proposals."""
    kernel_size = _odd_kernel_size(image, grouping_kernel_fraction)
    grouped = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
    height, width = image.shape[:2]
    padding = max(4, round(max(height, width) * patch_padding_fraction))
    regions = []
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] < minimum_group_area:
            continue
        group = labels == label
        green_area = int(np.count_nonzero(mask[group]))
        if green_area < minimum_region_green_area:
            continue
        y_values, x_values = np.nonzero(group)
        x1, x2 = int(x_values.min()), int(x_values.max()) + 1
        y1, y2 = int(y_values.min()), int(y_values.max()) + 1
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(width, x2 + padding), min(height, y2 + padding)
        regions.append(PlantRegion(
            region_id=len(regions) + 1,
            bounding_box=(x1, y1, x2, y2),
            green_area=green_area,
            patch_box=(px1, py1, px2, py2),
        ))
    return sorted(regions, key=lambda region: (region.patch_box[1], region.patch_box[0]))


def extract_plant_regions(image, **parameters):
    mask_keys = {
        "hue_min", "hue_max", "saturation_min", "value_min",
        "excess_green_min", "minimum_green_area",
    }
    mask_parameters = {key: value for key, value in parameters.items() if key in mask_keys}
    region_parameters = {key: value for key, value in parameters.items() if key not in mask_keys}
    mask = vegetation_mask(image, **mask_parameters)
    regions = find_plant_regions(image, mask, **region_parameters)
    return mask, regions


def _save_overlay(image, mask, regions, output_path):
    overlay = image.copy()
    selected = mask > 0
    # Magenta is deliberately used instead of green so the mask is visible on green leaves.
    magenta = np.array([255, 0, 255], dtype=np.float32)
    overlay[selected] = (
        0.35 * overlay[selected].astype(np.float32) + 0.65 * magenta
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255),
                     max(2, image.shape[1] // 700))
    thickness = max(2, image.shape[1] // 600)
    for region in regions:
        x1, y1, x2, y2 = region.patch_box
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), thickness)
        cv2.putText(
            overlay, str(region.region_id), (x1 + 4, max(24, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
    label = f"MAGENTA=vegetation regions={len(regions)} coverage={np.count_nonzero(mask) / mask.size:.3%}"
    cv2.putText(overlay, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    preview_width = min(1400, image.shape[1])
    preview_height = round(image.shape[0] * preview_width / image.shape[1])
    preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 91])


def generate_plant_region_audit(frame_audit_csv, output_dir, **parameters):
    frame_audit = pd.read_csv(frame_audit_csv)
    if set(frame_audit["split"]) - {"train", "val"}:
        raise ValueError("Plant-region development audit must not contain test images")
    if set(frame_audit["manual_status"]) != {"approved"}:
        raise ValueError("Frame audit must be manually approved before plant-region extraction")

    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    patches_dir = output_dir / "patches"
    for directory in (masks_dir, overlays_dir, patches_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in frame_audit.iterrows():
        image = cv2.imread(row["crop_path"])
        if image is None:
            raise FileNotFoundError(f"Unable to load approved frame crop: {row['crop_path']}")
        mask, regions = extract_plant_regions(image, **parameters)
        stem = Path(row["filename"]).stem
        mask_path = masks_dir / f"{stem}_vegetation.png"
        overlay_path = overlays_dir / f"{stem}_regions.jpg"
        cv2.imwrite(str(mask_path), mask)
        _save_overlay(image, mask, regions, overlay_path)

        patch_paths = []
        region_rows = []
        for region in regions:
            x1, y1, x2, y2 = region.patch_box
            patch = image[y1:y2, x1:x2]
            patch_path = patches_dir / f"{stem}_region_{region.region_id:02d}.jpg"
            cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 96])
            patch_paths.append(str(patch_path))
            region_rows.append({
                "region_id": region.region_id,
                "bounding_box": list(region.bounding_box),
                "patch_box": list(region.patch_box),
                "green_area": region.green_area,
                "patch_path": str(patch_path),
            })
        green_pixels = int(np.count_nonzero(mask))
        records.append({
            "filename": row["filename"], "split": row["split"],
            "mean_score": row["mean_score"], "score_bin": row["score_bin"],
            "frame_crop_path": row["crop_path"], "mask_path": str(mask_path),
            "overlay_path": str(overlay_path), "image_width": image.shape[1],
            "image_height": image.shape[0], "green_pixels": green_pixels,
            "green_fraction": green_pixels / mask.size, "region_count": len(regions),
            "regions_json": json.dumps(region_rows),
            "patch_paths": json.dumps(patch_paths),
            "automatic_status": "detected" if regions else "failed",
            "manual_status": "", "manual_notes": "",
        })
    audit = pd.DataFrame(records)
    audit_path = output_dir / "plant_region_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = {
        "schema_version": 1,
        "frame_audit_csv": str(Path(frame_audit_csv).resolve()),
        "sample_size": int(len(audit)),
        "parameters": parameters,
        "split_counts": audit["split"].value_counts().sort_index().to_dict(),
        "score_bin_counts": audit["score_bin"].value_counts().sort_index().to_dict(),
        "automatic_status_counts": audit["automatic_status"].value_counts().to_dict(),
        "region_count": {
            "minimum": int(audit["region_count"].min()),
            "median": float(audit["region_count"].median()),
            "maximum": int(audit["region_count"].max()),
            "total": int(audit["region_count"].sum()),
        },
        "green_fraction": {
            "minimum": float(audit["green_fraction"].min()),
            "median": float(audit["green_fraction"].median()),
            "maximum": float(audit["green_fraction"].max()),
        },
        "manual_review_complete": False,
    }
    summary_path = output_dir / "plant_region_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit, summary


def approve_plant_region_audit(output_dir, reviewer="user"):
    output_dir = Path(output_dir)
    audit_path = output_dir / "plant_region_audit.csv"
    summary_path = output_dir / "plant_region_audit_summary.json"
    if not audit_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Generate the plant-region audit before recording approval")
    audit = pd.read_csv(audit_path)
    if audit.empty or set(audit["automatic_status"]) != {"detected"}:
        raise ValueError("Approval requires at least one region in every audit image")
    audit["manual_status"] = "approved"
    audit["manual_notes"] = "Vegetation mask and high-resolution region proposals approved."
    audit.to_csv(audit_path, index=False)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary.update({
        "manual_review_complete": True,
        "manual_status_counts": {"approved": int(len(audit))},
        "reviewer": reviewer,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate an inside-frame plant-region audit")
    parser.add_argument("--config", default="configs/plant_region_audit.json")
    parser.add_argument("--approve-all", action="store_true")
    parser.add_argument("--reviewer", default="user")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.approve_all:
        summary = approve_plant_region_audit(config["output_dir"], reviewer=args.reviewer)
    else:
        _, summary = generate_plant_region_audit(**config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
