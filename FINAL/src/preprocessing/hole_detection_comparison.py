import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.direct_damage_audit import analyze_frame_damage, draw_damage_overlay
from src.preprocessing.frame_crop import crop_frame_interior, detect_frame, read_image_oriented
from src.preprocessing.plant_weight_audit import _write_contact_sheet


REVIEW_COLUMNS = [
    "new_holes_better", "pitting_still_usable", "soil_false_holes",
    "review_status", "review_notes",
]


def _write_review_pages(paths, output_dir, images_per_page=4, page_width=2400):
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(0, len(paths), images_per_page):
        page_images = []
        for path in paths[page_index:page_index + images_per_page]:
            image = cv2.imread(str(path))
            if image is None:
                continue
            scale = min(1.0, page_width / image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if image.shape[1] < page_width:
                right = page_width - image.shape[1]
                image = cv2.copyMakeBorder(image, 0, 0, 0, right, cv2.BORDER_CONSTANT)
            page_images.append(image)
        if not page_images:
            continue
        page = np.vstack(page_images)
        page_number = page_index // images_per_page + 1
        cv2.imwrite(
            str(output_dir / f"review_page_{page_number:02d}.jpg"), page,
            [cv2.IMWRITE_JPEG_QUALITY, 96],
        )


def generate_comparison(config):
    source = pd.read_csv(config["source_audit"])
    source = source[source["split"].isin(["train", "val"])].copy()
    source["priority"] = source["expert_score"] - source["damage_percentage"]
    selected = source.sort_values(
        ["hole_count", "priority"], ascending=[True, False]
    ).head(config["sample_size"])
    manifest = pd.read_csv(config["manifest"]).set_index("filename")
    output_dir = Path(config["output_dir"])
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    records, paths = [], []

    for _, old_row in selected.iterrows():
        row = manifest.loc[old_row["filename"]]
        image = read_image_oriented(row["image_path"])
        detection = detect_frame(image, working_width=config["working_width"])
        if detection.status != "detected":
            continue
        frame, _ = crop_frame_interior(
            image, detection.corners, inset_fraction=config["interior_inset_fraction"]
        )
        old = analyze_frame_damage(
            frame, config["plant_parameters"], config["hsv_bounds"], "brightness"
        )
        new = analyze_frame_damage(
            frame, config["plant_parameters"], config["hsv_bounds"],
            "soil_similarity", config["soil_lab_distance"]
        )
        old_metrics, old_leaf, old_holes, old_pitting = old
        new_metrics, new_leaf, new_holes, new_pitting = new
        old_panel = draw_damage_overlay(
            frame, old_leaf, old_holes, old_pitting,
            f"OLD holes={old_metrics['hole_count']} pitting={old_metrics['pitting_count']}"
        )
        new_panel = draw_damage_overlay(
            frame, new_leaf, new_holes, new_pitting,
            f"NEW holes={new_metrics['hole_count']} pitting={new_metrics['pitting_count']}"
        )
        height = min(old_panel.shape[0], new_panel.shape[0])
        comparison = np.hstack([old_panel[:height], new_panel[:height]])
        path = overlays_dir / f"{Path(old_row['filename']).stem}_old_new.jpg"
        cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(path)
        records.append({
            "filename": old_row["filename"], "split": old_row["split"],
            "expert_score": old_row["expert_score"],
            "old_hole_count": old_metrics["hole_count"],
            "new_hole_count": new_metrics["hole_count"],
            "old_pitting_count": old_metrics["pitting_count"],
            "new_pitting_count": new_metrics["pitting_count"],
            "old_damage_percentage": old_metrics["damage_percentage"],
            "new_damage_percentage": new_metrics["damage_percentage"],
            "comparison_path": str(path),
            "new_holes_better": "", "pitting_still_usable": "",
            "soil_false_holes": "", "review_status": "", "review_notes": "",
        })
    audit_path = output_dir / "hole_detection_comparison.csv"
    audit = pd.DataFrame(records)
    if audit_path.is_file():
        previous = pd.read_csv(audit_path, keep_default_na=False)
        if "filename" in previous and set(REVIEW_COLUMNS).issubset(previous.columns):
            reviews = previous[["filename", *REVIEW_COLUMNS]].drop_duplicates("filename")
            audit = audit.drop(columns=REVIEW_COLUMNS).merge(reviews, on="filename", how="left")
            audit[REVIEW_COLUMNS] = audit[REVIEW_COLUMNS].fillna("")
    audit.to_csv(audit_path, index=False)
    _write_contact_sheet(paths, output_dir / "hole_detection_comparison_sheet.jpg", columns=2)
    _write_review_pages(
        paths, output_dir / "review_pages",
        images_per_page=config.get("images_per_review_page", 4),
        page_width=config.get("review_page_width", 2400),
    )
    summary = {
        "sample_size": len(audit), "test_images": int((audit["split"] == "test").sum()),
        "soil_lab_distance": config["soil_lab_distance"],
        "manual_review_complete": False,
    }
    (output_dir / "hole_detection_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit, summary


def summarize_review(output_dir):
    output_dir = Path(output_dir)
    audit = pd.read_csv(output_dir / "hole_detection_comparison.csv", keep_default_na=False)
    if (audit["review_status"] == "").any() or set(audit["review_status"]) - {"better", "same", "worse"}:
        raise ValueError("Every review_status must be better, same, or worse")
    better = audit["review_status"] == "better"
    safe = audit["pitting_still_usable"].astype(str).str.lower() == "true"
    false_holes = audit["soil_false_holes"].astype(str).str.lower() == "true"
    summary_path = output_dir / "hole_detection_comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "manual_review_complete": True,
        "review_status_counts": audit["review_status"].value_counts().to_dict(),
        "better_count": int(better.sum()),
        "pitting_preserved_count": int(safe.sum()),
        "soil_false_hole_count": int(false_holes.sum()),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare old and soil-aware hole detection")
    parser.add_argument("--config", default="configs/hole_detection_comparison.json")
    parser.add_argument("--summarize-review", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.summarize_review:
        summary = summarize_review(config["output_dir"])
    else:
        _, summary = generate_comparison(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
