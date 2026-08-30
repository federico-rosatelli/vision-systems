import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


SCORE_BINS = (-0.001, 3.0, 8.0, 15.0, float("inf"))
SCORE_LABELS = ("low_0_3", "medium_3_8", "high_8_15", "very_high_15_plus")


@dataclass
class FrameDetection:
    corners: np.ndarray | None
    confidence: float
    status: str
    reason: str


def read_image_oriented(path):
    """Read an RGB image while applying its EXIF orientation."""
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _line_angle(line):
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _fit_line(segments):
    points = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], dtype=float), np.array([vx, vy], dtype=float)


def _intersection(line_a, line_b):
    point_a, vector_a = line_a
    point_b, vector_b = line_b
    matrix = np.column_stack((vector_a, -vector_b))
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-6:
        raise ValueError("Frame side lines are parallel")
    scale = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + scale * vector_a


def _metal_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Galvanized frame bars are bright, low-to-medium saturation blue/gray.
    mask = cv2.inRange(hsv, (65, 0, 105), (145, 105, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _cluster_outer_lines(lines, width, height):
    horizontal = []
    vertical = []
    for segment in lines:
        x1, y1, x2, y2 = map(float, segment)
        length = np.hypot(x2 - x1, y2 - y1)
        angle = _line_angle(segment)
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        if (
            min(abs(angle), abs(abs(angle) - 180)) <= 15
            and length >= 0.22 * width
            and 0.18 * width <= midpoint_x <= 0.82 * width
            and 0.10 * height <= midpoint_y <= 0.85 * height
            and min(x1, x2) <= 0.52 * width <= max(x1, x2)
        ):
            horizontal.append(segment)
        if (
            abs(abs(angle) - 90) <= 15
            and length >= 0.28 * height
            and 0.12 * width <= midpoint_x <= 0.88 * width
            and 0.12 * height <= midpoint_y <= 0.82 * height
            and min(y1, y2) <= 0.48 * height <= max(y1, y2)
        ):
            vertical.append(segment)
    if len(horizontal) < 2 or len(vertical) < 2:
        raise ValueError(
            f"insufficient side candidates: horizontal={len(horizontal)}, vertical={len(vertical)}"
        )

    horizontal = np.asarray(horizontal)
    vertical = np.asarray(vertical)
    h_positions = horizontal[:, [1, 3]].mean(axis=1)
    v_positions = vertical[:, [0, 2]].mean(axis=1)
    top_position, bottom_position = h_positions.min(), h_positions.max()
    left_position, right_position = v_positions.min(), v_positions.max()
    h_tolerance = 0.035 * height
    v_tolerance = 0.035 * width

    top = horizontal[np.abs(h_positions - top_position) <= h_tolerance]
    bottom = horizontal[np.abs(h_positions - bottom_position) <= h_tolerance]
    left = vertical[np.abs(v_positions - left_position) <= v_tolerance]
    right = vertical[np.abs(v_positions - right_position) <= v_tolerance]
    return _fit_line(top), _fit_line(bottom), _fit_line(left), _fit_line(right)


def _validate_corners(corners, width, height):
    if not np.isfinite(corners).all():
        return 0.0, "non-finite corner"
    normalized = corners / np.array([width, height], dtype=float)
    if (normalized < -0.02).any() or (normalized > 1.02).any():
        return 0.0, "corner outside image"

    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    polygon_area = abs(cv2.contourArea(corners.astype(np.float32)))
    area_fraction = polygon_area / (width * height)
    center = corners.mean(axis=0) / np.array([width, height])

    checks = [
        (0.28 <= min(top_width, bottom_width) / width <= 0.55, "invalid frame width"),
        (0.34 <= min(left_height, right_height) / height <= 0.72, "invalid frame height"),
        (0.12 <= area_fraction <= 0.34, "invalid frame area"),
        (0.34 <= center[0] <= 0.66, "invalid horizontal center"),
        (0.30 <= center[1] <= 0.68, "invalid vertical center"),
        (max(top_width, bottom_width) / min(top_width, bottom_width) <= 1.35, "width imbalance"),
        (max(left_height, right_height) / min(left_height, right_height) <= 1.35, "height imbalance"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        return 0.0, "; ".join(failures)

    expected_area = 0.20
    area_score = max(0.0, 1.0 - abs(area_fraction - expected_area) / 0.14)
    center_score = max(0.0, 1.0 - np.linalg.norm(center - np.array([0.5, 0.49])) / 0.30)
    balance_score = min(top_width, bottom_width) / max(top_width, bottom_width)
    balance_score *= min(left_height, right_height) / max(left_height, right_height)
    return float(np.clip((area_score + center_score + balance_score) / 3, 0, 1)), ""


def detect_frame(image, working_width=1000):
    """Detect outer frame corners ordered top-left, top-right, bottom-right, bottom-left."""
    height, width = image.shape[:2]
    scale = working_width / width
    working_height = max(1, round(height * scale))
    working = cv2.resize(image, (working_width, working_height), interpolation=cv2.INTER_AREA)
    mask = _metal_mask(working)
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 720, threshold=45,
        minLineLength=round(0.14 * working_width), maxLineGap=round(0.035 * working_width),
    )
    if lines is None:
        return FrameDetection(None, 0.0, "failed", "no Hough lines")
    try:
        top, bottom, left, right = _cluster_outer_lines(
            lines[:, 0, :], working_width, working_height
        )
        corners = np.asarray([
            _intersection(top, left), _intersection(top, right),
            _intersection(bottom, right), _intersection(bottom, left),
        ])
        confidence, reason = _validate_corners(corners, working_width, working_height)
        if reason:
            return FrameDetection(None, confidence, "failed", reason)
        corners /= scale
        return FrameDetection(corners.astype(np.float32), confidence, "detected", "")
    except (ValueError, np.linalg.LinAlgError) as error:
        return FrameDetection(None, 0.0, "failed", str(error))


def inset_polygon(corners, fraction=0.025):
    center = corners.mean(axis=0)
    return corners + fraction * (center - corners)


def _order_destination(corners):
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    width = max(1, round(max(top_width, bottom_width)))
    height = max(1, round(max(left_height, right_height)))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return destination, width, height


def crop_frame_interior(image, corners, inset_fraction=0.025):
    interior = inset_polygon(corners.astype(np.float32), inset_fraction)
    destination, width, height = _order_destination(interior)
    transform = cv2.getPerspectiveTransform(interior, destination)
    return cv2.warpPerspective(image, transform, (width, height)), interior


def select_audit_rows(manifest, sample_size=30, seed=42):
    rows = pd.read_csv(manifest)
    rows = rows[rows["split"].isin(["train", "val"])].copy()
    if rows.empty:
        raise ValueError("Manifest has no train/validation rows")
    rows["score_bin"] = pd.cut(
        rows["mean_score"], SCORE_BINS, labels=SCORE_LABELS, include_lowest=True
    )
    strata = list(rows.groupby(["split", "score_bin"], observed=True))
    if sample_size < len(strata):
        raise ValueError("sample_size must cover every non-empty split/score stratum")

    rng = np.random.default_rng(seed)
    selected_indices = []
    split_targets = {
        "train": sample_size // 2,
        "val": sample_size - sample_size // 2,
    }
    for split, split_target in split_targets.items():
        split_strata = [(key, group) for key, group in strata if key[0] == split]
        base, extra = divmod(split_target, len(split_strata))
        for stratum_index, (_, group) in enumerate(split_strata):
            take = min(len(group), base + (stratum_index < extra))
            chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
            selected_indices.extend(map(int, chosen))
        # Fill a split quota if an unusually small stratum could not supply its share.
        current = sum(rows.loc[index, "split"] == split for index in selected_indices)
        if current < split_target:
            available = rows[(rows["split"] == split) & ~rows.index.isin(selected_indices)]
            chosen = rng.choice(
                available.index.to_numpy(), size=split_target - current, replace=False
            )
            selected_indices.extend(map(int, chosen))
    selected = rows.loc[selected_indices].copy()
    return selected.sort_values(["score_bin", "split", "mean_score", "filename"]).reset_index(drop=True)


def _save_overlay(image, detection, output_path):
    overlay = image.copy()
    if detection.corners is not None:
        polygon = np.round(detection.corners).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, (0, 255, 0), max(4, image.shape[1] // 700))
        for index, point in enumerate(polygon):
            cv2.circle(overlay, tuple(point), max(8, image.shape[1] // 300), (0, 0, 255), -1)
            cv2.putText(overlay, str(index + 1), tuple(point + [12, -12]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    label = f"{detection.status} confidence={detection.confidence:.3f} {detection.reason}"
    cv2.putText(overlay, label, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 0) if detection.status == "detected" else (0, 0, 255), 3)
    preview_width = min(1600, image.shape[1])
    preview_height = round(image.shape[0] * preview_width / image.shape[1])
    preview = cv2.resize(overlay, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def generate_frame_audit(manifest, output_dir, sample_size=30, seed=42,
                         working_width=1000, interior_inset_fraction=0.025):
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    masks_dir = output_dir / "masks"
    for directory in (overlays_dir, crops_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = select_audit_rows(manifest, sample_size=sample_size, seed=seed)
    records = []
    for _, row in selected.iterrows():
        image = read_image_oriented(row["image_path"])
        detection = detect_frame(image, working_width=working_width)
        stem = Path(row["filename"]).stem
        overlay_path = overlays_dir / f"{stem}_overlay.jpg"
        _save_overlay(image, detection, overlay_path)
        crop_path = ""
        mask_path = ""
        corners_json = ""
        if detection.corners is not None:
            crop, interior = crop_frame_interior(
                image, detection.corners, inset_fraction=interior_inset_fraction
            )
            crop_path = str(crops_dir / f"{stem}_crop.jpg")
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(interior).astype(np.int32)], 255)
            mask_path = str(masks_dir / f"{stem}_mask.png")
            cv2.imwrite(mask_path, mask)
            corners_json = json.dumps(detection.corners.round(2).tolist())
        records.append({
            "filename": row["filename"], "image_path": row["image_path"],
            "split": row["split"], "mean_score": row["mean_score"],
            "score_bin": str(row["score_bin"]),
            "automatic_status": detection.status,
            "confidence": detection.confidence, "failure_reason": detection.reason,
            "corners_xy": corners_json, "overlay_path": str(overlay_path),
            "crop_path": crop_path, "mask_path": mask_path,
            "manual_status": "", "manual_notes": "",
        })
    audit = pd.DataFrame(records)
    audit.to_csv(output_dir / "frame_audit.csv", index=False)
    summary = {
        "schema_version": 1,
        "manifest": str(Path(manifest)),
        "sample_size": int(len(audit)), "seed": seed,
        "working_width": working_width,
        "interior_inset_fraction": interior_inset_fraction,
        "split_counts": audit["split"].value_counts().sort_index().to_dict(),
        "score_bin_counts": audit["score_bin"].value_counts().sort_index().to_dict(),
        "automatic_status_counts": audit["automatic_status"].value_counts().to_dict(),
        "mean_confidence": float(audit["confidence"].mean()),
        "manual_review_complete": False,
    }
    with (output_dir / "frame_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit, summary


def approve_frame_audit(output_dir, reviewer="user"):
    """Record explicit human approval after reviewing both audit contact sheets."""
    output_dir = Path(output_dir)
    audit_path = output_dir / "frame_audit.csv"
    summary_path = output_dir / "frame_audit_summary.json"
    if not audit_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Generate the frame audit before recording approval")
    audit = pd.read_csv(audit_path)
    if audit.empty or set(audit["automatic_status"]) != {"detected"}:
        raise ValueError("Approval requires a non-empty audit with all frames detected")
    audit["manual_status"] = "approved"
    audit["manual_notes"] = "Outer frame polygon and rectified inside-frame crop approved."
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
    parser = argparse.ArgumentParser(description="Generate a train/validation frame-crop audit")
    parser.add_argument("--config", default="configs/frame_audit.json")
    parser.add_argument("--approve-all", action="store_true",
                        help="Record human approval for every generated audit row")
    parser.add_argument("--reviewer", default="user")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.approve_all:
        summary = approve_frame_audit(config["output_dir"], reviewer=args.reviewer)
    else:
        _, summary = generate_frame_audit(**config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
