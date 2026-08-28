import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CURATED_RELATIVE_PATH = Path(
    "RSFB-Phenotyping_training_set/RSFB-Phenotyping_training_set/"
    "RSFB-Phenotyping_training_set_scores.csv"
)
IMAGE_RELATIVE_DIR = Path(
    "RSFB-Phenotyping_training_set/RSFB-Phenotyping_training_set"
)
METADATA_FILENAME = "2025_10_21_RSFB-Phenotyping_GG1_scores.csv"


class BaselineManifestError(ValueError):
    pass


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_filename(value):
    filename = Path(str(value).strip()).name.lower()
    if not filename.endswith((".jpg", ".jpeg")):
        filename += ".jpg"
    return filename


def _required(row, column, source):
    value = str(row.get(column, "")).strip()
    if not value or value.lower() == "nan":
        raise BaselineManifestError(f"Missing {column!r} in {source}")
    return value


def build_baseline_manifest(
    raw_dir,
    output_manifest,
    output_audit,
    curated_csv=None,
    metadata_csv=None,
):
    """Build the read-only, supervisor-curated 470-image BBCH10 manifest."""
    raw_dir = Path(raw_dir)
    curated_path = Path(curated_csv) if curated_csv else raw_dir / CURATED_RELATIVE_PATH
    metadata_path = Path(metadata_csv) if metadata_csv else raw_dir / METADATA_FILENAME
    image_dir = raw_dir / IMAGE_RELATIVE_DIR

    if not raw_dir.is_dir():
        raise BaselineManifestError(f"Dataset directory not found: {raw_dir}")
    for source_path in (curated_path, metadata_path):
        if not source_path.is_file():
            raise BaselineManifestError(f"Required source CSV not found: {source_path}")

    curated_rows = _read_csv(curated_path)
    metadata_rows = _read_csv(metadata_path)

    metadata_by_filename = {}
    duplicate_metadata = []
    for row in metadata_rows:
        filename = _normalize_filename(_required(row, "Filename", metadata_path.name))
        if filename in metadata_by_filename:
            duplicate_metadata.append(filename)
        metadata_by_filename[filename] = row

    manifest = []
    seen_filenames = set()
    duplicate_curated = []
    missing_metadata = []
    missing_images = []
    score_mismatches = []

    for curated in curated_rows:
        filename = _normalize_filename(_required(curated, "Filename", curated_path.name))
        if filename in seen_filenames:
            duplicate_curated.append(filename)
            continue
        seen_filenames.add(filename)

        metadata = metadata_by_filename.get(filename)
        if metadata is None:
            missing_metadata.append(filename)
            continue

        score_jlu = float(_required(curated, "Score_JLU", filename))
        score_gau = float(_required(curated, "Score_GAU", filename))
        supplied_mean = float(_required(curated, "mean_score", filename))
        mean_score = (score_jlu + score_gau) / 2.0
        disagreement = abs(score_jlu - score_gau)

        metadata_jlu = float(_required(metadata, "Score_JLU", filename))
        metadata_gau = float(_required(metadata, "Score_GAU", filename))
        if metadata_jlu != score_jlu or metadata_gau != score_gau or supplied_mean != mean_score:
            score_mismatches.append(filename)

        image_path = image_dir / filename
        if not image_path.is_file():
            missing_images.append(filename)

        plot_group = _required(metadata, "QR-Code", filename)
        genotype = _required(metadata, "Genotyp", filename)
        manifest.append(
            {
                "filename": filename,
                "image_path": str(image_path),
                "plot_group": plot_group,
                "genotype": genotype,
                "plot_number": _required(metadata, "Plotnr", filename),
                "r4s_number": str(metadata.get("R4Snr", "")).strip(),
                "row": _required(metadata, "row", filename),
                "column": _required(metadata, "col", filename),
                "score_jlu": score_jlu,
                "score_gau": score_gau,
                "mean_score": mean_score,
                "disagreement": disagreement,
                "location": "Gross-Gerau",
                "experiment": "Insects",
                "sampling_date": "2025-10-21",
                "bbch": "BBCH10",
                "source_curated_csv": str(curated_path),
                "source_metadata_csv": str(metadata_path),
            }
        )

    disagreements = [row["disagreement"] for row in manifest]
    duplicate_metadata_used = sorted(set(duplicate_metadata) & seen_filenames)
    audit = {
        "status": "pass",
        "dataset_mode": "read_only",
        "raw_dir": str(raw_dir),
        "curated_source_rows": len(curated_rows),
        "manifest_rows": len(manifest),
        "unique_filenames": len({row["filename"] for row in manifest}),
        "unique_image_paths": len({row["image_path"] for row in manifest}),
        "unique_plot_groups": len({row["plot_group"] for row in manifest}),
        "unique_genotypes": len({row["genotype"] for row in manifest}),
        "missing_metadata": missing_metadata,
        "missing_images": missing_images,
        "duplicate_curated_filenames": duplicate_curated,
        "duplicate_metadata_filenames": sorted(set(duplicate_metadata)),
        "duplicate_metadata_filenames_used": duplicate_metadata_used,
        "score_mismatches": score_mismatches,
        "disagreement_lt_5": sum(value < 5.0 for value in disagreements),
        "disagreement_eq_5": sum(value == 5.0 for value in disagreements),
        "disagreement_gt_5": sum(value > 5.0 for value in disagreements),
        "max_disagreement": max(disagreements) if disagreements else None,
        "images_per_plot": dict(sorted(Counter(row["plot_group"] for row in manifest).items())),
        "note": (
            "The authoritative curated file contains 470 images: 443 have disagreement "
            "below 5 and 27 have disagreement exactly 5. The complete curated set is preserved."
        ),
    }

    failures = []
    if len(curated_rows) != 470:
        failures.append(f"Expected 470 curated rows, found {len(curated_rows)}")
    if len(manifest) != 470:
        failures.append(f"Expected 470 manifest rows, built {len(manifest)}")
    for label, values in (
        ("duplicate curated filenames", duplicate_curated),
        ("duplicate metadata filenames used by baseline", duplicate_metadata_used),
        ("missing metadata", missing_metadata),
        ("missing images", missing_images),
        ("score mismatches", score_mismatches),
    ):
        if values:
            failures.append(f"Found {len(values)} {label}")
    if any(value > 5.0 for value in disagreements):
        failures.append("Found disagreement above 5 percentage points")

    if failures:
        audit["status"] = "fail"
        audit["failures"] = failures

    output_manifest = Path(output_manifest)
    output_audit = Path(output_audit)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(manifest[0]) if manifest else []
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    with output_audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if failures:
        raise BaselineManifestError("; ".join(failures))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the curated CSFB baseline manifest")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-manifest", default="outputs/tables/baseline_manifest.csv")
    parser.add_argument("--output-audit", default="outputs/tables/baseline_audit.json")
    args = parser.parse_args()
    result = build_baseline_manifest(args.raw_dir, args.output_manifest, args.output_audit)
    print(json.dumps({key: value for key, value in result.items() if key != "images_per_plot"}, indent=2))
