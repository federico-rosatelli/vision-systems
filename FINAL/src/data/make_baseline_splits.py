import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


class BaselineSplitError(ValueError):
    pass


def _score_bin(score):
    if score <= 5:
        return "0-5"
    if score <= 10:
        return "5-10"
    if score <= 20:
        return "10-20"
    if score <= 40:
        return "20-40"
    return "40-100"


def _largest_remainder_counts(total, ratios):
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(ratios, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def create_baseline_splits(
    manifest_path,
    output_manifest,
    output_audit,
    output_groups,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
):
    """Create deterministic score-stratified splits without breaking plot groups."""
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise BaselineSplitError("Split ratios must be positive and sum to 1")

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise BaselineSplitError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 470:
        raise BaselineSplitError(f"Expected 470 baseline rows, found {len(rows)}")

    by_group = defaultdict(list)
    for row in rows:
        group = str(row.get("plot_group", "")).strip()
        if not group or group.lower() == "unknown":
            raise BaselineSplitError(f"Invalid plot group for {row.get('filename')}")
        by_group[group].append(row)

    groups_by_bin = defaultdict(list)
    group_mean_scores = {}
    for group, group_rows in by_group.items():
        mean_score = sum(float(row["mean_score"]) for row in group_rows) / len(group_rows)
        group_mean_scores[group] = mean_score
        groups_by_bin[_score_bin(mean_score)].append(group)

    rng = random.Random(seed)
    assignments = {}
    for score_bin in sorted(groups_by_bin):
        groups = sorted(groups_by_bin[score_bin])
        rng.shuffle(groups)
        counts = _largest_remainder_counts(len(groups), ratios)
        cursor = 0
        for split in ("train", "val", "test"):
            selected = groups[cursor : cursor + counts[split]]
            assignments.update({group: split for group in selected})
            cursor += counts[split]

    output_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["split"] = assignments[row["plot_group"]]
        output_rows.append(enriched)

    split_images = Counter(row["split"] for row in output_rows)
    split_groups = Counter(assignments.values())
    split_scores = {}
    for split in ratios:
        values = [float(row["mean_score"]) for row in output_rows if row["split"] == split]
        split_scores[split] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    groups_by_split = {
        split: sorted(group for group, assigned in assignments.items() if assigned == split)
        for split in ratios
    }
    leakage = sorted(
        (set(groups_by_split["train"]) & set(groups_by_split["val"]))
        | (set(groups_by_split["train"]) & set(groups_by_split["test"]))
        | (set(groups_by_split["val"]) & set(groups_by_split["test"]))
    )
    duplicate_filenames = [
        filename for filename, count in Counter(row["filename"] for row in output_rows).items() if count > 1
    ]
    unassigned = [row["filename"] for row in output_rows if row["split"] not in ratios]

    audit = {
        "status": "pass",
        "seed": seed,
        "target_ratios": ratios,
        "total_images": len(output_rows),
        "total_plot_groups": len(by_group),
        "image_counts": dict(split_images),
        "image_ratios": {split: split_images[split] / len(output_rows) for split in ratios},
        "plot_group_counts": dict(split_groups),
        "score_summary": split_scores,
        "plot_group_leakage": leakage,
        "duplicate_filenames": duplicate_filenames,
        "unassigned_images": unassigned,
        "score_bin_group_counts": {
            score_bin: dict(Counter(assignments[group] for group in groups))
            for score_bin, groups in sorted(groups_by_bin.items())
        },
    }

    failures = []
    if leakage:
        failures.append(f"Found {len(leakage)} plot groups in multiple splits")
    if duplicate_filenames:
        failures.append(f"Found {len(duplicate_filenames)} duplicate filenames")
    if unassigned:
        failures.append(f"Found {len(unassigned)} unassigned images")
    for split, target in ratios.items():
        actual = audit["image_ratios"][split]
        if abs(actual - target) > 0.03:
            failures.append(f"{split} image ratio {actual:.3f} is too far from {target:.3f}")
    if failures:
        audit["status"] = "fail"
        audit["failures"] = failures

    output_manifest = Path(output_manifest)
    output_audit = Path(output_audit)
    output_groups = Path(output_groups)
    for path in (output_manifest, output_audit, output_groups):
        path.parent.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with output_audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with output_groups.open("w", encoding="utf-8") as handle:
        json.dump(groups_by_split, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if failures:
        raise BaselineSplitError("; ".join(failures))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the curated CSFB baseline by plot group")
    parser.add_argument("--manifest", default="outputs/tables/baseline_manifest.csv")
    parser.add_argument("--output-manifest", default="outputs/tables/baseline_manifest_split.csv")
    parser.add_argument("--output-audit", default="outputs/tables/baseline_split_audit.json")
    parser.add_argument("--output-groups", default="outputs/tables/baseline_split_groups.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_baseline_splits(
        args.manifest, args.output_manifest, args.output_audit, args.output_groups, args.seed
    )
    print(json.dumps(result, indent=2))
