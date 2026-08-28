import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _metrics(targets, prediction):
    errors = [target - prediction for target in targets]
    return {
        "prediction": prediction,
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def evaluate_constant_baselines(manifest_path, output_path):
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [float(row["mean_score"]) for row in rows if row["split"] == "train"]
    test = [float(row["mean_score"]) for row in rows if row["split"] == "test"]
    if not train or not test:
        raise ValueError("Manifest must contain non-empty train and test splits")

    results = {
        "train_samples": len(train),
        "test_samples": len(test),
        "train_target_mean": statistics.mean(train),
        "train_target_median": statistics.median(train),
        "mean_predictor": _metrics(test, statistics.mean(train)),
        "median_predictor": _metrics(test, statistics.median(train)),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate constant CSFB baselines")
    parser.add_argument("--manifest", default="outputs/tables/baseline_manifest_split.csv")
    parser.add_argument("--output", default="outputs/runs/baseline_references/metrics.json")
    args = parser.parse_args()
    print(json.dumps(evaluate_constant_baselines(args.manifest, args.output), indent=2))
