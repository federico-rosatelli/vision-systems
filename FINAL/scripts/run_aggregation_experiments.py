import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patch_dataset import get_patch_dataloaders, validate_fixed_manifest
from src.evaluation.evaluate_mil import pairwise_ranking_accuracy
from src.models.mil_model import DINOv3MILRegressor
from src.training.train_mil import train_mil_model


def experiment_specs(config):
    return [
        {
            "aggregation": aggregation,
            "seed": seed,
            "run_name": f"aggregation_{aggregation}_seed{seed}",
        }
        for aggregation in config["aggregations"]
        for seed in config["seeds"]
    ]


def best_validation_row(log_path):
    log = pd.read_csv(log_path)
    return log.loc[log["val_mae"].idxmin()]


def validation_diagnostics(config, run_name):
    checkpoint_path = Path(config["output_dir"]) / run_name / "checkpoints" / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DINOv3MILRegressor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, val_loader, _ = get_patch_dataloaders(
        manifest_path=config["manifest"], cache_path=config["cache_path"],
        batch_size=config["batch_size"], num_workers=0, high_quality_only=False,
    )
    targets, predictions = [], []
    with torch.no_grad():
        for features, areas, batch_targets, _ in val_loader:
            predictions.extend(model(features, areas).squeeze(1).numpy().tolist())
            targets.extend(batch_targets.numpy().tolist())
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    bins = [(-np.inf, 3, "0-3"), (3, 8, "3-8"), (8, 15, "8-15"), (15, np.inf, "15+")]
    bin_mae = {
        label: float(np.mean(np.abs(targets[mask] - predictions[mask])))
        for low, high, label in bins
        if (mask := ((targets >= low) & (targets < high))).any()
    }
    constants = checkpoint["constants"]
    return {
        "val_rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std(ddof=1)),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "target_std": float(targets.std(ddof=1)),
        "pairwise_accuracy_gap5": pairwise_ranking_accuracy(targets, predictions, 5)["accuracy"],
        "pairwise_accuracy_gap10": pairwise_ranking_accuracy(targets, predictions, 10)["accuracy"],
        "pairwise_accuracy_gap20": pairwise_ranking_accuracy(targets, predictions, 20)["accuracy"],
        "score_bin_mae": json.dumps(bin_mae, sort_keys=True),
        "constant_mean_mae": constants["mean_mae"],
        "constant_median_mae": constants["median_mae"],
    }


def run(config_path, dry_run=False, summarize_existing=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_fixed_manifest(config["manifest"])
    specs = experiment_specs(config)
    if dry_run:
        for spec in specs:
            print(spec["run_name"])
        return None

    records = []
    for spec in specs:
        if not summarize_existing:
            train_mil_model(
                manifest=config["manifest"], cache_path=config["cache_path"],
                epochs=config["epochs"], batch_size=config["batch_size"],
                lr=config["learning_rate"], loss=config["loss"],
                patience=config["patience"], out_dir=config["output_dir"],
                run_name=spec["run_name"], seed=spec["seed"], num_workers=0,
                high_quality_only=False, model_name=config["model_name"],
                head_width=config["head_width"], dropout_p=config["dropout"],
                attn_L=config["attention_width"], weights_path=config["weights_path"],
                aggregation=spec["aggregation"], training_mode="regression",
            )
        log_path = Path(config["output_dir"]) / spec["run_name"] / "logs" / "training_log.csv"
        best = best_validation_row(log_path)
        records.append({
            **spec,
            "best_epoch": int(best["epoch"]),
            "val_mae": float(best["val_mae"]),
            "val_pearson": float(best["val_pearson"]),
            "val_spearman": float(best["val_spearman"]),
            **validation_diagnostics(config, spec["run_name"]),
        })

    results = pd.DataFrame(records)
    summary_path = Path(config["summary_csv"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_path, index=False)
    aggregate = results.groupby("aggregation").agg(
        val_mae_mean=("val_mae", "mean"), val_mae_std=("val_mae", "std"),
        val_spearman_mean=("val_spearman", "mean"),
        val_spearman_std=("val_spearman", "std"),
    ).sort_values("val_mae_mean")
    aggregate.to_csv(summary_path.with_name("aggregation_validation_summary.csv"))
    print(aggregate.to_string())
    return results


def main():
    parser = argparse.ArgumentParser(description="Run validation-only aggregation experiments")
    parser.add_argument("--config", default="configs/aggregation_experiments.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run, summarize_existing=args.summarize_existing)


if __name__ == "__main__":
    main()
