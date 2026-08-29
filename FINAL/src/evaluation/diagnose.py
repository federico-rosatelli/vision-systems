import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import CSFBDataset
from src.models.dinov3_regressor import DINOv3Regressor


ALLOWED_SPLITS = ("train", "val")


def summarize_predictions(y_true, y_pred, constant_predictions):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {
        "samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic),
        "targets": {
            "mean": float(y_true.mean()), "std": float(y_true.std(ddof=1)),
            "minimum": float(y_true.min()), "maximum": float(y_true.max()),
        },
        "predictions": {
            "mean": float(y_pred.mean()), "std": float(y_pred.std(ddof=1)),
            "minimum": float(y_pred.min()), "maximum": float(y_pred.max()),
        },
        "constant_baselines": {},
    }
    for name, prediction in constant_predictions.items():
        constant = np.full_like(y_true, prediction)
        result["constant_baselines"][name] = {
            "prediction": float(prediction),
            "mae": float(mean_absolute_error(y_true, constant)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, constant))),
        }
    return result


def diagnose(manifest_path, checkpoint_path, output_dir, splits=ALLOWED_SPLITS,
             batch_size=32, num_workers=4):
    invalid = set(splits) - set(ALLOWED_SPLITS)
    if invalid:
        raise ValueError(f"Diagnostics may not access the frozen test split: {sorted(invalid)}")

    manifest = pd.read_csv(manifest_path)
    train_targets = manifest.loc[manifest["split"] == "train", "mean_score"]
    constants = {"train_mean": train_targets.mean(), "train_median": train_targets.median()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DINOv3Regressor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    image_size = checkpoint["model_config"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_validation_mae": checkpoint.get("best_val_mae"),
        "splits": {},
    }
    for split in splits:
        dataset = CSFBDataset(manifest_path, split=split, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        targets, predictions, groups = [], [], []
        with torch.no_grad():
            for images, batch_targets, batch_groups in loader:
                batch_predictions = model(images.to(device)).cpu().numpy()
                targets.extend(batch_targets.numpy())
                predictions.extend(batch_predictions)
                groups.extend(batch_groups)
        report["splits"][split] = summarize_predictions(targets, predictions, constants)
        pd.DataFrame({
            "filename": dataset.df["filename"].tolist(),
            "image_path": dataset.df["image_path"].tolist(),
            "plot_group": groups, "true_score": targets, "pred_score": predictions,
            "absolute_error": np.abs(np.asarray(targets) - np.asarray(predictions)),
        }).to_csv(output_dir / f"{split}_predictions.csv", index=False)

    with (output_dir / "diagnostic_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose a checkpoint without accessing test data")
    parser.add_argument("--manifest", default="outputs/tables/baseline_manifest_split.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.manifest, args.checkpoint, args.output_dir,
                              num_workers=args.num_workers), indent=2))
