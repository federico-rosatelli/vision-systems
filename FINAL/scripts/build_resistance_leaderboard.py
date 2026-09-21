"""
Build a plot-level CSFB damage/resistance leaderboard from the frozen
470-image baseline manifest, using cached DINOv3 patch-feature bags and a
trained MIL checkpoint.

Replaces the previous outputs/tables/resistance_leaderboard.csv, which was
built from a different, unfrozen manifest and contained an invalid "unknown"
plot group. Within the frozen manifest, only 7 of 217 genotypes have 2 or
more plot-group replicates, so a genotype-level ranking is only reported as
a small, explicitly caveated secondary table; the primary output is
plot-level.

Note: predictions for train/val plots are made by a model that was fit on
those labels, so they are not a held-out estimate of damage for those plots.
This leaderboard is a descriptive tool over the whole curated dataset, not a
model-evaluation artifact (see outputs/runs/*/tables/test_metrics.json for
held-out evaluation numbers).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.patch_dataset import CSFBCachedBagDataset, patch_collate_fn
from src.models.mil_model import DINOv3MILRegressor


def run_inference(model_path, cache_path, manifest_path, device, batch_size=32):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint lacks model_config metadata")

    model = DINOv3MILRegressor(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = CSFBCachedBagDataset(cache_path, split=None, manifest_path=manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=patch_collate_fn)

    filenames = dataset.df["filename"].tolist()
    splits = dataset.df["split"].tolist()

    preds = []
    trues = []
    plot_groups = []
    with torch.no_grad():
        for patch_tensors, area_tensors, targets, batch_plot_groups in loader:
            patch_tensors = [p.to(device) for p in patch_tensors]
            area_tensors = [a.to(device) for a in area_tensors] if area_tensors else None
            batch_preds = model(patch_tensors, area_tensors).cpu().squeeze(-1).numpy()
            if batch_preds.ndim == 0:
                batch_preds = np.array([batch_preds.item()])
            preds.extend(batch_preds.tolist())
            trues.extend(targets.numpy().tolist())
            plot_groups.extend(batch_plot_groups)

    return pd.DataFrame({
        "filename": filenames,
        "split": splits,
        "plot_group": plot_groups,
        "pred_score": preds,
        "true_score": trues,
    })


def build_plot_leaderboard(preds_df, manifest_df):
    plot_scores = preds_df.groupby("plot_group").agg(
        avg_pred_damage=("pred_score", "mean"),
        avg_true_damage=("true_score", "mean"),
        image_count=("pred_score", "count"),
    ).reset_index()

    plot_meta = manifest_df[["plot_group", "genotype", "location", "experiment", "split"]].drop_duplicates("plot_group")
    plot_scores = plot_scores.merge(plot_meta, on="plot_group", how="left")
    plot_scores = plot_scores.sort_values("avg_pred_damage", ascending=True).reset_index(drop=True)
    plot_scores.insert(0, "resistance_rank", range(1, len(plot_scores) + 1))
    return plot_scores


def build_genotype_subset(plot_leaderboard, min_plots=2):
    genotype_plot_counts = plot_leaderboard.groupby("genotype")["plot_group"].nunique()
    replicated_genotypes = genotype_plot_counts[genotype_plot_counts >= min_plots].index
    subset = plot_leaderboard[plot_leaderboard["genotype"].isin(replicated_genotypes)]

    genotype_table = subset.groupby("genotype").agg(
        avg_pred_damage=("avg_pred_damage", "mean"),
        avg_true_damage=("avg_true_damage", "mean"),
        plot_count=("plot_group", "nunique"),
        image_count=("image_count", "sum"),
    ).reset_index()
    genotype_table = genotype_table.sort_values("avg_pred_damage", ascending=True).reset_index(drop=True)
    genotype_table.insert(0, "resistance_rank", range(1, len(genotype_table) + 1))
    return genotype_table


def main():
    parser = argparse.ArgumentParser(description="Build the plot-level CSFB resistance leaderboard")
    parser.add_argument("--manifest", default="outputs/tables/baseline_manifest_split.csv")
    parser.add_argument("--model-path", default="outputs/runs/aggregation_weighted_seed42/checkpoints/best_model.pth")
    parser.add_argument("--cache-path", default="outputs/cache/dinov3_bags.pt")
    parser.add_argument("--out-plot-leaderboard", default="outputs/tables/plot_resistance_leaderboard.csv")
    parser.add_argument("--out-genotype-subset", default="outputs/tables/genotype_resistance_subset.csv")
    parser.add_argument("--min-genotype-plots", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on all frozen-manifest images using device: {device}")

    manifest_df = pd.read_csv(args.manifest)
    preds_df = run_inference(args.model_path, args.cache_path, args.manifest, device)

    plot_leaderboard = build_plot_leaderboard(preds_df, manifest_df)
    genotype_subset = build_genotype_subset(plot_leaderboard, min_plots=args.min_genotype_plots)

    r_pearson, _ = pearsonr(plot_leaderboard["avg_pred_damage"], plot_leaderboard["avg_true_damage"])
    r_spearman, _ = spearmanr(plot_leaderboard["avg_pred_damage"], plot_leaderboard["avg_true_damage"])
    print(f"Plot-level leaderboard: {len(plot_leaderboard)} plot groups, "
          f"{plot_leaderboard['image_count'].sum()} images")
    print(f"Plot-level pred-vs-true agreement: Pearson r={r_pearson:.4f}, Spearman rho={r_spearman:.4f}")
    print(f"Genotype subset (>= {args.min_genotype_plots} plot replicates): "
          f"{len(genotype_subset)} of {manifest_df['genotype'].nunique()} genotypes")

    Path(args.out_plot_leaderboard).parent.mkdir(parents=True, exist_ok=True)
    plot_leaderboard.to_csv(args.out_plot_leaderboard, index=False)
    genotype_subset.to_csv(args.out_genotype_subset, index=False)
    print(f"Saved plot-level leaderboard to {args.out_plot_leaderboard}")
    print(f"Saved caveated genotype subset to {args.out_genotype_subset}")


if __name__ == "__main__":
    main()
