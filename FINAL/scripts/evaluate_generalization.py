"""Extended zero-shot generalization study for the report.

Extracts plant-bag DINOv3 features once for
  * the two OOD field trials (Weilburger Grenze WG1 and DSV Trial 1), and
  * the Gross-Gerau GG1 images that were *not* in the 470-image calibration set because
    JLU and GAU disagreed, restricted to plot groups outside the train/val splits,
then evaluates every MIL checkpoint on them, next to constant predictors and the
inter-rater agreement. Preprocessing mirrors CSFBPlantPatchDataset exactly and also
records per-image diagnostics (frame detected, plant count, brightness) for stratification.

Usage:
    python scripts/evaluate_generalization.py            # extract (if needed) + evaluate
    python scripts/evaluate_generalization.py --skip-extract
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate_mil import pairwise_ranking_accuracy
from src.evaluation.evaluate_ood import build_ood_manifest
from src.models.mil_model import DINOv3MILRegressor
from src.preprocessing.frame_crop import crop_frame_interior, detect_frame, read_image_oriented
from src.preprocessing.plant_regions import extract_plant_regions

RAW_DIR = "/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage"
MANIFEST = "outputs/tables/baseline_manifest_split.csv"
CACHE = "outputs/cache/generalization_bags.pt"
OUT_DIR = Path("outputs/tables/generalization")
WEIGHTS = "weights/dinov3-vits16-hf"
HSV = {"lower": [35, 40, 40], "upper": [85, 255, 255]}  # CSFBPlantPatchDataset defaults
MIN_PLANT_AREA = 150
CHECKPOINTS = (
    [f"aggregation_{a}_seed{s}" for a in ("weighted", "uniform", "abmil", "gated_abmil") for s in (42, 43, 44)]
    + [f"joint_weighted_seed{s}" for s in (42, 43, 44)]
    + ["mil_weighted_seed42"]
)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_gg1_heldout(manifest):
    scores = pd.read_csv(Path(RAW_DIR) / "2025_10_21_RSFB-Phenotyping_GG1_scores.csv",
                         sep=None, engine="python", encoding="utf-8-sig")
    scores["filename"] = scores["Filename"].astype(str).str.replace(r"\.jpg$", "", regex=True) + ".jpg"
    scores["score_jlu"] = pd.to_numeric(scores["Score_JLU"], errors="coerce")
    scores["score_gau"] = pd.to_numeric(scores["Score_GAU"], errors="coerce")
    scores = scores.dropna(subset=["score_jlu", "score_gau"])
    seen_groups = set(manifest.loc[manifest.split.isin(["train", "val"]), "plot_group"])
    keep = scores[~scores.filename.isin(manifest.filename) & ~scores["QR-Code"].isin(seen_groups)].copy()
    folder = Path(RAW_DIR) / "2025_10_21_RSFB-Phenotyping_GG1_JLU"
    paths = {p.name: str(p) for p in folder.rglob("*.jpg")}
    keep["image_path"] = keep.filename.map(paths)
    keep = keep.dropna(subset=["image_path"])
    return pd.DataFrame({
        "filename": keep.filename, "image_path": keep.image_path,
        "mean_score": (keep.score_jlu + keep.score_gau) / 2, "plot_group": keep["QR-Code"],
        "score_jlu": keep.score_jlu, "score_gau": keep.score_gau,
    })


class BagDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = read_image_oriented(row.image_path)
        detection = detect_frame(image)
        frame_ok = detection.status == "detected"
        crop = crop_frame_interior(image, detection.corners)[0] if frame_ok else image
        mask, regions = extract_plant_regions(
            crop, hue_min=HSV["lower"][0], hue_max=HSV["upper"][0],
            saturation_min=HSV["lower"][1], value_min=HSV["lower"][2],
            minimum_region_green_area=MIN_PLANT_AREA,
        )
        tensors, areas = [], []
        for region in regions:
            x1, y1, x2, y2 = region.patch_box
            rgb = cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensors.append(TRANSFORM(Image.fromarray(rgb)))
            areas.append(float(region.green_area))
        if not tensors:
            tensors, areas = [TRANSFORM(Image.new("RGB", (224, 224)))], [0.0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        diag = {
            "frame_detected": frame_ok, "n_plants": len(regions),
            "green_fraction": float(np.count_nonzero(mask)) / mask.size,
            "brightness": float(hsv[:, :, 2].mean()),
        }
        return torch.stack(tensors), torch.tensor(areas), idx, diag


def extract(datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3MILRegressor(model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                               local_weights_path=WEIGHTS).to(device).eval()
    bags = []
    for name, df in datasets.items():
        loader = DataLoader(BagDataset(df), batch_size=1, num_workers=8, collate_fn=lambda b: b[0])
        with torch.no_grad():
            for patches, areas, idx, diag in tqdm(loader, desc=f"extract {name}"):
                row = df.iloc[idx]
                bags.append({
                    "dataset": name, "filename": row.filename, "plot_group": row.plot_group,
                    "target": float(row.mean_score),
                    "score_jlu": float(row.get("score_jlu", np.nan)),
                    "score_gau": float(row.get("score_gau", np.nan)),
                    "features": model.extract_patch_features(patches.to(device)).cpu(),
                    "areas": areas, **diag,
                })
    torch.save({"weights": WEIGHTS, "hsv": HSV, "min_plant_area": MIN_PLANT_AREA, "bags": bags}, CACHE)
    return bags


def metrics(y, p, rng=None):
    out = {
        "n": len(y), "mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "pearson": float(pearsonr(y, p)[0]) if np.std(p) > 0 else np.nan,
        "spearman": float(spearmanr(y, p)[0]) if np.std(p) > 0 else np.nan,
    }
    if np.std(p) > 0:
        out["pairwise_gap5"] = pairwise_ranking_accuracy(y, p, 5.0)["accuracy"]
    if rng is not None and np.std(p) > 0:
        boot = [spearmanr(y[i], p[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
        out["spearman_ci95"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    return out


def predict(run, bags, device):
    ckpt = torch.load(f"outputs/runs/{run}/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model = DINOv3MILRegressor(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        return np.array([
            model([b["features"].to(device)], [b["areas"].to(device)]).item() for b in bags
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)

    if args.skip_extract and os.path.exists(CACHE):
        bags = torch.load(CACHE, weights_only=False)["bags"]
    else:
        datasets = {
            "WG1_Weilburger_Grenze": build_ood_manifest(RAW_DIR, "2025_10_07_RSFB-Phenotyping_WG1_JLU",
                                                        "2025_10_07_RSFB-Phenotyping_WG1_JLU_scores.csv"),
            "DSV_Trial_1": build_ood_manifest(RAW_DIR, "2025_09_15_Res4StRes_T1_DSV",
                                              "2025_09_15_Res4StRes_T1_DSV_scores.csv"),
            "GG1_heldout_disagreement": build_gg1_heldout(manifest),
            # Sanity check: must reproduce the cached-feature test metrics.
            "GG1_calibration_test": manifest[manifest.split == "test"][
                ["filename", "image_path", "mean_score", "plot_group", "score_jlu", "score_gau"]],
        }
        bags = extract(datasets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    train = manifest[manifest.split == "train"].mean_score
    constants = {"constant_train_mean": float(train.mean()), "constant_train_median": float(train.median())}

    diag = pd.DataFrame([{k: v for k, v in b.items() if k not in ("features", "areas")} for b in bags])
    preds = {run: predict(run, bags, device) for run in CHECKPOINTS}
    for run, p in preds.items():
        diag[f"pred_{run}"] = p
    diag.to_csv(OUT_DIR / "per_image_predictions.csv", index=False)

    rows, strata, rater = [], [], []
    for name, d in diag.groupby("dataset"):
        y = d.target.to_numpy()
        for c, v in constants.items():
            rows.append({"dataset": name, "model": c, **metrics(y, np.full_like(y, v))})
        for run in CHECKPOINTS:
            p = d[f"pred_{run}"].to_numpy()
            rows.append({"dataset": name, "model": run, **metrics(y, p, rng)})
            plot = d.assign(p=p).groupby("plot_group")[["target", "p"]].mean()
            rows[-1]["plot_level_spearman"] = float(spearmanr(plot.target, plot.p)[0])
            rows[-1]["n_plots"] = len(plot)
        main_pred = d["pred_aggregation_weighted_seed42"].to_numpy()
        for col, bins, labels in [
            ("target", [-1, 5, 10, 20, 101], ["0-5", "5-10", "10-20", ">20"]),
            ("brightness", np.nanpercentile(d.brightness, [0, 33.3, 66.7, 100]), ["dark", "mid", "bright"]),
            ("frame_detected", None, None),
        ]:
            groups = d[col] if bins is None else pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
            for g, sub in d.groupby(groups, observed=True):
                ys, ps = sub.target.to_numpy(), sub["pred_aggregation_weighted_seed42"].to_numpy()
                strata.append({"dataset": name, "stratum": col, "value": str(g), "n": len(sub),
                               "mae": float(np.abs(ys - ps).mean()), "mean_true": float(ys.mean()),
                               "mean_pred": float(ps.mean()),
                               "spearman": float(spearmanr(ys, ps)[0]) if len(sub) > 2 else np.nan})
        if d.score_jlu.notna().all():
            jlu, gau = d.score_jlu.to_numpy(), d.score_gau.to_numpy()
            rater.append({"dataset": name, "pair": "JLU vs GAU", **metrics(jlu, gau)})
            for run in ("aggregation_weighted_seed42", "joint_weighted_seed42"):
                p = d[f"pred_{run}"].to_numpy()
                rater.append({"dataset": name, "pair": f"{run} vs JLU", **metrics(jlu, p)})
                rater.append({"dataset": name, "pair": f"{run} vs GAU", **metrics(gau, p)})
                rater.append({"dataset": name, "pair": f"{run} vs mean(JLU,GAU)", **metrics(d.target.to_numpy(), p)})

    pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_by_model.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT_DIR / "strata_aggregation_weighted_seed42.csv", index=False)
    pd.DataFrame(rater).to_csv(OUT_DIR / "rater_agreement.csv", index=False)
    diag_summary = diag.groupby("dataset")[["frame_detected", "n_plants", "green_fraction", "brightness", "target"]].agg(["mean", "median"])
    diag_summary.to_csv(OUT_DIR / "preprocessing_diagnostics.csv")
    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(rater).round(3).to_string())
    print(diag_summary.round(3).to_string())


if __name__ == "__main__":
    main()
