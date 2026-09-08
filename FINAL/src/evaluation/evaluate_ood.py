import os
import csv
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.mil_model import DINOv3MILRegressor
from src.data.patch_dataset import CSFBPlantPatchDataset, patch_collate_fn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def build_ood_manifest(raw_dir, folder_name, score_csv_name):
    """
    Builds a temporary manifest DataFrame for an OOD folder.
    """
    raw_dir = Path(raw_dir)
    folder_path = raw_dir / folder_name
    score_csv = raw_dir / score_csv_name
    
    if not score_csv.is_file():
        # Check inside folder_path
        score_csv = folder_path / score_csv_name
        if not score_csv.is_file():
            return None

    with open(score_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    manifest_rows = []
    for r in rows:
        fn = r.get("Filename") or r.get("filename")
        if not fn:
            continue
        if not fn.endswith(".jpg"):
            fn += ".jpg"
        
        # Look for physical image file
        img_p = folder_path / fn
        if not img_p.is_file():
            img_p = folder_path / folder_name / fn
        if not img_p.is_file():
            continue

        score_jlu = float(r.get("Score_JLU", 0.0) or 0.0)
        score_gau = float(r.get("Score_GAU", 0.0) or 0.0)
        supplied_mean = float(r.get("mean_score", r.get("Score", 0.0)) or 0.0)
        mean_score = (score_jlu + score_gau) / 2.0 if (score_jlu and score_gau) else supplied_mean

        plot_group = r.get("QR-Code") or r.get("Plotnr") or "unknown"
        manifest_rows.append({
            "filename": fn,
            "image_path": str(img_p),
            "file_exists": True,
            "mean_score": mean_score,
            "plot_group": plot_group,
            "folder": folder_name
        })

    if not manifest_rows:
        return None

    return pd.DataFrame(manifest_rows)

def evaluate_ood_model(
    model_path="outputs/runs/mil_abmil_seed42/checkpoints/best_model.pth",
    raw_dir="../dataset/Pictures_CFSB_leaf_damage",
    output_dir="outputs/tables"
):
    """
    Evaluates a trained MIL checkpoint zero-shot on Out-of-Distribution (OOD) field trial datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Zero-Shot OOD Evaluation using model: {model_path} on device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config')
    if not model_config:
        raise ValueError("Checkpoint lacks model_config metadata")

    image_size = model_config.get("image_size", 224)
    model = DINOv3MILRegressor(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # OOD Folders to benchmark
    ood_benchmarks = [
        {"folder": "2025_10_07_RSFB-Phenotyping_WG1_JLU", "csv": "2025_10_07_RSFB-Phenotyping_WG1_JLU_scores.csv", "name": "Rauischholzhausen_WG1"},
        {"folder": "2025_09_15_Res4StRes_T1_DSV", "csv": "2025_09_15_Res4StRes_T1_DSV_scores.csv", "name": "DSV_Trial_1"},
    ]

    results = []

    for bench in ood_benchmarks:
        print(f"\n--- Benchmarking OOD Dataset: {bench['name']} ({bench['folder']}) ---")
        df_ood = build_ood_manifest(raw_dir, bench['folder'], bench['csv'])
        if df_ood is None or len(df_ood) == 0:
            print(f"Warning: Could not load valid manifest for {bench['name']}. Skipping.")
            continue

        temp_manifest_path = f"outputs/tables/temp_ood_{bench['name']}.csv"
        df_ood.to_csv(temp_manifest_path, index=False)

        dataset = CSFBPlantPatchDataset(manifest_path=temp_manifest_path, transform=eval_transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=patch_collate_fn)

        y_true, y_pred = [], []
        with torch.no_grad():
            for patch_tensors, area_tensors, targets, _ in tqdm(loader, desc=f"OOD {bench['name']}"):
                patch_tensors = [p.to(device) for p in patch_tensors]
                area_tensors = [a.to(device) for a in area_tensors] if area_tensors else None

                preds = model(patch_tensors, area_tensors).cpu().squeeze(-1).numpy()
                if preds.ndim == 0:
                    preds = np.array([preds.item()])

                y_pred.extend(preds)
                y_true.extend(targets.numpy())

        if os.path.exists(temp_manifest_path):
            os.remove(temp_manifest_path)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        p_r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)
        s_rho, _ = spearmanr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)

        print(f"Result [{bench['name']}] (N={len(y_true)}) -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, Pearson r: {p_r:.4f}, Spearman rho: {s_rho:.4f}")

        results.append({
            "ood_dataset": bench['name'],
            "folder": bench['folder'],
            "sample_count": len(y_true),
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_r": float(p_r),
            "spearman_rho": float(s_rho)
        })

    os.makedirs(output_dir, exist_ok=True)
    df_res = pd.DataFrame(results)
    csv_out = os.path.join(output_dir, "ood_evaluation_results.csv")
    df_res.to_csv(csv_out, index=False)

    json_out = os.path.join(output_dir, "ood_evaluation_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nOOD Evaluation Complete. Results saved to {csv_out}")
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Out-of-Distribution Evaluation")
    parser.add_argument("--model_path", default="outputs/runs/mil_abmil_seed42/checkpoints/best_model.pth")
    parser.add_argument("--raw_dir", default="../dataset/Pictures_CFSB_leaf_damage")
    args = parser.parse_args()
    evaluate_ood_model(args.model_path, args.raw_dir)
