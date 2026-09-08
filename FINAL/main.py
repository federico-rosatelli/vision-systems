import argparse
import os
import json
import sys
from src.data.make_baseline_manifest import build_baseline_manifest
from src.data.make_baseline_splits import create_baseline_splits
from src.data.cache_embeddings import cache_dinov3_patch_embeddings
from src.training.train import train_model
from src.training.train_patch import train_patch_model
from src.training.train_mil import train_mil_model
from src.evaluation.evaluate import evaluate_model
from src.evaluation.evaluate_mil import evaluate_mil_model
from src.evaluation.evaluate_ood import evaluate_ood_model
from src.evaluation.genotype_ranking import rank_genotypes
from src.inference.predict import predict_single_image
from src.visualization.plots import plot_training_history
from tests.test_pipeline import test_full_pipeline

def parse_args():
    # 1. Pre-parse to check if --config was passed on command line
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    known_args, _ = config_parser.parse_known_args()

    # 2. Determine configuration file
    config_file = known_args.config
    if config_file is None and os.path.exists("configs/config.json"):
        config_file = "configs/config.json"

    config_defaults = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config_defaults = json.load(f)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")
    elif config_file:
        print(f"Warning: Configuration file {config_file} not found.")

    parser = argparse.ArgumentParser(description="CSFB Damage Quantification Pipeline")
    
    # 3. Add arguments with defaults
    default_action = config_defaults.get("action", None)
    parser.add_argument("action", type=str, nargs="?", default=default_action, choices=[
        "prepare_data", "create_splits", "cache_embeddings", "train", "train_patch", 
        "train_mil", "evaluate", "evaluate_mil", "evaluate_ood", "rank", "predict", 
        "plot_logs", "test", "all"
    ], help="Pipeline action to perform")
    
    # Data preparation arguments
    parser.add_argument("--raw_dir", type=str, default="../dataset/Pictures_CFSB_leaf_damage", 
                        help="Path to the raw data directory")
    parser.add_argument("--manifest", type=str, default="outputs/tables/data_manifest.csv", 
                        help="Path to save/load the manifest in CSV format")
    parser.add_argument("--audit", type=str, default="outputs/tables/baseline_audit.json",
                        help="Path to save the baseline data audit")
    parser.add_argument("--split_audit", type=str, default="outputs/tables/baseline_split_audit.json")
    parser.add_argument("--split_groups", type=str, default="outputs/tables/baseline_split_groups.json")
    parser.add_argument("--out_manifest", type=str, default="outputs/tables/data_manifest_split.csv", 
                        help="Path to save/load the output manifest with split column")
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Path to pre-computed patch embedding cache file (.pt)")
    parser.add_argument("--disagreement_threshold", type=float, default=10.0, help="Max percentage diff between raters")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set size fraction")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set size fraction")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"], help="Loss function")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--out_dir", type=str, default="outputs/runs", help="Parent directory for experiment runs")
    parser.add_argument("--run_name", type=str, default="baseline_seed42")
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--weights_path", type=str, default="weights/dinov3-vits16-hf",
                        help="Authorized local DINOv3 checkpoint path")
    parser.add_argument("--head_width", type=int, default=256)
    parser.add_argument("--attn_L", type=int, default=128, help="Hidden dimension L for Attention MIL pool")
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--training_mode", type=str, default="regression", choices=["regression", "joint"], 
                        help="Mode of training: regression or joint (ranking)")
    parser.add_argument("--high_quality_only", type=lambda x: (str(x).lower() == 'true'), default=True, help="Filter for high quality images during training")
    parser.add_argument("--joint_margin", type=float, default=5.0, help="Margin for Joint Ranking loss")
    parser.add_argument("--aggregation", type=str, default="abmil", choices=["abmil", "gated_abmil", "weighted", "uniform"], 
                        help="Aggregation method for patch features (MIL only)")
    
    # Evaluation arguments
    parser.add_argument("--model_path", type=str, default="outputs/runs/baseline_regression_seed42/checkpoints/best_model.pth",
                        help="Path to the trained model checkpoint for evaluation")
    parser.add_argument("--preds_file", type=str, default="outputs/tables/test_predictions.csv", 
                        help="Path to the predictions CSV for ranking")
    parser.add_argument("--out_rank", type=str, default="outputs/tables/resistance_leaderboard.csv", 
                        help="Path to save the resistance leaderboard")
    
    # Inference arguments
    parser.add_argument("--image", type=str, help="Path to the single image to analyze (for predict action)")
    
    # Plotting arguments
    parser.add_argument("--log_file", type=str, default="outputs/runs/baseline_regression_seed42/logs/training_log.csv",
                        help="Path to the training log CSV for plotting")
                        
    # Configuration file
    parser.add_argument("--config", type=str, default=config_file, help="Path to a JSON configuration file to override arguments")

    # Override defaults with JSON config
    if config_defaults:
        parser.set_defaults(**config_defaults)
    
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.action:
        print("Error: No action specified on command line or in configuration file.")
        return
    
    if args.action in ["prepare_data", "all"]:
        print("=== Step 1: Data Preparation ===")
        print("Building Manifest...")
        build_baseline_manifest(args.raw_dir, args.manifest, args.audit)

    if args.action in ["create_splits", "all"]:
        print("=== Step 2: Grouped Data Split ===")
        create_baseline_splits(
            args.manifest,
            args.out_manifest,
            args.split_audit,
            args.split_groups,
            seed=args.seed,
            train_ratio=1.0 - args.test_size - args.val_size,
            val_ratio=args.val_size,
            test_ratio=args.test_size,
        )

    if args.action == "cache_embeddings":
        print("=== Step 2b: Cache DINOv3 Patch Embeddings ===")
        cache_out = args.cache_path or "outputs/cache/dinov3_bags.pt"
        cache_dinov3_patch_embeddings(
            manifest_path=args.out_manifest,
            output_cache_path=cache_out,
            weights_path=args.weights_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
        
    if args.action in ["train"]:
        print("=== Step 2: Training (Whole-Image Baseline) ===")
        train_model(
            manifest=args.out_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss=args.loss,
            patience=args.patience,
            out_dir=args.out_dir,
            run_name=args.run_name,
            seed=args.seed,
            num_workers=args.num_workers,
            image_size=args.image_size,
            training_mode=args.training_mode,
            high_quality_only=args.high_quality_only,
            joint_margin=args.joint_margin,
            model_name=args.model_name,
            head_width=args.head_width,
            dropout_p=args.dropout_p,
            weights_path=args.weights_path,
        )
        
    if args.action in ["train_patch"]:
        print("=== Step 2: Training (Patch Baseline) ===")
        train_patch_model(
            manifest=args.out_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss=args.loss,
            patience=args.patience,
            out_dir=args.out_dir,
            run_name=args.run_name,
            seed=args.seed,
            num_workers=args.num_workers,
            image_size=args.image_size,
            high_quality_only=args.high_quality_only,
            model_name=args.model_name,
            head_width=args.head_width,
            dropout_p=args.dropout_p,
            weights_path=args.weights_path,
            aggregation=args.aggregation,
            training_mode=args.training_mode,
            joint_margin=args.joint_margin,
        )

    if args.action in ["train_mil", "all"]:
        print("=== Step 2: Training (Multiple Instance Learning / MIL) ===")
        train_mil_model(
            manifest=args.out_manifest,
            cache_path=args.cache_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss=args.loss,
            patience=args.patience,
            out_dir=args.out_dir,
            run_name=args.run_name,
            seed=args.seed,
            num_workers=args.num_workers,
            image_size=args.image_size,
            high_quality_only=args.high_quality_only,
            model_name=args.model_name,
            head_width=args.head_width,
            attn_L=args.attn_L,
            dropout_p=args.dropout_p,
            weights_path=args.weights_path,
            aggregation=args.aggregation,
            training_mode=args.training_mode,
            joint_margin=args.joint_margin,
        )
        
    if args.action in ["evaluate"]:
        print("=== Step 3: Evaluation (Whole-Image) ===")
        evaluate_model(
            manifest=args.out_manifest,
            model_path=args.model_path,
            batch_size=args.batch_size,
            out_dir=None,
            num_workers=args.num_workers,
            image_size=args.image_size
        )

    if args.action in ["evaluate_mil", "all"]:
        print("=== Step 3: Evaluation (MIL Model) ===")
        evaluate_mil_model(
            manifest=args.out_manifest,
            model_path=args.model_path,
            batch_size=args.batch_size,
            out_dir=None,
            num_workers=args.num_workers,
            cache_path=args.cache_path
        )

    if args.action in ["evaluate_ood"]:
        print("=== Step 3b: Out-of-Distribution (OOD) Evaluation ===")
        evaluate_ood_model(
            model_path=args.model_path,
            raw_dir=args.raw_dir,
            output_dir="outputs/tables"
        )
        
    if args.action in ["rank"]:
        print("=== Step 4: Genotype Resistance Ranking ===")
        rank_genotypes(args.preds_file, args.out_manifest, args.out_rank)
        
    if args.action == "predict":
        print("=== Single Image Inference ===")
        if not args.image:
            print("Error: --image is required for the predict action.")
        else:
            predict_single_image(args.image, args.model_path, args.image_size)
            
    if args.action == "plot_logs":
        print("=== Plotting Training Logs ===")
        plots_dir = os.path.join(args.out_dir, "plots")
        plot_training_history(args.log_file, plots_dir)
        
    if args.action in ["test"]:
        print("=== Step 5: Testing ===")
        test_full_pipeline()

if __name__ == "__main__":
    main()
