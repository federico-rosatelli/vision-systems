import argparse
import sys
from src.data.make_manifest import build_manifest
from src.data.make_splits import create_splits
from src.training.train import train_model
from tests.test_pipeline import test_full_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="CSFB Damage Quantification Pipeline")
    parser.add_argument("action", type=str, choices=["prepare_data", "train", "test", "all"], 
                        help="Action to perform: prepare_data, train, test, or all")
    
    # Data preparation arguments
    parser.add_argument("--raw_dir", type=str, default="/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage", 
                        help="Absolute path to the raw data directory")
    parser.add_argument("--manifest", type=str, default="outputs/tables/data_manifest.csv", 
                        help="Path to save/load the manifest in CSV format")
    parser.add_argument("--out_manifest", type=str, default="outputs/tables/data_manifest_split.csv", 
                        help="Path to save/load the output manifest with split column")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"], help="Loss function")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--image_size", type=int, default=224, help="Input image resolution")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.action in ["prepare_data", "all"]:
        print("=== Step 1: Data Preparation ===")
        print("Building Manifest...")
        build_manifest(args.raw_dir, args.manifest)
        print("Creating Splits...")
        create_splits(args.manifest, args.out_manifest, args.seed)
        
    if args.action in ["train", "all"]:
        print("=== Step 2: Training ===")
        train_model(
            manifest=args.out_manifest,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            loss=args.loss,
            patience=args.patience,
            out_dir=args.out_dir,
            seed=args.seed,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
        
    if args.action in ["test", "all"]:
        print("=== Step 3: Testing ===")
        test_full_pipeline()

if __name__ == "__main__":
    main()
