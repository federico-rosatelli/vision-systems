"""Train the area-weighted MIL model with the joint Huber + pairwise ranking loss (3 seeds).

Uses exactly the hyperparameters of configs/aggregation_experiments.json, so the runs
`outputs/runs/joint_weighted_seed{42,43,44}` are directly comparable with
`aggregation_weighted_seed*`. Needs the cached bags (outputs/cache/dinov3_bags.pt).

Usage:
    python scripts/train_joint_ranking.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train_mil import train_mil_model

config = json.load(open("configs/aggregation_experiments.json"))
for seed in (42, 43, 44):
    train_mil_model(
        manifest=config["manifest"], cache_path=config["cache_path"], epochs=config["epochs"],
        batch_size=config["batch_size"], lr=config["learning_rate"], loss=config["loss"],
        patience=config["patience"], out_dir=config["output_dir"],
        run_name=f"joint_weighted_seed{seed}", seed=seed, num_workers=0,
        high_quality_only=False, model_name=config["model_name"],
        head_width=config["head_width"], dropout_p=config["dropout"],
        attn_L=config["attention_width"], weights_path=config["weights_path"],
        aggregation="weighted", training_mode="joint", joint_margin=5.0,
    )
