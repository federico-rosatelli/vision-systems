"""Fine-tune RF-DETR on the manually annotated hole/pitting COCO dataset.

This is exploratory side work (see `analyses/HOLE_PITTING_ANNOTATION_PLAN.md`) and is
independent of the frozen area-weighted MIL model. `dataset_dir` must point at a COCO
export (train/valid/test subfolders with `_annotations.coco.json`) produced by manually
annotating `outputs/hole_pitting_annotations/images/` in x-anylabeling — that annotation
step cannot be automated.
"""
import argparse
import json
from pathlib import Path

from rfdetr import RFDETRNano


def load_config():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR for hole/pitting detection")
    parser.add_argument("--config", type=str, default="configs/rfdetr_hole_pitting_train.json")
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        return json.load(f)


def main():
    config = load_config()
    dataset_dir = Path(config["dataset_dir"])
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(
            f"No COCO annotations found under {dataset_dir}. Annotate "
            "outputs/hole_pitting_annotations/images/ in x-anylabeling and export as "
            "COCO to this path before training."
        )

    model = RFDETRNano()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=config["output_dir"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    main()
