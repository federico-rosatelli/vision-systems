"""Collect the raw images used for the hole/pitting manual-annotation side project.

Gathers the union of filenames already reviewed in the direct-damage audit and the
hole-detection comparison (both rejected as automated detectors, per
`analyses/CONTINUATION_PLAN.md`), resolves their raw paths from the frozen baseline
manifest, copies them into a flat folder ready for annotation in x-anylabeling, and
writes a manifest that reserves a held-out eval slice. No test-split images are
included, matching the frozen-test-split constraint used throughout the project.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def load_config():
    parser = argparse.ArgumentParser(description="Collect candidate images for hole/pitting annotation")
    parser.add_argument("--config", type=str, default="configs/hole_pitting_candidates.json")
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as f:
        return json.load(f)


def collect_candidate_filenames(config):
    sources = [
        pd.read_csv(config["direct_damage_audit_csv"])[["filename", "split"]],
        pd.read_csv(config["hole_detection_comparison_csv"])[["filename", "split"]],
    ]
    combined = pd.concat(sources, ignore_index=True).drop_duplicates(subset="filename")
    if (combined["split"] == "test").any():
        raise ValueError("Candidate set must not include test-split images")
    return combined.reset_index(drop=True)


def resolve_raw_paths(candidates, manifest_path):
    manifest = pd.read_csv(manifest_path)[["filename", "image_path"]]
    merged = candidates.merge(manifest, on="filename", how="left")
    missing = merged[merged["image_path"].isna()]
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} candidate filenames not found in baseline manifest: "
                          f"{missing['filename'].tolist()}")
    return merged


def assign_eval_holdout(df, eval_fraction, seed):
    df = df.copy()
    df["annotation_subset"] = "train"
    holdout = df.groupby("split", group_keys=False).apply(
        lambda g: g.sample(frac=eval_fraction, random_state=seed)
    )
    df.loc[holdout.index, "annotation_subset"] = "eval"
    return df


def copy_images(df, images_dir):
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        dest = images_dir / row.filename
        if not dest.exists():
            shutil.copy2(row.image_path, dest)
    return images_dir


def main():
    config = load_config()
    candidates = collect_candidate_filenames(config)
    resolved = resolve_raw_paths(candidates, config["manifest"])
    resolved = assign_eval_holdout(resolved, config["eval_fraction"], config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = copy_images(resolved, output_dir / "images")

    manifest_out = output_dir / "annotation_candidates.csv"
    resolved.to_csv(manifest_out, index=False)

    print(f"Collected {len(resolved)} candidate images -> {images_dir}")
    print(f"  train subset: {(resolved['annotation_subset'] == 'train').sum()}")
    print(f"  eval subset:  {(resolved['annotation_subset'] == 'eval').sum()}")
    print(f"Manifest written to {manifest_out}")


if __name__ == "__main__":
    main()
