import os
import argparse
import pandas as pd

def rank_genotypes(predictions_csv, manifest_csv, out_csv):
    """
    Reads the image-level predictions, maps them back to the manifest 
    to extract genotype (or uses plot_group as a proxy), and ranks them.
    """
    if not os.path.exists(predictions_csv):
        print(f"Predictions file {predictions_csv} not found.")
        return
        
    preds_df = pd.read_csv(predictions_csv)
    manifest_df = pd.read_csv(manifest_csv)
    
    # 1. Aggregate at Plot Level
    # We take the mean prediction for the 3 images of a plot
    plot_scores = preds_df.groupby('plot_group').agg(
        avg_pred_damage=('pred_score', 'mean'),
        avg_true_damage=('true_score', 'mean'),
        image_count=('pred_score', 'count')
    ).reset_index()
    
    # 2. Join with Genotype information if available in manifest
    # If the manifest doesn't have a 'genotype' column, we will attempt to parse it 
    # from plot_group, or just output the plot leaderboard.
    if 'genotype' in manifest_df.columns:
        # Keep only unique plot_group to genotype mappings
        plot_to_geno = manifest_df[['plot_group', 'genotype']].drop_duplicates()
        plot_scores = plot_scores.merge(plot_to_geno, on='plot_group', how='left')
        
        # Aggregate at Genotype Level
        leaderboard = plot_scores.groupby('genotype').agg(
            genotype_damage=('avg_pred_damage', 'mean'),
            plot_count=('plot_group', 'count')
        ).reset_index()
        
        leaderboard = leaderboard.sort_values(by='genotype_damage', ascending=True)
        print("--- Genotype Resistance Leaderboard ---")
        
    else:
        # Fallback: Just rank plots
        print("No 'genotype' column found in manifest. Ranking plots directly.")
        leaderboard = plot_scores.sort_values(by='avg_pred_damage', ascending=True)
        print("--- Plot Resistance Leaderboard ---")
        
    # Print Top 10 Most Resistant
    print(leaderboard.head(10).to_string(index=False))
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    leaderboard.to_csv(out_csv, index=False)
    print(f"\nLeaderboard saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, default="outputs/tables/test_predictions.csv")
    parser.add_argument("--manifest", type=str, default="outputs/tables/data_manifest_split.csv")
    parser.add_argument("--out", type=str, default="outputs/tables/resistance_leaderboard.csv")
    args = parser.parse_args()
    
    rank_genotypes(args.preds, args.manifest, args.out)
