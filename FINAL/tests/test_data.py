import unittest
import os
import tempfile
import pandas as pd
from src.data.make_manifest import parse_score_csv
from src.data.make_splits import create_splits
from src.data.dataset import CSFBDataset
from PIL import Image

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_parse_score_csv_robustness(self):
        """Test parsing of CSVs with different delimiters and column names."""
        # 1. Comma separated
        csv1 = os.path.join(self.base_dir, "scores1.csv")
        with open(csv1, "w") as f:
            f.write("Filename,Score_JLU,Score_GAU\n")
            f.write("img1.jpg,10,12\n")
            
        df1 = parse_score_csv(csv1)
        self.assertEqual(df1['filename'].iloc[0], "img1.jpg")
        
        # 2. Semicolon separated, different column name for filename
        csv2 = os.path.join(self.base_dir, "scores2.csv")
        with open(csv2, "w") as f:
            f.write("Picture;Score_JLU;Score_GAU\n")
            f.write("img2;15;15\n") # Missing .jpg
            
        df2 = parse_score_csv(csv2)
        self.assertEqual(df2['filename'].iloc[0], "img2.jpg") # Should auto-append .jpg
        
    def test_create_splits_leakage(self):
        """Ensure no plot_group leaks across train/val/test splits."""
        # Create a mock manifest
        manifest_path = os.path.join(self.base_dir, "mock_manifest.csv")
        data = []
        for i in range(50): # 50 plots
            for j in range(3): # 3 images per plot
                data.append({
                    'filename': f'plot{i}_img{j}.jpg',
                    'plot_group': f'plot_{i}',
                    'mean_score': 10 + (i % 5) # Distribute scores
                })
        df = pd.DataFrame(data)
        df.to_csv(manifest_path, index=False)
        
        out_path = os.path.join(self.base_dir, "split_manifest.csv")
        create_splits(manifest_path, out_path, random_state=42)
        
        split_df = pd.read_csv(out_path)
        
        train_plots = set(split_df[split_df['split'] == 'train']['plot_group'])
        val_plots = set(split_df[split_df['split'] == 'val']['plot_group'])
        test_plots = set(split_df[split_df['split'] == 'test']['plot_group'])
        
        self.assertTrue(train_plots.isdisjoint(val_plots), "Leakage between train and val")
        self.assertTrue(train_plots.isdisjoint(test_plots), "Leakage between train and test")
        self.assertTrue(val_plots.isdisjoint(test_plots), "Leakage between val and test")
        
    def test_dataset_item(self):
        """Ensure CSFBDataset yields correctly formatted outputs."""
        # Create a mock manifest and image
        img_path = os.path.join(self.base_dir, "test_img.jpg")
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        manifest_path = os.path.join(self.base_dir, "mock_manifest_ds.csv")
        df = pd.DataFrame([{
            'filename': 'test_img.jpg',
            'absolute_path': img_path,
            'mean_score': 45.5,
            'plot_group': 'plot_test',
            'is_high_quality': True
        }])
        df.to_csv(manifest_path, index=False)
        
        dataset = CSFBDataset(manifest_path, high_quality_only=True)
        
        image, target, plot_group = dataset[0]
        
        self.assertEqual(target.item(), 45.5)
        self.assertEqual(plot_group, 'plot_test')
        self.assertIsInstance(image, Image.Image)

if __name__ == '__main__':
    unittest.main()
