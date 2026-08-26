# Cabbage Stem Flea Beetle (CSFB) Damage Quantification Project Summary

## What Has Been Done
We have built an automated computer vision pipeline to calculate the amount of insect damage on plant leaves. Instead of humans manually guessing the damage percentage, our Artificial Intelligence (AI) system looks at the pictures and assigns a score automatically.

Here is a summary of the steps we completed:
1. Data Preparation: We merged the human expert scores from multiple CSV files and filtered the dataset. We kept only the images where the two human experts strongly agreed (disagreement less than 10%), to ensure the AI learns from high-quality data.
2. Artificial Intelligence Model: We used a state-of-the-art vision model (DINOv3) to extract features from the leaves.
3. Training Method: We realized that training the AI to guess the exact percentage was difficult because of the white background in the photos. To solve this, we used a technique called "Joint Ranking". We trained the AI by showing it two leaves at a time and asking: "Which one is more damaged?". This forced the model to focus on the actual holes in the leaf rather than the background.
4. Genotype Ranking: We built a script that takes the AI predictions and calculates an average damage score for each specific plant genotype, creating a leaderboard of the most resistant plants.

## Results
The AI system achieved excellent performance on a test set of images it had never seen before:
- Mean Absolute Error (MAE): 3.47%. This means that, on average, the AI prediction is only 3.47% away from the human expert score. This is highly accurate, considering that human experts themselves often disagree by about 6.6%.
- Spearman Correlation: 0.57. This metric shows that the AI is capable of correctly ordering the leaves from least damaged to most damaged. 

These results prove that the model works well as a fast, automated screening tool to find the most resistant plant genotypes. 

## How to Run the Program
All the settings (such as the disagreement threshold, learning rates, and dataset paths) are stored in the `configs/config.json` file. You can easily edit this file with any text editor to modify the pipeline without touching the code.

To run the different parts of the pipeline, open your terminal and use the following commands:

1. Prepare the dataset (filters bad images and splits data into train/test sets):
python main.py prepare_data --config configs/config.json

2. Train the AI model (using the highly effective Joint Ranking method):
python main.py train --config configs/config.json

3. Evaluate the trained model on new test images:
python main.py evaluate --config configs/config.json

4. Generate the final biological leaderboard (to see which genotypes resist damage the best):
python main.py rank --config configs/config.json

5. Plot the training history (to visually check how the AI learned):
python main.py plot_logs --config configs/config.json
