import os
import shutil
import json

# Define directories
base_dir = "."  # Base directory containing the original result files
structured_dir = "./structured_results"  # Directory to store the unified structure

# Define datasets and metrics
datasets = ['cancellations', 'context_switch', 'corrections', 'happy_paths', 'multi_skill']
metrics = ['costs', 'input_tokens', 'latencies', 'output_tokens']

# Create structured directories and move files
for dataset in datasets:
    dataset_dir = os.path.join(structured_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if combined metric files exist and split them
    combined_file = os.path.join(base_dir, f"{dataset}_distributions.json")
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            data = json.load(f)
            for metric in metrics:
                if metric in data:
                    metric_data = data[metric]
                    metric_file = os.path.join(dataset_dir, f"{metric}.json")
                    with open(metric_file, 'w') as mf:
                        json.dump(metric_data, mf, indent=4)
                    print(f"Created {metric_file} from {combined_file}")
        # Optionally, remove the combined file after splitting
        os.remove(combined_file)
    
    # Move existing separated metric files into the dataset directory
    for metric in metrics:
        separate_file = os.path.join(base_dir, f"{dataset}_{metric}.json")
        if os.path.exists(separate_file):
            new_file = os.path.join(dataset_dir, f"{metric}.json")
            shutil.move(separate_file, new_file)
            print(f"Moved {separate_file} to {new_file}")

# Optional: move the predictions file if needed
predictions_path = os.path.join(base_dir, "predictions.yaml")
new_predictions_path = os.path.join(structured_dir, "predictions.yaml")
if os.path.exists(predictions_path):
    shutil.move(predictions_path, new_predictions_path)
    print(f"Moved {predictions_path} to {new_predictions_path}")