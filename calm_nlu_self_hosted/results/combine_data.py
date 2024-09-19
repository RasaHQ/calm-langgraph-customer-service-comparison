import os
import json

# Define directories
base_dir = "."  # Current directory containing the combined distribution files
structured_dir = "./structured_results"  # Directory to store the unified structure

# Define datasets and metrics
datasets = ['cancellations', 'context_switch', 'corrections', 'happy_paths', 'multi_skill']
metrics = {
    'total_cost': 'costs',
    'completion_tokens': 'completion_tokens',
    'prompt_tokens': 'input_tokens',
    'latency': 'latencies'
}

# Create structured directories and split combined files
for dataset in datasets:
    dataset_dir = os.path.join(structured_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Path to the combined file
    combined_file = os.path.join(base_dir, f"{dataset}_distributions.json")
    print(f"Processing file: {combined_file}")

    if os.path.exists(combined_file):
        print(f"Found file: {combined_file}")
        with open(combined_file, 'r') as f:
            try:
                data = json.load(f)
                print(f"Data loaded from {combined_file}: {list(data.keys())}")
                for metric_key, metric_name in metrics.items():
                    if metric_key in data:
                        metric_data = data[metric_key]
                        metric_file = os.path.join(dataset_dir, f"{metric_name}.json")
                        with open(metric_file, 'w') as mf:
                            json.dump(metric_data, mf, indent=4)
                        print(f"Created {metric_file} with {metric_key} data.")
                    else:
                        print(f"Metric {metric_key} not found in {combined_file}.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {combined_file}: {e}")
        
        # Optionally, remove the combined file after splitting
        os.remove(combined_file)
        print(f"Removed {combined_file}")
    else:
        print(f"File not found: {combined_file}")

print("Splitting complete and files organized.")