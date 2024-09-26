import yaml
import os
import numpy as np
import json
from app import run_conversation, initialize_graph, reset_db

def get_test_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def load_test(file_path):
    loaded_test = None
    with open(file_path) as stream:
        try:
            loaded_test = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded_test

def run_test(test_case):
    name = test_case.get("test_case")
    steps = test_case.get("steps")
    user_messages = [step.get("user") for step in steps if not step.get("bot")]
    runtime_stats = run_conversation(user_messages)
    return name, runtime_stats

def dump_as_convo(name, runtime_info):
    steps = []
    for turn in runtime_info:
        if turn.get("user"):
            steps.append({"user": turn.get("user")})
        if turn.get("bot_response"):
            steps.append({"bot": turn.get("bot_response")})
    return {"test_case": name, "steps": steps}

def print_metric_stats(data):
    print("Min:", np.min(data))
    print("Mean:", np.mean(data))
    print("Median:", np.median(data))
    print("Max:", np.max(data))

def save_distributions(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Distributions saved to {filename}")

if __name__ == '__main__':
    test_folders = [
        "../e2e_tests/happy_paths",
        "../e2e_tests/multi_skill",
        "../e2e_tests/corrections",
        "../e2e_tests/context_switch",
        "../e2e_tests/cancellations",
    ]
    predicted_tests = []
    results_dir = "./results"  # Directory for storing results
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist

    for folder in test_folders:
        print(f"Running tests in {folder}")
        print("==============================\n")
        if not os.path.exists(folder):
            print("Skipping since folder does not exist\n")
            continue

        test_paths = get_test_paths(folder)

        all_test_cases = []
        for path in test_paths:
            test = load_test(path)
            if test and "test_cases" in test:
                all_test_cases += test["test_cases"]

        all_stats = []

        for test in all_test_cases:
            initialize_graph()
            reset_db()
            name, runtime_info = run_test(test)
            predicted_test = dump_as_convo(name, runtime_info)
            predicted_tests.append(predicted_test)
            all_stats += runtime_info

        all_input_tokens = [stat["tokens"][0] for stat in all_stats if stat.get("success")]
        all_output_tokens = [stat["tokens"][1] for stat in all_stats if stat.get("success")]
        all_costs = [stat["cost"] for stat in all_stats if stat.get("success")]
        all_latencies = [stat["latency"] for stat in all_stats if stat.get("success")]

        print(f"Runtime stats for {folder}")
        print("==============================")
        print("Cost stats:")
        print_metric_stats(all_costs)

        print("------------------------------")

        print("Input tokens stats:")
        print_metric_stats(all_input_tokens)

        print("------------------------------")

        print("Output tokens stats:")
        print_metric_stats(all_output_tokens)

        print("------------------------------")

        print("Latency stats:")
        print_metric_stats(all_latencies)

        print("------------------------------")
        print("\n")

        # Save the distributions to JSON files in the results directory
        folder_name = os.path.basename(folder.rstrip('/'))
        save_distributions(all_costs, os.path.join(results_dir, f'{folder_name}_costs.json'))
        save_distributions(all_input_tokens, os.path.join(results_dir, f'{folder_name}_input_tokens.json'))
        save_distributions(all_output_tokens, os.path.join(results_dir, f'{folder_name}_output_tokens.json'))
        save_distributions(all_latencies, os.path.join(results_dir, f'{folder_name}_latencies.json'))

    # Save the predicted tests in the results directory
    with open(os.path.join(results_dir, "predictions.yaml"), "w") as f:
        yaml.dump({"test_cases": predicted_tests}, f)
