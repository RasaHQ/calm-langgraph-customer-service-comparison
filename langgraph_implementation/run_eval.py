import yaml, os
import numpy as np
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
    user_messages = []
    for step in steps:
        if step.get("bot") or step.get("utter"):
            continue
        else:
            user_messages.append(step.get("user"))
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


if __name__ == '__main__':

    test_folders = [
            "../e2e_tests/happy_paths",
            "../e2e_tests/multi_skill",
            "../e2e_tests/corrections",
            "../e2e_tests/context_switch",
            "../e2e_tests/cancellations",
        ]
    predicted_tests = []
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

    with open("predictions.yaml", "w") as f:
        yaml.dump({"test_cases": predicted_tests}, f)
