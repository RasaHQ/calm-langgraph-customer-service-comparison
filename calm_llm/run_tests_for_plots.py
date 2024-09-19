import os
import sys
import json
import numpy as np
import asyncio
from typing import Dict, List
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from custom_test_runner import CustomE2ETestRunner
    from rasa.cli.e2e_test import read_test_cases, split_into_passed_failed
    from custom_test_runner import TOTAL_COST, COMPLETION_TOKENS, PROMPT_TOKENS, LATENCY

    import rasa.cli.arguments.run
    import rasa.cli.utils
    import rasa.shared.data
    import rasa.shared.utils.cli
    import rasa.shared.utils.io
    from rasa.core.exceptions import AgentNotReady
    from rasa.core.utils import AvailableEndpoints
    from rasa.e2e_test.e2e_test_result import TestResult
    import rasa.utils.io
    from rasa.shared.utils.cli import pad
    from rasa.e2e_test.utils.io import print_failed_case, print_e2e_help, \
        print_test_summary, print_final_line

import structlog
import logging
from rich import print  # Import rich for printing formatted text

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

log = structlog.get_logger()

def save_distributions(statistics: List[Dict[str, List[float]]], filename: str) -> None:
    def flatten(xss, key):
        return [x for xs in xss for x in xs[key]]

    # Extract data from the statistics
    total_cost = flatten(statistics, TOTAL_COST)
    completion_tokens = flatten(statistics, COMPLETION_TOKENS)
    prompt_tokens = flatten(statistics, PROMPT_TOKENS)
    latency = flatten(statistics, LATENCY)

    # Create a dictionary to store all the distributions
    distributions = {
        "total_cost": total_cost,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "latency": latency,
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the distributions to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(distributions, json_file, indent=4)

    print(f"Distributions saved to {filename}")

def print_statistics(statistics: List[Dict[str, List[float]]]) -> None:
    def flatten(xss, key):
        return [x for xs in xss for x in xs[key]]

    def _print_stats(data, name):
        mean = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        std_dev = np.std(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)

        print(f"--- {name} ---")
        print(f"Mean: {mean}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Median: {median}")
        print(f"Standard Deviation: {std_dev}")
        print(f"25th Percentile (P25): {p25}")
        print(f"75th Percentile (P75): {p75}")
        print("---------------------------------\n")

    total_cost = flatten(statistics, TOTAL_COST)
    completion_tokens = flatten(statistics, COMPLETION_TOKENS)
    prompt_tokens = flatten(statistics, PROMPT_TOKENS)
    latency = flatten(statistics, LATENCY)

    print("COST PER USER MESSAGE (USD)")
    _print_stats(total_cost, "Cost")
    print("COMPLETION TOKENS PER USER MESSAGE")
    _print_stats(completion_tokens, "Completion Tokens")
    print("PROMPT TOKENS PER USER MESSAGE")
    _print_stats(prompt_tokens, "Prompt Tokens")
    print("LATENCY PER USER MESSAGE (sec)")
    _print_stats(latency, "Latency")

def print_test_result(
    passed: List[TestResult],
    failed: List[TestResult],
    fail_fast: bool = False,
) -> None:
    """Print the result of the test run.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        fail_fast: If true, stop after the first failure.
    """
    if failed:
        # print failure headline
        print("\n")
        print(f"[bold]{pad('FAILURES', char='=')}[/bold]")

    # print failed test case
    for fail in failed:
        print_failed_case(fail)

    print_test_summary(failed)

    if fail_fast:
        rasa.shared.utils.cli.print_error(pad("stopping after 1 failure", char="!"))
        has_failed = True
    elif len(failed) + len(passed) == 0:
        # no tests were run, print error
        rasa.shared.utils.cli.print_error(pad("no test cases found", char="!"))
        print_e2e_help()
        has_failed = True
    elif failed:
        has_failed = True
    else:
        has_failed = False

    print_final_line(passed, failed, has_failed=has_failed)

def execute_e2e_tests(path_to_test_cases: str, results_dir: str) -> None:
    """Run the end-to-end tests.

    Args:
        path_to_test_cases: Path to the directory containing test cases.
        results_dir: Directory where results will be stored.
    """
    endpoints_path = "endpoints.yml"
    endpoints = AvailableEndpoints.read_endpoints(endpoints_path)

    # Ignore all endpoints apart from action server, model, nlu and nlg
    endpoints.tracker_store = None
    endpoints.lock_store = None
    endpoints.event_broker = None

    model_path = "models"

    test_suite = read_test_cases(path_to_test_cases)

    try:
        test_runner = CustomE2ETestRunner(
            remote_storage=None,
            model_path=model_path,
            model_server=endpoints.model,
            endpoints=endpoints,
        )
    except AgentNotReady as error:
        log.error(msg=error.message)
        sys.exit(1)

    results, statistics = asyncio.run(
        test_runner.run_tests(test_suite.test_cases,
                              test_suite.fixtures,
                              fail_fast=False,
                              input_metadata=test_suite.metadata)
    )

    passed, failed = split_into_passed_failed(results)
    print_statistics(statistics)

    # Extract the folder name from the path and create a results filename
    folder_name = os.path.basename(path_to_test_cases.rstrip('/'))
    save_distributions(statistics, os.path.join(results_dir, f'{folder_name}_distributions.json'))
    print_test_result(passed, failed, fail_fast=False)


if __name__ == '__main__':
    test_folders = [
        ".././e2e_tests/happy_paths",
        ".././e2e_tests/multi_skill",
        ".././e2e_tests/corrections",
        ".././e2e_tests/context_switch",
        ".././e2e_tests/cancellations",
    ]

    results_dir = "./results"  # Directory for storing results
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist

    for folder in test_folders:
        print(f"Running tests from {folder}")
        print("=============================")
        execute_e2e_tests(folder, results_dir)
