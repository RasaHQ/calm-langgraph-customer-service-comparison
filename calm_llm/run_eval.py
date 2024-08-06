import asyncio
import sys
from typing import Dict, List

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from custom_test_runner import CustomE2ETestRunner
    from rasa.cli.e2e_test import read_test_cases, \
        split_into_passed_failed
    from custom_test_runner import TOTAL_COST, COMPLETION_TOKENS, PROMPT_TOKENS, \
        LATENCY

    import rasa.cli.arguments.run
    import rasa.cli.utils
    import rasa.shared.data
    import rasa.shared.utils.cli
    import rasa.shared.utils.io
    import rich
    from rasa.core.exceptions import AgentNotReady
    from rasa.core.utils import AvailableEndpoints
    from rasa.e2e_test.e2e_test_result import TestResult
    import rasa.utils.io
    from rasa.cli.e2e_test import print_failed_case, print_e2e_help, \
        print_test_summary, print_final_line, pad

import structlog
import logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

log = structlog.get_logger()


def print_statistics(statistics: List[Dict[str, List[float]]]) -> None:
    def flatten(xss, key):
        return [x for xs in xss for x in xs[key]]

    def _print_stats(data):
        mean = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        # p75 = np.percentile(data, 75)
        # p25 = np.percentile(data, 25)

        print("---------------------------------")

        print(f"Mean: {mean}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Median: {median}")
        # print(f"25th Percentile (P25): {p25}")
        # print(f"75th Percentile (P75): {p75}")

        print("---------------------------------\n")

    number_of_steps_per_test_case = [len(s.values()) for s in statistics]
    total_cost = flatten(statistics, TOTAL_COST)
    completion_tokens = flatten(statistics, COMPLETION_TOKENS)
    prompt_tokens = flatten(statistics, PROMPT_TOKENS)
    latency = flatten(statistics, LATENCY)

    import numpy as np
    print("COST PER USER MESSAGE (USD)")
    _print_stats(total_cost)
    print("COMPLETION TOKENS PER USER MESSAGE")
    _print_stats(completion_tokens)
    print("PROMPT TOKENS PER USER MESSAGE")
    _print_stats(prompt_tokens)
    print("LATENCY PER USER MESSAGE (sec)")
    _print_stats(latency)


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
        rich.print(f"[bold]{pad('FAILURES', char='=')}[/bold]")

    # print failed test_Case
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


def execute_e2e_tests(path_to_test_cases: str) -> None:
    """Run the end-to-end tests.

    Args:
        args: Commandline arguments.
    """
    endpoints_path = "endpoints.yml"
    endpoints = AvailableEndpoints.read_endpoints(endpoints_path)

    # Ignore all endpoints apart from action server, model, nlu and nlg
    # to ensure InMemoryTrackerStore is being used instead of production
    # tracker store
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
    print_test_result(passed, failed, fail_fast=False)


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
        print(f"Running tests from {folder}")
        print("=============================")
        execute_e2e_tests(folder)
