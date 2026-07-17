import asyncio
import sys
from pathlib import Path
from typing import Dict, List

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from custom_test_runner import CustomE2ETestRunner
    from rasa.e2e_test.utils.io import (
        read_test_cases,
        split_into_passed_failed,
        print_failed_case,
        print_e2e_help,
        print_test_summary,
        print_final_line,
    )
    from custom_test_runner import TOTAL_COST, COMPLETION_TOKENS, PROMPT_TOKENS, \
        LATENCY

    import rasa.shared.utils.cli
    import rich
    from rasa.core.exceptions import AgentNotReady
    from rasa.core.config.configuration import Configuration, EndpointsConfigPath
    from rasa.e2e_test.e2e_test_result import TestResult
    from rasa.shared.utils.cli import pad

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
        import numpy as np
        mean = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)

        print("---------------------------------")
        print(f"Mean: {mean}")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Median: {median}")
        print("---------------------------------\n")

    total_cost = flatten(statistics, TOTAL_COST)
    completion_tokens = flatten(statistics, COMPLETION_TOKENS)
    prompt_tokens = flatten(statistics, PROMPT_TOKENS)
    latency = flatten(statistics, LATENCY)

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
    if failed:
        print("\n")
        rich.print(f"[bold]{pad('FAILURES', char='=')}[/bold]")

    for fail in failed:
        print_failed_case(fail)

    print_test_summary(failed)

    if fail_fast:
        rasa.shared.utils.cli.print_error(pad("stopping after 1 failure", char="!"))
        has_failed = True
    elif len(failed) + len(passed) == 0:
        rasa.shared.utils.cli.print_error(pad("no test cases found", char="!"))
        print_e2e_help()
        has_failed = True
    elif failed:
        has_failed = True
    else:
        has_failed = False

    print_final_line(passed, failed, has_failed=has_failed)


def execute_e2e_tests(path_to_test_cases: str) -> None:
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=EndpointsConfigPath.validate(Path("endpoints.yml"))
    ).endpoints

    # Prefer InMemoryTrackerStore for isolated test runs
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
        log.error(msg=getattr(error, "message", str(error)))
        sys.exit(1)

    results, statistics = asyncio.run(
        test_runner.run_tests(
            test_suite.test_cases,
            test_suite.fixtures_per_test,
            fail_fast=False,
            input_metadata=test_suite.metadata,
        )
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
    for folder in test_folders:
        print(f"Running tests from {folder}")
        print("=============================")
        execute_e2e_tests(folder)
