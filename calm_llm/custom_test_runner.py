from rasa.e2e_test.e2e_test_runner import E2ETestRunner, get_fixtures_for_test_case
import logging
import time
from asyncio import CancelledError
from typing import Dict, List, Text, Tuple, Union, Optional, Any

import rasa.shared.utils.io
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.shared.core.events import UserUttered

from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Metadata,
    TestCase,
    TestCaseFixtures,
    TestStep,
)

from rasa.e2e_test.e2e_test_result import (
    TestResult,
)

from rasa.telemetry import track_e2e_test_run

logger = logging.getLogger(__name__)
TEST_TURNS_TYPE = Dict[int, Union[TestStep, ActualStepOutput]]

COMPLETION_TOKENS = "completion_tokens"
TOTAL_COST = "total_cost"
PROMPT_TOKENS = "prompt_tokens"
LATENCY = "latency"


class CustomE2ETestRunner(E2ETestRunner):

    async def run_prediction_loop(
        self,
        collector: CollectingOutputChannel,
        steps: List[TestStep],
        sender_id: Text,
        test_case_metadata: Optional[Metadata] = None,
        input_metadata: Optional[List[Metadata]] = None,
    ) -> Tuple[TEST_TURNS_TYPE, Dict[str, List[float]]]:
        """Runs dialogue prediction and collects per-turn cost/latency stats."""
        turns: TEST_TURNS_TYPE = {}
        event_cursor = 0

        test_case_statistics = {
            LATENCY: [],
            TOTAL_COST: [],
            COMPLETION_TOKENS: [],
            PROMPT_TOKENS: [],
        }

        tracker = await self.agent.processor.fetch_tracker_with_initial_session(  # type: ignore[union-attr]
            sender_id
        )
        turns[-1], event_cursor = self.get_actual_step_output(
            tracker, TestStep(actor="bot", text=None), event_cursor
        )

        for position, step in enumerate(steps):
            if step.actor != "user":
                turns[position] = step
                continue
            elif not step.text:
                rasa.shared.utils.io.raise_warning(
                    f"The test case '{sender_id}' contains a `user` step in line "
                    f"{position + 1} without a text value. "
                    f"Skipping this step and proceeding to the next user step.",
                    UserWarning,
                )
                continue

            start = time.time()
            try:
                await self.agent.handle_message(
                    UserMessage(
                        step.text,
                        collector,
                        sender_id,
                    )
                )
            except CancelledError:
                logger.error(
                    f"Message handling timed out for user message '{step.text}'.",
                    exc_info=True,
                )
            except Exception:
                logger.exception(
                    f"An exception occurred while handling "
                    f"user message '{step.text}'."
                )
            end = time.time()

            tracker = await self.agent.tracker_store.retrieve(sender_id)
            latest = tracker.latest_message if tracker else None
            self._update_test_case_statistics(
                test_case_statistics, latest, start, end
            )

            turns[position], event_cursor = self.get_actual_step_output(
                tracker, step, event_cursor
            )
        return turns, test_case_statistics

    @staticmethod
    def _update_test_case_statistics(
        test_case_statistics: Dict[str, List[float]],
        message: Optional[UserUttered],
        start: float,
        end: float,
    ) -> None:
        test_case_statistics[LATENCY].append(end - start)
        parse_data = getattr(message, "parse_data", None) or {}
        test_case_statistics[TOTAL_COST].append(parse_data.get(TOTAL_COST, 0) or 0)
        test_case_statistics[COMPLETION_TOKENS].append(
            parse_data.get(COMPLETION_TOKENS, 0) or 0
        )
        test_case_statistics[PROMPT_TOKENS].append(
            parse_data.get(PROMPT_TOKENS, 0) or 0
        )

    async def run_tests(
        self,
        input_test_cases: List[TestCase],
        fixtures_per_test: List[TestCaseFixtures],
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Tuple[List["TestResult"], List[Dict[str, List[float]]]]:
        """Runs the test cases and returns results plus per-case statistics."""
        results = []
        statistics = []
        input_metadata = kwargs.get("input_metadata", None)

        track_e2e_test_run(input_test_cases, fixtures_per_test, input_metadata)

        for test_case in input_test_cases:
            collector = CollectingOutputChannel()
            sender_id = self.generate_sender_id(test_case.name)

            input_fixtures = get_fixtures_for_test_case(test_case, fixtures_per_test)
            if input_fixtures:
                await self.set_up_fixtures(input_fixtures, sender_id)

            test_case_metadata = None
            if input_metadata:
                test_case_metadata = self.filter_metadata_for_input(
                    test_case.metadata_name, input_metadata
                )

            turns, test_case_statistics = await self.run_prediction_loop(
                collector,
                test_case.steps,
                sender_id,
                test_case_metadata,
                input_metadata,
            )
            statistics.append(test_case_statistics)

            if not test_case.uses_assertions():
                test_result = self.generate_test_result(turns, test_case)
            else:
                test_result = await self.run_assertions(
                    sender_id, test_case, input_metadata
                )
            results.append(test_result)

            if fail_fast and not test_result.pass_status:
                break

        return results, statistics
