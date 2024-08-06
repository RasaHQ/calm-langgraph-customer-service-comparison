from rasa.e2e_test.e2e_test_runner import E2ETestRunner
import datetime
import logging
import time
from asyncio import CancelledError
from typing import Dict, List, Text, Tuple, Union, Optional, Any

import rasa.shared.utils.io
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.shared.core.events import UserUttered

from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
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
        """Runs dialogue prediction.

        Args:
            collector: Output channel.
            steps: List of steps to run.
            sender_id: The test case name with added timestamp suffix.

        Returns:
        Test turns: {turn_sequence (int) : TestStep or ActualStepOutput}.
        """
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
        # turn -1 i used to contain events that happen during
        # the start of the session and before the first user message
        # TestStep is a placeholder just for the sake of having a turn
        # to specify the actor
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
            self._update_test_case_statistics(
                test_case_statistics, tracker.latest_message, start, end
            )

            turns[position], event_cursor = self.get_actual_step_output(
                tracker, step, event_cursor
            )
        return turns, test_case_statistics

    @staticmethod
    def _update_test_case_statistics(
        test_case_statistics: Dict[str, List[float]],
        message: UserUttered,
        start: float,
        end: time,
    ) -> None:
        test_case_statistics[LATENCY].append(end - start)
        test_case_statistics[TOTAL_COST].append(message.parse_data.get(TOTAL_COST, 0))
        test_case_statistics[COMPLETION_TOKENS].append(
            message.parse_data.get(COMPLETION_TOKENS, 0)
        )
        test_case_statistics[PROMPT_TOKENS].append(
            message.parse_data.get(PROMPT_TOKENS, 0)
        )

    async def run_tests(
        self,
        input_test_cases: List[TestCase],
        input_fixtures: List[Fixture],
        fail_fast: bool = False,
        **kwargs: Any,
    ) -> Tuple[List["TestResult"], List[Dict[str, List[float]]]]:
        """Runs the test cases.

        Args:
            input_test_cases: Input test cases.
            input_fixtures: Input fixtures.
            fail_fast: Whether to fail fast.

        Returns:
        List of test results.
        """
        results = []
        statistics = []
        input_metadata = kwargs.get("input_metadata", None)

        # telemetry call for tracking test runs
        track_e2e_test_run(input_test_cases, input_fixtures, input_metadata)

        for test_case in input_test_cases:
            collector = CollectingOutputChannel()

            # add timestamp suffix to ensure sender_id is unique
            sender_id = f"{test_case.name}_{datetime.datetime.now()}"

            if input_fixtures:
                test_fixtures = self.filter_fixtures_for_test_case(
                    test_case, input_fixtures
                )
                await self.set_up_fixtures(test_fixtures, sender_id)

            tracker, test_case_statistics = await self.run_prediction_loop(
                collector, test_case.steps, sender_id
            )
            statistics.append(test_case_statistics)

            test_result = self.generate_test_result(tracker, test_case)
            results.append(test_result)

            if fail_fast and not test_result.pass_status:
                break

        return results, statistics
