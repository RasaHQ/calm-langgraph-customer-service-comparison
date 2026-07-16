from custom_test_runner import TOTAL_COST, COMPLETION_TOKENS, PROMPT_TOKENS
from rasa.dialogue_understanding.generator import SingleStepLLMCommandGenerator
from typing import Any, List, Optional

import structlog
from rasa.dialogue_understanding.commands import (
    Command
)
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    LLMInput,
)

COMMAND_PROMPT_FILE_NAME = "command_prompt.jinja2"
DEFAULT_LLM_CONFIG = {
    "provider": "openai",
    "model": DEFAULT_OPENAI_CHAT_MODEL_NAME,
    "temperature": 0.0,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    "timeout": 20,
}
structlogger = structlog.get_logger()

# Fallback rate card when litellm cannot price the model (gpt-5.2 rates).
_GPT4_INPUT_PER_TOKEN = 0.00000175  # $1.75 / 1M
_GPT4_OUTPUT_PER_TOKEN = 0.000014  # $14 / 1M


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class CustomLLMCommandGenerator(SingleStepLLMCommandGenerator):
    """Tracks prompt/completion tokens and estimated USD cost on the NLU message."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending_message: Optional[Message] = None

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        self._pending_message = message
        try:
            return await super().predict_commands(message, flows, tracker, **kwargs)
        finally:
            self._pending_message = None

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[LLMResponse]:
        response = await super().invoke_llm(llm_input)
        self._record_usage(response)
        return response

    def _record_usage(self, response: Optional[LLMResponse]) -> None:
        message = self._pending_message
        if message is None or response is None or response.usage is None:
            return

        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0
        total_cost = self._estimate_cost(
            prompt_tokens, completion_tokens, response.model
        )

        message.set(COMPLETION_TOKENS, completion_tokens, add_to_output=True)
        message.set(PROMPT_TOKENS, prompt_tokens, add_to_output=True)
        message.set(TOTAL_COST, total_cost, add_to_output=True)

    @staticmethod
    def _estimate_cost(
        prompt_tokens: int, completion_tokens: int, model: Optional[str]
    ) -> float:
        try:
            import litellm

            cost = litellm.completion_cost(
                completion_response={
                    "model": model or "gpt-4",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
            if cost is not None:
                return float(cost)
        except Exception as e:
            structlogger.debug(
                "custom_llm_command_generator.cost_estimate.fallback",
                error=str(e),
            )

        return (
            prompt_tokens * _GPT4_INPUT_PER_TOKEN
            + completion_tokens * _GPT4_OUTPUT_PER_TOKEN
        )
