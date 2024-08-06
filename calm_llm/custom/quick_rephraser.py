from typing import Any, Dict, Optional, Text
from rasa.shared.utils.llm import (
    tracker_as_readable_transcript,
)
from rasa.core.tracker_store import DialogueStateTracker
from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser


MAX_TURNS_DEFAULT = 8


class QuickResponseRephraser(ContextualResponseRephraser):

    async def _create_history(self, tracker: DialogueStateTracker) -> str:
        """Creates the history for the prompt.

        Args:
            tracker: The tracker to use for the history.


        Returns:
        The history for the prompt.
        """
        return tracker_as_readable_transcript(tracker, max_turns=MAX_TURNS_DEFAULT)
