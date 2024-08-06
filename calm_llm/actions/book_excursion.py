from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import search_trip_recommendations

class SearchTripRecommendations(Action):
    def name(self) -> str:
        return "search_trip_recommendations"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        location = tracker.get_slot("trip_destination")
        results = search_trip_recommendations(location)
        results_readable = ", ".join([e['name'] for e in results[:10]])
        return [SlotSet("excursion_search_results_readable", str(results_readable))]
