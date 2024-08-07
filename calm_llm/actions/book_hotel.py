from typing import Any, Dict, Text

from typing import List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from actions.db import search_hotels
from actions.book_rental_car import ValidateCarRentalStartDate


class SearchHotels(Action):
    def name(self) -> str:
        return "search_hotels"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        location = tracker.get_slot("trip_destination")
        results = search_hotels(location)
        def format_readable(result):
            return f"* {result['name']} - {result['price_tier']}"
        print(results)
        results_readable = "\n".join([format_readable(r) for r in results[:3]])
        events = [
            SlotSet("hotel_search_results", results),
            SlotSet("hotel_search_results_readable", results_readable)
            ]
        return events


class ValidateHotelEndDate(ValidateCarRentalStartDate):
    def name(self) -> str:
        return "validate_hotel_end_date"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        return self.validate_date("hotel_end_date", tracker, dispatcher)


class ValidateHotelStartDate(ValidateCarRentalStartDate):
    def name(self) -> str:
        return "validate_hotel_start_date"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        return self.validate_date("hotel_start_date", tracker, dispatcher)
