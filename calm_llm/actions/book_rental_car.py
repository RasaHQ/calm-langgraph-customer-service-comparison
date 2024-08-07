from typing import Any, Dict, Text

from datetime import datetime
from typing import List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

from actions.change_flight import parse_datetime
from actions.db import search_car_rentals


class SearchCarRentals(Action):
    def name(self) -> str:
        return "search_car_rentals"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        location = tracker.get_slot("trip_destination")
        name = tracker.get_slot("car_rental_name")
        price_tier = tracker.get_slot("car_rental_price_tier")
        start_date = tracker.get_slot("car_rental_start_date")
        end_date = tracker.get_slot("car_rental_end_date")
        if price_tier is 'Any':
            price_tier = None
        if name is 'Any':
            name = None
        def format_readable(r):
            return f"* {r['name']} - {r['price_tier']}"

        results = search_car_rentals()
        results_readable = "\n".join([format_readable(r) for r in results])
        print(results)
        events = [
            SlotSet("car_rental_search_results", results),
            SlotSet("car_rental_search_results_readable", results_readable)
            ]
        return events


class ValidateCarRentalStartDate(Action):
    def name(self) -> str:
        return "validate_car_rental_start_date"

    def validate_date(self, slot_name, tracker, dispatcher):
        current_time = datetime.strptime("2024-04-27 18:13:47.329404 +0530", "%Y-%m-%d %H:%M:%S.%f %z")
        current_value = tracker.get_slot(slot_name)
        if not current_value:
            return []
        try:
            start_date = parse_datetime(current_value)
            if start_date is None or (start_date and start_date < current_time):
                dispatcher.utter_message(response="utter_invalid_date")
                return [SlotSet(slot_name, None)]
        except Exception as e:
            dispatcher.utter_message(response="utter_invalid_date")
            return [SlotSet(slot_name, None)]

        return [
            SlotSet(slot_name, start_date.isoformat()),
            SlotSet(f"{slot_name}_readable", start_date.strftime("%A, %B %d"))
            ]

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        return self.validate_date("car_rental_start_date", tracker, dispatcher)


class ValidateCarRentalEndDate(ValidateCarRentalStartDate):
    def name(self) -> str:
        return "validate_car_rental_end_date"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> List[EventType]:
        return self.validate_date("car_rental_end_date", tracker, dispatcher)
