from typing import Any, Dict, Text

from datetime import datetime
from typing import List
import re

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

    # When the user reuses dates from the other booking, the command generator may
    # pass through a relative phrase instead of the ISO value — resolve it here.
    _SAME_DATE_PHRASES = (
        "same date",
        "same dates",
        "those dates",
        "same as the car",
        "same as the hotel",
        "as the car rental",
        "as the hotel",
        "as the car",
    )

    _CROSS_SLOT = {
        "hotel_start_date": "car_rental_start_date",
        "hotel_end_date": "car_rental_end_date",
        "car_rental_start_date": "hotel_start_date",
        "car_rental_end_date": "hotel_end_date",
    }

    def _resolve_relative_date(self, slot_name: str, value: Any, tracker: Tracker) -> Any:
        if not isinstance(value, str):
            return value
        lowered = value.lower().strip()
        if not any(phrase in lowered for phrase in self._SAME_DATE_PHRASES):
            return value
        source_slot = self._CROSS_SLOT.get(slot_name)
        if not source_slot:
            return value
        source_value = tracker.get_slot(source_slot)
        return source_value if source_value else value

    def _latest_user_text(self, tracker: Tracker) -> str:
        # Prefer the last user utterance from the event stream — more reliable
        # during slot validation than tracker.latest_message alone.
        for event in reversed(list(tracker.events)):
            if event.get("event") == "user":
                return str(event.get("text") or "").lower()
        message = tracker.latest_message or {}
        return str(message.get("text") or "").lower()

    def _message_has_explicit_date(self, text: str) -> bool:
        if not text:
            return False
        date_markers = (
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
            "nov", "dec", "today", "tomorrow", "tonight",
            "month", "week", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        )
        has_marker = any(marker in text for marker in date_markers) or "/" in text
        has_digit = any(ch.isdigit() for ch in text)
        # "the 24th", "27th" — day ordinals without a month name
        has_day_ordinal = bool(
            re.search(r"\b\d{1,2}(st|nd|rd|th)\b", text)
        )
        return (has_marker and has_digit) or has_day_ordinal

    def _user_authorized_cross_booking_dates(self, tracker: Tracker) -> bool:
        text = self._latest_user_text(tracker)
        return any(phrase in text for phrase in self._SAME_DATE_PHRASES)

    def _reject_silent_cross_copy(
        self, slot_name: str, value: Any, tracker: Tracker
    ) -> bool:
        """Reject date fills that reuse the other booking without user asking.

        Only when the other booking still has dates (persisted via
        reset_after_flow_ends: false). Ungrounded fills with no cross-booking
        context are left to the LLM / flow collect descriptions.
        """
        source_slot = self._CROSS_SLOT.get(slot_name)
        if not source_slot or not tracker.get_slot(source_slot):
            return False
        if self._user_authorized_cross_booking_dates(tracker):
            return False
        text = self._latest_user_text(tracker)
        if self._message_has_explicit_date(text):
            return False
        return True

    def validate_date(self, slot_name, tracker, dispatcher):
        current_time = datetime.strptime(
            "2024-04-27 18:13:47.329404 +0530", "%Y-%m-%d %H:%M:%S.%f %z"
        )
        current_value = tracker.get_slot(slot_name)
        if not current_value:
            return []
        if self._reject_silent_cross_copy(slot_name, current_value, tracker):
            # Force the flow to ask — do not treat as an invalid calendar date.
            return [SlotSet(slot_name, None), SlotSet(f"{slot_name}_readable", None)]
        current_value = self._resolve_relative_date(slot_name, current_value, tracker)
        try:
            start_date = parse_datetime(current_value)
            if start_date is None or (start_date and start_date < current_time):
                dispatcher.utter_message(response="utter_invalid_date")
                return [SlotSet(slot_name, None)]
        except Exception:
            dispatcher.utter_message(response="utter_invalid_date")
            return [SlotSet(slot_name, None)]

        return [
            SlotSet(slot_name, start_date.isoformat()),
            SlotSet(f"{slot_name}_readable", start_date.strftime("%A, %B %d")),
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
