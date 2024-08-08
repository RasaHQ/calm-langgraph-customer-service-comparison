from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import datetime

from actions.db import get_flight_bookings


class ListFlightBookings(Action):
    def name(self) -> str:
        return "list_flight_bookings"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        passenger_id = "3442 587242"
        bookings = get_flight_bookings(passenger_id)

        def destination_from_atis(atis):
            return {"BSL": "Basel", "CDG": "Paris"}.get(atis, None)

        def format_readable(booking):
            date_readable = datetime.datetime.fromisoformat(booking['scheduled_departure']).strftime('%B %-d')
            return (
                f"a flight from {destination_from_atis(booking['departure_airport'])}"
                f" ({booking['departure_airport']}) to "
                f"{destination_from_atis(booking['arrival_airport'])}"
                f" ({booking['arrival_airport']}) on {date_readable}"
            )
        bookings_readable = "\n\n ".join([format_readable(b) for b in bookings])
        events = []
        if len(bookings) > 0:
            events.append(SlotSet("bookings_list", bookings))
            events.append(SlotSet("bookings_list_readable", bookings_readable))
        if len(bookings) == 1:
            events.append(SlotSet("unique_booking", bookings[0]))
            trip_destination = destination_from_atis(bookings[0]['arrival_airport'])
            events.append(SlotSet("trip_destination", trip_destination))

        return events
