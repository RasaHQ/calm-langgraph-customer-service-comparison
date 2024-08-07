from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import datetime

from actions.db import search_flights
from dateutil.parser import parse
import pytz

TZOFFSETS = {"BRST": -10800}


def parse_datetime(text: str):
    time = None
    if not text:
        return time
    try:
        time = datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f %z")
    except ValueError:
        time = parse(text, tzinfos=TZOFFSETS)
        if not time.tzinfo:
            timezone = pytz.timezone('America/New_York')
            time = timezone.localize(time)
    return time


class SearchFlights(Action):
    def name(self) -> str:
        return "search_flights"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:


        after_date = parse_datetime(tracker.get_slot("flight_search_start_date"))
        print(after_date)
        before_date = tracker.get_slot("flight_search_end_date")
        print(before_date)
        if before_date != 'Any':
            before_date = parse_datetime(before_date)
        else:
            before_date = (after_date + datetime.timedelta(days=6))

        print(before_date, after_date)

        booking = tracker.get_slot('unique_booking')
        departure_airport = booking['departure_airport']
        arrival_airport = booking['arrival_airport']

        def prettify_results(results):

            results = sorted(results, key=lambda x: datetime.datetime.fromisoformat(x['scheduled_departure']).strftime('%-m/%-d %-I:%M %p'))

            return "\n".join([
                "* " + datetime.datetime.fromisoformat(r['scheduled_departure']).strftime("%A, %B %d at %-I:%M %p") + f" ({r['flight_id']})"
                for r in results
            ])
        
        results = search_flights(departure_airport, arrival_airport, after_date.isoformat(), before_date.isoformat(), limit=3)

        results_readable = prettify_results(results)
        return [SlotSet("flight_search_results", results), SlotSet("results_readable", results_readable)]

class NewFlightDetails(Action):
    def name(self) -> str:
        return "new_flight_details"
    
    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        passenger_id = "3442 587242"
        new_flight_id = tracker.get_slot("selected_flight_id")
        search_results = tracker.get_slot("flight_search_results")
        new_flight = [f for f in search_results if str(f['flight_id']) == new_flight_id][0]
        new_departure_time = new_flight['scheduled_departure']

        def destination_from_atis(atis):
            return {"BSL": "Basel"}.get(atis, None)
        trip_destination = destination_from_atis(new_flight['arrival_airport'])
        return [
            SlotSet("new_departure_time", datetime.datetime.fromisoformat(new_departure_time).strftime('%A, %B %d at %-I:%M %p')),
            SlotSet("new_flight_number", new_flight["flight_no"]),
            SlotSet("trip_destination", trip_destination)
            ]


class ChangeFlight(Action):
    def name(self) -> str:
        return "change_flight"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        return [SlotSet("change_flight_success", True)]
