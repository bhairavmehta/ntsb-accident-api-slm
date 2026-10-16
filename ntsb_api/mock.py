"""Synthetic sample data for offline development, demos and tests.

These records are INVENTED. They mimic the *shape* of CAROL export JSON so the
whole stack (normalizer, API, UI) can run without network access. Every NTSB
number starts with ``DEMO`` so they can never be mistaken for real cases.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

_MAKES = [("BOEING", ["737-800", "737-9", "777-200ER", "787-9", "767-300"]),
          ("AIRBUS", ["A320-214", "A321-231", "A220-300", "A330-300"]),
          ("EMBRAER", ["ERJ 170-200 LR", "ERJ 175", "EMB-145"]),
          ("BOMBARDIER", ["CL-600-2D24", "DHC-8-402"]),
          ("CESSNA", ["172S", "208B"]), ("PIPER", ["PA-28-181"])]
_OPERATORS = ["Example Air Lines", "Sample Airways", "Demo Regional", "Test Cargo Co",
              "Placeholder Air", "Private owner"]
_PLACES = [("Seattle", "WA"), ("Chicago", "IL"), ("Denver", "CO"), ("Atlanta", "GA"),
           ("Dallas", "TX"), ("Anchorage", "AK"), ("Phoenix", "AZ"), ("Newark", "NJ"),
           ("San Francisco", "CA"), ("Miami", "FL")]
_AIRPORTS = {"Seattle": "SEA", "Chicago": "ORD", "Denver": "DEN", "Atlanta": "ATL",
             "Dallas": "DFW", "Anchorage": "ANC", "Phoenix": "PHX", "Newark": "EWR",
             "San Francisco": "SFO", "Miami": "MIA"}
_SCENARIOS = [
    ("turbulence encounter during descent; a flight attendant was seriously injured",
     "The airplane's encounter with convective turbulence during descent.",
     ["Environmental issues-Weather-Turbulence (convective)-Effect on operation",
      "Personnel issues-Task performance-Use of equip/system-Cabin crew"]),
    ("runway excursion after landing on a contaminated runway",
     "The flight crew's failure to maintain directional control during the landing roll on a contaminated runway.",
     ["Environmental issues-Conditions/weather-Runway contamination", "Personnel issues-Action/decision-Directional control"]),
    ("engine failure shortly after takeoff followed by a return to the departure airport",
     "A fatigue fracture of a fan blade, which resulted in an uncontained engine failure.",
     ["Aircraft-Aircraft power plant-Engine (turbofan)-Fan section-Fatigue/wear/corrosion"]),
    ("thrust reverser did not deploy normally during landing rollout",
     "Slow deployment of the left thrust reverser due to a degraded actuator, resulting in asymmetric deceleration.",
     ["Aircraft-Aircraft power plant-Thrust reverser system-Malfunction"]),
    ("loss of separation with another aircraft on the taxiway",
     "The ground controller's failure to ensure separation between taxiing aircraft.",
     ["Personnel issues-Action/decision-ATC personnel-Clearance"]),
    ("hard landing in gusty crosswind conditions resulting in substantial damage",
     "The pilot's improper flare in gusting crosswind conditions, which resulted in a hard landing.",
     ["Environmental issues-Conditions/weather-Gusts", "Personnel issues-Task performance-Landing flare"]),
]


def _make_cases(n: int = 60, seed: int = 7) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    out = []
    start = date(2015, 1, 1)
    for k in range(n):
        make, models = rnd.choice(_MAKES)
        model = rnd.choice(models)
        city, st = rnd.choice(_PLACES)
        when = start + timedelta(days=rnd.randint(0, 3950))
        scen = rnd.choice(_SCENARIOS)
        injury = rnd.choices(["None", "Minor", "Serious", "Fatal"], [5, 3, 3, 1])[0]
        part = "Part 91: General aviation" if make in ("CESSNA", "PIPER") else rnd.choice(
            ["Part 121: Air carrier", "Part 121: Air carrier", "Part 135: Air taxi & commuter"])
        fatal = rnd.randint(1, 4) if injury == "Fatal" else 0
        serious = rnd.randint(1, 3) if injury in ("Serious", "Fatal") else 0
        code = "DCA" if part.startswith("Part 121") else rnd.choice(["WPR", "CEN", "ERA"])
        ntsb = f"DEMO{str(when.year)[2:]}{code}{k:03d}"
        out.append({
            "cm_mkey": 900000 + k,
            "cm_ntsbNum": ntsb,
            "cm_eventDate": when.isoformat() + "T12:00:00Z",
            "cm_eventType": rnd.choices(["ACC", "INC"], [3, 1])[0],
            "cm_mode": "Aviation",
            "cm_city": city, "cm_state": st, "cm_country": "USA",
            "cm_Latitude": round(rnd.uniform(25, 61), 4),
            "cm_Longitude": round(rnd.uniform(-150, -75), 4),
            "airportId": _AIRPORTS[city], "airportName": f"{city} (sample)",
            "cm_highestInjury": injury,
            "cm_fatalInjuryCount": fatal, "cm_seriousInjuryCount": serious,
            "cm_minorInjuryCount": rnd.randint(0, 5) if injury != "None" else 0,
            "cm_completionStatus": rnd.choice(["Completed", "Completed", "In work"]),
            "cm_reportType": "Final" if k % 3 else "Preliminary",
            "cm_originalPublishedDate": (when + timedelta(days=rnd.randint(200, 700))).isoformat(),
            "cm_hasSafetyRec": rnd.random() < 0.2,
            "cm_probableCause": scen[1],
            "prelimNarrative": f"SYNTHETIC SAMPLE. On {when:%B %d, %Y}, a {make.title()} {model} "
                               f"experienced a {scen[0]} near {city}, {st}.",
            "factualNarrative": f"SYNTHETIC SAMPLE. The flight was operating under {part}. "
                                f"The crew reported a {scen[0]}. Data from the flight data recorder "
                                f"and crew interviews were reviewed.",
            "analysisNarrative": "SYNTHETIC SAMPLE. Analysis text would appear here for completed investigations.",
            "cm_findings": [{"findingText": t, "causeFactor": "C" if j == 0 else "F"}
                            for j, t in enumerate(scen[2])],
            "cm_vehicles": [{
                "make": make, "model": model,
                "registrationNumber": f"N{rnd.randint(100, 999)}{rnd.choice('ABCDEFGHJK')}{rnd.choice('ABCDEFGHJK')}",
                "aircraftCategory": "AIR",
                "damageLevel": rnd.choice(["None", "Minor", "Substantial", "Destroyed"]),
                "operatorName": rnd.choice(_OPERATORS[:5]) if part.startswith("Part 1") else "Private owner",
                "regulationFlightConductedUnder": part,
                "numberOfEngines": 2 if make not in ("CESSNA", "PIPER") else 1,
            }],
        })
    return sorted(out, key=lambda c: c["cm_eventDate"], reverse=True)


MOCK_CASES = _make_cases()


def mock_docket_html(ntsb_number: str) -> str:
    rows = "".join(
        f'<tr><td>{i}</td><td><a href="Document/docBLOB?ID={4000 + i}&FileExtension=pdf&FileName={t.replace(" ", "_")}">'
        f"{t} (sample)</a></td><td>{p}</td></tr>"
        for i, (t, p) in enumerate([("Operational Factors Factual Report", 24),
                                     ("Airworthiness Group Chairman Factual Report", 41),
                                     ("Flight Data Recorder Specialist's Factual Report", 33),
                                     ("Interview Summaries", 18),
                                     ("Photographs", 12)], start=1))
    return f"<html><body><h1>Docket {ntsb_number}</h1><table>{rows}</table></body></html>"


# ---------------------------------------------------------------------------
# Synthetic multi-page "final report" (mirrors NTSB report section headings)
# ---------------------------------------------------------------------------
_TIMELINES = {
    "turbulence": [("1342", "the flight crew received a pilot report of moderate turbulence along the arrival"),
                   ("1351", "the captain turned on the seat belt sign and asked flight attendants to be seated"),
                   ("1354", "the airplane encountered convective turbulence while descending through 17,000 ft"),
                   ("1355", "a flight attendant in the aft galley fell and was seriously injured"),
                   ("1418", "the airplane landed and was met by emergency medical services")],
    "contaminated": [("0712", "the flight crew received an updated field condition report of compacted snow"),
                     ("0726", "the airplane touched down about 1,500 ft beyond the runway threshold"),
                     ("0727", "the airplane began drifting left as the crew applied reverse thrust"),
                     ("0727", "the airplane exited the left side of the runway at about 40 knots"),
                     ("0729", "aircraft rescue and firefighting personnel arrived at the airplane")],
    "fan blade": [("1102", "the airplane departed and began its initial climb"),
                  ("1103", "the crew heard a loud bang and the left engine N1 decreased rapidly"),
                  ("1104", "the captain declared an emergency and requested a return"),
                  ("1105", "the crew completed the engine failure checklist and shut down the left engine"),
                  ("1121", "the airplane landed on the departure runway without further incident")],
    "thrust reverser": [("1630", "the airplane touched down within the touchdown zone"),
                        ("1630", "the crew selected reverse thrust on both engines"),
                        ("1630", "the left thrust reverser deployed about 4 seconds later than the right"),
                        ("1631", "the airplane yawed right and the captain applied left rudder and braking"),
                        ("1632", "the airplane stopped on the runway centerline area and taxied to the gate")],
    "separation": [("2214", "the ground controller cleared the airplane to taxi via taxiway B"),
                   ("2216", "a regional jet was cleared to cross taxiway B at the same intersection"),
                   ("2217", "the captain observed the regional jet and applied maximum braking"),
                   ("2217", "the airplanes came within about 60 ft of each other")],
    "flare": [("1505", "the pilot reported winds from 240 degrees at 18 knots gusting 29 knots"),
              ("1511", "the airplane crossed the threshold about 10 knots above the reference speed"),
              ("1512", "the airplane touched down hard and bounced"),
              ("1512", "the second touchdown resulted in substantial damage to the fuselage")],
}


def _timeline_for(cause: str) -> list[tuple[str, str]]:
    c = cause.lower()
    for key, tl in _TIMELINES.items():
        if key in c:
            return tl
    return _TIMELINES["flare"]


def mock_report_blocks(raw: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    v = raw["cm_vehicles"][0]
    tl = _timeline_for(raw["cm_probableCause"])
    when = raw["cm_eventDate"][:10]
    history = " ".join(f"At {t} local time, {e}." for t, e in tl)
    findings = "\n".join(f"{f['findingText']} - {'C' if f['causeFactor'] == 'C' else 'F'}"
                         for f in raw["cm_findings"])
    title = f"Aviation Investigation Final Report - {raw['cm_ntsbNum']} (SYNTHETIC SAMPLE)"
    blocks = [
        ("p", f"Location: {raw['cm_city']}, {raw['cm_state']}    Accident Number: {raw['cm_ntsbNum']}\n"
              f"Date: {when}    Aircraft: {v['make'].title()} {v['model']}\n"
              f"Defining Event: {raw['prelimNarrative'].split('experienced a ')[-1].split(' near')[0]}\n"
              f"Injuries: {raw['cm_fatalInjuryCount']} Fatal, {raw['cm_seriousInjuryCount']} Serious, "
              f"{raw['cm_minorInjuryCount']} Minor\n"
              f"Flight Conducted Under: {v['regulationFlightConductedUnder']}"),
        ("h2", "Analysis"),
        ("p", f"This synthetic sample report illustrates the structure of an NTSB final report. "
              f"The {v['make'].title()} {v['model']} operated by {v['operatorName']} was involved in a "
              f"{raw['prelimNarrative'].split('experienced a ')[-1]} "
              f"Recorded flight data were consistent with the crew's statements. "
              f"Examination of the airplane found damage classified as {v['damageLevel'].lower()}. "
              f"The investigation identified the probable cause as {raw['cm_probableCause'][0].lower() + raw['cm_probableCause'][1:]}"),
        ("p", "Crew duty times were within regulatory limits and there was no evidence of impairment. "
              "The operator's procedures were reviewed and found to address the circumstances of the event, "
              "although guidance on the relevant checklist items was not emphasized during recurrent training."),
        ("h2", "Probable Cause and Findings"),
        ("p", "The National Transportation Safety Board determines the probable cause(s) of this accident to be: "
              + raw["cm_probableCause"]),
        ("h2", "Findings"),
        ("p", findings),
        ("h2", "Factual Information"),
        ("h2", "History of Flight"),
        ("p", f"On {when}, a {v['make'].title()} {v['model']}, {v['registrationNumber']}, operated by "
              f"{v['operatorName']} under {v['regulationFlightConductedUnder']}, was involved in an "
              f"occurrence near {raw['cm_city']}, {raw['cm_state']}. " + history),
        ("h2", "Pilot Information"),
        ("p", "The captain held an airline transport pilot certificate with about 11,200 total hours of flight "
              "experience, including 4,300 hours in type. The first officer held an airline transport pilot "
              "certificate with about 3,900 total hours, including 1,100 hours in type."),
        ("h2", "Aircraft and Owner/Operator Information"),
        ("p", f"Aircraft make: {v['make'].title()}; model: {v['model']}; registration: {v['registrationNumber']}; "
              f"engines: {v['numberOfEngines']}. Maintenance records showed no open discrepancies relevant to the event."),
        ("h2", "Meteorological Information and Flight Plan"),
        ("p", "The weather observation closest to the time of the event reported visibility 10 statute miles, "
              "scattered clouds at 4,500 ft, and wind from 240 degrees at 14 knots."),
        ("h2", "Wreckage and Impact Information"),
        ("p", f"Aircraft damage: {v['damageLevel']}. Crew and passenger injuries are summarized in the table of "
              f"injuries: {raw['cm_fatalInjuryCount']} fatal, {raw['cm_seriousInjuryCount']} serious, "
              f"{raw['cm_minorInjuryCount']} minor."),
        ("h2", "Administrative Information"),
        ("p", "Investigator In Charge (IIC): Sample Investigator. Report date: synthetic. "
              "This document is generated sample data for software testing and is not an NTSB product."),
    ]
    return title, blocks


def mock_report_pdf(raw: dict[str, Any]) -> bytes:
    from .pdfwrite import text_pdf
    title, blocks = mock_report_blocks(raw)
    return text_pdf(blocks, title=title)
