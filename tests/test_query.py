from datetime import date

import pytest
from pydantic import ValidationError

from ntsb_api.query import CaseFilters, Rule, export_payload, search_payload


def test_filters_to_rules_and_payload():
    f = CaseFilters(make="Boeing", date_from=date(2020, 1, 1), airline_only=True, state="wa")
    rules = f.to_rules()
    cols = [(r.column, r.operator, r.value) for r in rules]
    assert ("Event.Mode", "is", "Aviation") in cols
    assert ("Aircraft.AircraftMake", "contains", "Boeing") in cols
    assert ("Event.EventDate", "is on or after", "2020-01-01") in cols
    assert ("Event.State", "is", "WA") in cols
    assert any(c[0] == "AviationOperation.RegulationFlightConductedUnder" for c in cols)

    p = search_payload(rules, limit=10, offset=20)
    assert p["ResultSetSize"] == 10 and p["ResultSetOffset"] == 20
    rule = p["QueryGroups"][0]["QueryRules"][1]
    assert rule["Columns"] == ["Aircraft.AircraftMake"] and rule["selectedOption"]["InputType"] == "Text"
    assert export_payload(rules)["ExportFormat"] == "data"


def test_rule_validation():
    with pytest.raises(ValidationError):
        Rule(column="Aircraft.Nope", value="x")
    with pytest.raises(ValidationError):
        Rule(column="Event.State", operator="contains", value="WA")  # dropdown: is / is not only
