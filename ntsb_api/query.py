"""Build NTSB CAROL query payloads.

CAROL's query builder posts a JSON document made of *query groups*, each holding
*query rules*. A rule targets a column such as ``Aircraft.AircraftMake`` with an
operator such as ``contains`` and a list of values. This module turns friendly,
typed filters into that payload, and validates columns/operators against the
field catalog shipped in ``data/fields.json``.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from importlib import resources
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings

Operator = Literal[
    "is", "is not", "contains", "does not contain", "starts with", "ends with",
    "is greater than", "is less than", "is after", "is before",
    "is on or after", "is on or before",
]


# ---------------------------------------------------------------------------
# Field catalog
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def field_catalog() -> dict[str, dict[str, Any]]:
    """Return {column: field-spec} for every CAROL query-builder field."""
    raw = resources.files("ntsb_api").joinpath("data/fields.json").read_text("utf-8")
    return {f["column"]: f for f in json.loads(raw)["fields"]}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Rule(BaseModel):
    """One low-level CAROL rule, e.g. ``Aircraft.AircraftMake contains BOEING``."""

    column: str = Field(..., examples=["Aircraft.AircraftMake"],
                        description="Catalog column (see GET /api/v1/fields)")
    operator: Operator = Field("contains", examples=["contains"])
    value: str = Field(..., examples=["Boeing"])

    @field_validator("column")
    @classmethod
    def _known_column(cls, v: str) -> str:
        if v not in field_catalog():
            raise ValueError(f"Unknown column '{v}'. See GET /api/v1/fields for valid columns.")
        return v

    def model_post_init(self, __ctx: Any) -> None:  # operator must be valid for the column
        allowed = field_catalog()[self.column]["operators"]
        if self.operator not in allowed:
            raise ValueError(f"Operator '{self.operator}' not allowed for {self.column}; use one of {allowed}")


class CaseFilters(BaseModel):
    """Friendly filters for aviation accident searches. All are optional and AND-ed."""

    text: Optional[str] = Field(None, description="Free text searched in the factual narrative")
    probable_cause: Optional[str] = Field(None, description="Text contained in the probable cause")
    finding: Optional[str] = Field(None, description="Text contained in an NTSB finding")
    ntsb_number: Optional[str] = Field(None, description="NTSB number (contains), e.g. DCA24MA063")
    make: Optional[str] = Field(None, description="Aircraft make (contains), e.g. Boeing")
    model: Optional[str] = Field(None, description="Aircraft model (contains), e.g. 737")
    registration: Optional[str] = Field(None, description="Tail number (contains), e.g. N704AL")
    operator_name: Optional[str] = Field(None, description="Operator / airline name (contains)")
    airport: Optional[str] = Field(None, description="Airport ID (contains), e.g. SEA")
    city: Optional[str] = None
    state: Optional[str] = Field(None, description="US state code, e.g. WA")
    country: Optional[str] = Field(None, description="CAROL country code, e.g. US")
    date_from: Optional[date] = Field(None, description="Event date on or after (YYYY-MM-DD)")
    date_to: Optional[date] = Field(None, description="Event date on or before (YYYY-MM-DD)")
    highest_injury: Optional[Literal["Fatal", "Serious", "Minor", "None"]] = None
    event_type: Optional[Literal["ACC", "INC", "OCC"]] = Field(
        None, description="ACC = accident, INC = incident, OCC = occurrence")
    damage: Optional[Literal["Destroyed", "Substantial", "Minor", "None", "Unknown"]] = None
    weather: Optional[Literal["IMC", "VMC"]] = None
    airline_only: bool = Field(False, description="Restrict to 14 CFR Part 121 air-carrier operations")
    has_safety_rec: Optional[bool] = Field(None, description="Only cases that produced safety recommendations")
    mode: str = Field("Aviation", description="Investigation mode")

    def to_rules(self) -> list[Rule]:
        r: list[Rule] = [Rule(column="Event.Mode", operator="is", value=self.mode)]

        def add(column: str, op: str, value: Any) -> None:
            if value is None or value == "":
                return
            if isinstance(value, bool):
                value = "true" if value else "false"
            r.append(Rule(column=column, operator=op, value=str(value)))

        add("Narrative.Factual", "contains", self.text)
        add("Case.ProbableCause", "contains", self.probable_cause)
        add("Finding.FindingText", "contains", self.finding)
        add("Event.NTSBNumber", "contains", self.ntsb_number and self.ntsb_number.upper())
        add("Aircraft.AircraftMake", "contains", self.make)
        add("Aircraft.AircraftModel", "contains", self.model)
        add("Aircraft.RegistrationNumber", "contains", self.registration and self.registration.upper())
        add("AviationOperation.OperatorName", "contains", self.operator_name)
        add("Airport.AirportId", "contains", self.airport and self.airport.upper())
        add("Event.City", "contains", self.city)
        add("Event.State", "is", self.state and self.state.upper())
        add("Event.Country", "is", self.country and self.country.upper())
        add("Event.EventDate", "is on or after", self.date_from and self.date_from.isoformat())
        add("Event.EventDate", "is on or before", self.date_to and self.date_to.isoformat())
        add("Event.HighestInjury", "is", self.highest_injury)
        add("Event.EventType", "is", self.event_type)
        add("Aircraft.Damage", "is", self.damage)
        add("Weather.AccidentSiteCondition", "is", self.weather)
        if self.airline_only:
            add("AviationOperation.RegulationFlightConductedUnder", "is", settings.airline_regulation_value)
        if self.has_safety_rec is not None:
            add("HasSafetyRec", "is", self.has_safety_rec)
        return r


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------
def _rule_json(rule: Rule) -> dict[str, Any]:
    spec = field_catalog()[rule.column]
    return {
        "RuleType": "Simple",
        "Values": [rule.value],
        "Columns": [rule.column],
        "Operator": rule.operator,
        "selectedOption": {
            "FieldName": spec["name"],
            "DisplayText": "",
            "Columns": [rule.column],
            "Selectable": True,
            "InputType": spec["input"],
            "RuleType": 0,
            "Options": None,
            "TargetCollection": "cases",
            "UnderDevelopment": False,
        },
        "overrideColumn": "",
    }


def _groups(rules: list[Rule], match: str) -> list[dict[str, Any]]:
    return [{
        "QueryRules": [_rule_json(x) for x in rules],
        "AndOr": "and" if match == "all" else "or",
        "inLastSearch": False,
        "editedSinceLastSearch": False,
    }]


def search_payload(rules: list[Rule], *, limit: int = 50, offset: int = 0,
                   match: str = "all", sort_column: Optional[str] = None,
                   descending: bool = True) -> dict[str, Any]:
    """Payload for POST {carol}/Main (paged search returning a count)."""
    return {
        "ResultSetSize": limit,
        "ResultSetOffset": offset,
        "QueryGroups": _groups(rules, match),
        "AndOr": "and",
        "SortColumn": sort_column,
        "SortDescending": descending,
        "TargetCollection": "cases",
        "SessionId": 100000,
    }


def export_payload(rules: list[Rule], *, fmt: str = "data", limit: int = 500,
                   match: str = "all") -> dict[str, Any]:
    """Payload for POST {carol}/FileExport (ZIP containing full JSON or summary CSV)."""
    return {
        "QueryGroups": _groups(rules, match),
        "AndOr": "and",
        "TargetCollection": "cases",
        "ExportFormat": fmt,  # "data" -> JSON, "summary" -> CSV
        "SessionId": 100100,
        "ResultSetSize": limit,
        "SortDescending": True,
    }
