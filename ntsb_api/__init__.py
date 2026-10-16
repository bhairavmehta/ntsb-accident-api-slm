"""NTSB Aviation Accident API - client, REST service, CLI and UI over NTSB CAROL data."""
__version__ = "1.0.0"

from .client import NTSBClient, NTSBError  # noqa: E402,F401
from .query import CaseFilters, Rule  # noqa: E402,F401
