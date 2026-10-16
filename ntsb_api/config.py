"""Runtime configuration (all overridable with environment variables)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- NTSB upstream endpoints -------------------------------------------------
    # Public CAROL query API (the same JSON API that powers data.ntsb.gov search).
    carol_base: str = os.getenv("NTSB_CAROL_BASE", "https://data.ntsb.gov/carol-main-public/api/Query")
    # Report generator for aviation case PDFs (keyed by the internal case key, "Mkey").
    report_base: str = os.getenv("NTSB_REPORT_BASE", "https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain")
    # Public docket web app.
    docket_base: str = os.getenv("NTSB_DOCKET_BASE", "https://data.ntsb.gov/Docket")

    # --- Behaviour ---------------------------------------------------------------
    mock: bool = _bool("NTSB_MOCK", False)  # serve synthetic data, never call NTSB
    timeout_s: float = float(os.getenv("NTSB_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("NTSB_MAX_RETRIES", "3"))
    cache_ttl_s: int = int(os.getenv("NTSB_CACHE_TTL", "900"))
    # Minimum spacing between upstream calls, to be a polite client of a public service.
    min_interval_s: float = float(os.getenv("NTSB_MIN_INTERVAL", "0.5"))
    user_agent: str = os.getenv(
        "NTSB_USER_AGENT",
        "ntsb-accident-api/1.0 (+https://github.com/; research client)",
    )
    # CAROL dropdown value used for the "airline only" preset (14 CFR Part 121).
    airline_regulation_value: str = os.getenv("NTSB_AIRLINE_REG", "Part 121: Air carrier")
    cors_origins: list[str] = field(
        default_factory=lambda: [o for o in os.getenv("NTSB_CORS", "*").split(",") if o]
    )


settings = Settings()
