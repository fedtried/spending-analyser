from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


@dataclass(frozen=True)
class AppConfig:
    currency: str = "£"
    locale: str = "en_GB"
    gemini_api_key: Optional[str] = None


def load_config() -> AppConfig:
    """Load configuration from Streamlit secrets with safe defaults.

    Handles missing `.streamlit/secrets.toml` gracefully, returning defaults.
    """
    # Accessing st.secrets can raise FileNotFoundError if no secrets file exists.
    try:
        raw_secrets = st.secrets  # type: ignore[attr-defined]
    except FileNotFoundError:
        raw_secrets = {}
    except Exception:
        raw_secrets = {}

    # Normalize to dict for easy access
    try:
        secrets: dict = dict(raw_secrets) if raw_secrets else {}
    except Exception:
        secrets = {}

    api_section = secrets.get("api", {}) if isinstance(secrets, dict) else {}
    app_section = secrets.get("app", {}) if isinstance(secrets, dict) else {}

    currency = app_section.get("currency", "£")
    locale = app_section.get("locale", "en_GB")
    gemini_api_key = api_section.get("gemini_api_key")

    return AppConfig(currency=currency, locale=locale, gemini_api_key=gemini_api_key)
