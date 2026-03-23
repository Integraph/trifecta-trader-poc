"""
Startup dependency verification.

Called during app creation to log warnings about missing packages.
The API still starts — but the admin gets an immediate heads-up in the
console and via GET /health/dependencies.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

# (module_name_to_import, pip_install_name, required_for)
REQUIRED_PACKAGES: list[tuple[str, str, str]] = [
    ("langchain_anthropic",   "langchain-anthropic",   "Anthropic LLM provider"),
    ("langchain_openai",      "langchain-openai",      "OpenAI-compatible LLM provider"),
    ("langchain_google_genai","langchain-google-genai","Google Gemini LLM provider"),
    ("stockstats",            "stockstats",            "Technical indicator calculations"),
    ("rank_bm25",             "rank-bm25",             "Financial memory retrieval"),
    ("apscheduler",           "apscheduler",           "Scheduled watchlist scans"),
    ("supabase",              "supabase",              "Supabase signal publishing"),
    ("yfinance",              "yfinance",              "Yahoo Finance data fetching"),
]


def check_dependencies() -> list[dict]:
    """Return a list of {package, pip_name, required_for, installed: bool}."""
    results = []
    for module_name, pip_name, required_for in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            results.append({
                "package":      module_name,
                "pip_name":     pip_name,
                "required_for": required_for,
                "installed":    True,
            })
        except ImportError:
            results.append({
                "package":      module_name,
                "pip_name":     pip_name,
                "required_for": required_for,
                "installed":    False,
            })
    return results


def get_missing() -> list[str]:
    """Return pip install names for all missing packages."""
    return [r["pip_name"] for r in check_dependencies() if not r["installed"]]


def log_missing_warnings() -> None:
    """Log a warning for each missing optional package. Called at startup."""
    missing = get_missing()
    if missing:
        logger.warning(
            "Missing optional packages — some features may not work: %s\n"
            "Fix with: pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
    else:
        logger.debug("All optional packages present.")
