"""Point-in-time / no-leak mode for historical backtests (TRI-69).

When enabled, patches the vendor data layer AT OUR LAYER (the vendor tree
stays zero-mod) so that a run dated in the past cannot see data from after
its as-of date:

  1. ``get_fundamentals`` is NEUTRALIZED. The vendor implementation ignores
     ``curr_date`` and returns today's ``Ticker.info`` — 52-week hi/lo,
     50/200-day averages, TTM ratios — all future-relative for a past
     decision date (the TRI-69 pre-reg 🔴 leak). The replacement returns a
     pointer to the date-bounded statement tools instead.
  2. ``get_prediction_markets`` (Polymarket) is DISABLED — it returns live
     market odds with no reconstructable point-in-time path.
  3. ``get_insider_transactions`` is DISABLED — the vendor path has no date
     parameter at all and returns filings as of real-now (leak found in the
     TRI-69 date-bounding audit, beyond the two pre-registered ones).
  4. Live Reddit + StockTwits pre-fetches in the sentiment analyst are
     DISABLED (both fetch as of real-now with no date bound). The analyst's
     date-bounded news block is untouched.
  5. Statement tools (balance sheet / cashflow / income statement) get
     ``curr_date`` FORCED to the as-of date when the calling LLM omits it —
     the vendor filter is safe only when curr_date is actually passed.
  6. Every remaining vendor-routed data method gets a generic DATE CLAMP:
     any YYYY-MM-DD argument later than the as-of date is replaced with the
     as-of date (a hallucinated future end_date can't widen the window).
     ``build_verified_market_snapshot`` (which bypasses the vendor router)
     gets the same clamp.
  7. LEAK GUARD: every string returned by a vendor-routed method is scanned
     for raw ``Ticker.info`` future-relative field keys
     (fiftyTwoWeek*/fiftyDayAverage/twoHundredDayAverage/trailing*); the
     fundamentals category is additionally scanned for the formatted labels
     ("52 Week High:", "PE Ratio (TTM):", ...). A hit raises
     ``LookAheadLeakError`` — the run fails loudly rather than scoring a
     contaminated decision.

Activation: ``run_analysis`` calls ``enable_point_in_time_mode(trade_date)``
when the environment variable ``TRIFECTA_POINT_IN_TIME=1`` is set. TEST-ONLY
plumbing for the TRI-69 edge check; never on by default.
"""

import logging
import re
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# Raw yfinance Ticker.info keys that are future-relative for a past as-of
# date. These camelCase keys never occur in news prose, so they are safe to
# enforce on EVERY tool output.
FORBIDDEN_INFO_KEYS_RE = re.compile(
    r"fiftyTwoWeek\w*|fiftyDayAverage|twoHundredDayAverage"
    r"|trailingPE\b|trailingEps\b|trailingPegRatio\b|trailingAnnualDividend\w*"
)

# Formatted labels emitted by the vendor's get_fundamentals renderer. Only
# enforced on fundamentals-category outputs — historical news can
# legitimately say "hit a 52-week high" in prose.
FORBIDDEN_FUNDAMENTALS_LABELS_RE = re.compile(
    r"52 Week High:|52 Week Low:|50 Day Average:|200 Day Average:"
    r"|PE Ratio \(TTM\):|EPS \(TTM\):"
)

_DATE_ARG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Vendor-routed methods that are fully disabled in point-in-time mode.
_DISABLED_METHODS = {
    "get_prediction_markets": (
        "Prediction-markets data is disabled in point-in-time backtest mode: "
        "Polymarket only serves live (current) odds, which would leak "
        "post-decision-date information. Proceed without it."
    ),
    "get_insider_transactions": (
        "Insider-transaction data is disabled in point-in-time backtest mode: "
        "the vendor feed has no historical date filter and would leak filings "
        "made after the decision date. Proceed without it."
    ),
}

_NEUTRALIZED_FUNDAMENTALS_MSG = (
    "The fundamentals OVERVIEW is unavailable in point-in-time backtest mode "
    "(the live overview contains future-relative fields such as 52-week "
    "high/low and TTM ratios). Use get_balance_sheet, get_cashflow, and "
    "get_income_statement instead — those are date-bounded to the decision "
    "date and are the authoritative fundamentals for this analysis."
)

# Statement methods whose curr_date (3rd positional / kwarg) must never be
# None: the vendor date filter silently no-ops when it is.
_STATEMENT_METHODS = {"get_balance_sheet", "get_cashflow", "get_income_statement"}

_enabled_asof: str | None = None


class LookAheadLeakError(RuntimeError):
    """A tool output contained future-relative fields in point-in-time mode."""


def _clamp_date_args(args: tuple, kwargs: dict, asof: str) -> tuple:
    """Replace any YYYY-MM-DD argument later than *asof* with *asof*."""
    def clamp(v):
        if isinstance(v, str) and _DATE_ARG_RE.match(v) and v > asof:
            logger.warning("point_in_time: clamped date arg %s -> %s", v, asof)
            return asof
        return v

    return tuple(clamp(a) for a in args), {k: clamp(v) for k, v in kwargs.items()}


def _scan_output(method: str, result, fundamentals_category: bool):
    if not isinstance(result, str):
        return result
    m = FORBIDDEN_INFO_KEYS_RE.search(result)
    if m is None and fundamentals_category:
        m = FORBIDDEN_FUNDAMENTALS_LABELS_RE.search(result)
    if m is not None:
        raise LookAheadLeakError(
            f"point-in-time leak guard: '{method}' output contained "
            f"future-relative field {m.group(0)!r}"
        )
    return result


def _wrap_vendor_impl(method: str, impl: Callable, asof: str) -> Callable:
    fundamentals_category = method in _STATEMENT_METHODS or method == "get_fundamentals"

    def wrapped(*args, **kwargs):
        args, kwargs = _clamp_date_args(args, kwargs, asof)
        if method in _STATEMENT_METHODS:
            # Signature: (ticker, freq="quarterly", curr_date=None). Force a
            # missing/None curr_date to the as-of date so the vendor's
            # fiscal-period filter always engages.
            if "curr_date" in kwargs:
                if kwargs["curr_date"] is None:
                    kwargs["curr_date"] = asof
            elif len(args) >= 3:
                if args[2] is None:
                    args = args[:2] + (asof,) + args[3:]
            elif len(args) == 2:
                kwargs["curr_date"] = asof
            elif len(args) == 1:
                kwargs.setdefault("freq", "quarterly")
                kwargs["curr_date"] = asof
        return _scan_output(method, impl(*args, **kwargs), fundamentals_category)

    wrapped.__name__ = f"pit_{getattr(impl, '__name__', method)}"
    return wrapped


def _make_stub(message: str) -> Callable:
    def stub(*_args, **_kwargs):
        return message
    return stub


def enable_point_in_time_mode(asof_date: str) -> None:
    """Enable point-in-time mode with all data bounded to *asof_date*.

    Idempotent for the same date; raises if re-enabled with a different date
    within one process (a process serves exactly one decision date).
    """
    global _enabled_asof
    if not _DATE_ARG_RE.match(asof_date or ""):
        raise ValueError(f"asof_date must be YYYY-MM-DD, got {asof_date!r}")
    if _enabled_asof is not None:
        if _enabled_asof != asof_date:
            raise RuntimeError(
                f"point-in-time mode already enabled for {_enabled_asof}; "
                f"refusing to re-enable for {asof_date}"
            )
        return

    from tradingagents.dataflows import interface as vendor_interface

    vendor_methods: Dict[str, Dict] = vendor_interface.VENDOR_METHODS

    for method, vendors in vendor_methods.items():
        for vendor_name in list(vendors.keys()):
            if method == "get_fundamentals":
                vendors[vendor_name] = _make_stub(_NEUTRALIZED_FUNDAMENTALS_MSG)
            elif method in _DISABLED_METHODS:
                vendors[vendor_name] = _make_stub(_DISABLED_METHODS[method])
            else:
                vendor_impl = vendors[vendor_name]
                if isinstance(vendor_impl, list):
                    vendors[vendor_name] = [
                        _wrap_vendor_impl(method, f, asof_date) for f in vendor_impl
                    ]
                else:
                    vendors[vendor_name] = _wrap_vendor_impl(
                        method, vendor_impl, asof_date
                    )

    # Live social pre-fetches (sentiment analyst binds these at import time,
    # so patch BOTH the source dataflow modules and the analyst's namespace).
    social_stub_st = _make_stub(
        "StockTwits is disabled in point-in-time backtest mode (live feed, "
        "no historical filter). Base sentiment on the news block only."
    )
    social_stub_rd = _make_stub(
        "Reddit is disabled in point-in-time backtest mode (live feed, "
        "no historical filter). Base sentiment on the news block only."
    )
    from tradingagents.dataflows import reddit as _reddit
    from tradingagents.dataflows import stocktwits as _stocktwits
    _reddit.fetch_reddit_posts = social_stub_rd
    _stocktwits.fetch_stocktwits_messages = social_stub_st
    from tradingagents.agents.analysts import sentiment_analyst as _sa
    _sa.fetch_reddit_posts = social_stub_rd
    _sa.fetch_stocktwits_messages = social_stub_st

    # get_verified_market_snapshot bypasses route_to_vendor; clamp its
    # curr_date the same way (patch source module + tool module namespace).
    from tradingagents.dataflows import market_data_validator as _mdv
    from tradingagents.agents.utils import market_data_validation_tools as _mdvt

    real_snapshot = _mdv.build_verified_market_snapshot

    def clamped_snapshot(symbol, curr_date, look_back_days=30, *args, **kwargs):
        (symbol, curr_date, look_back_days), kwargs = _clamp_date_args(
            (symbol, curr_date, look_back_days), kwargs, asof_date
        )
        return real_snapshot(symbol, curr_date, look_back_days, *args, **kwargs)

    _mdv.build_verified_market_snapshot = clamped_snapshot
    _mdvt.build_verified_market_snapshot = clamped_snapshot

    _enabled_asof = asof_date
    logger.info("point-in-time mode ENABLED, as-of %s (fundamentals overview "
                "neutralized; polymarket/insider/reddit/stocktwits disabled; "
                "date args clamped; leak guard armed)", asof_date)


def is_enabled() -> bool:
    return _enabled_asof is not None


def enabled_asof() -> str | None:
    return _enabled_asof
