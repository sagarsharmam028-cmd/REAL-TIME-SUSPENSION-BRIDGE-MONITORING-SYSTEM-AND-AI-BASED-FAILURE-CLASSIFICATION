"""
alert_logger.py
===============
In-memory alert event logger for the Bridge Structural Health Monitoring System.

Design decisions:
  - Pure in-memory (no disk I/O) — log lives for the duration of a Streamlit session.
  - Thread-safe via threading.Lock so the background serial thread can call log_event()
    without racing the dashboard render loop.
  - Only records TRANSITIONS between states (SAFE→WARNING, WARNING→DANGER, etc.) to avoid
    flooding the log with thousands of identical SAFE entries every second.
  - Exposes to_dataframe() for direct use with st.dataframe() and st.download_button().
"""

import threading
from datetime import datetime
import pandas as pd

# ──────────────────────────────────────────────
# Internal state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_history: list[dict] = []          # List of event dicts
_last_logged_state: str | None = None  # Track previous state for transition detection

MAX_EVENTS = 500   # Cap to avoid unbounded memory growth in long sessions


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def log_event(
    state: str,
    confidence: float,
    weight: float,
    deflection: float,
    vibration: float,
    risk_score: float,
    features: dict | None = None,
    force: bool = False
) -> bool:
    """
    Record an alert event.

    Parameters
    ----------
    state        : 'SAFE', 'WARNING', or 'DANGER'
    confidence   : ML model confidence (0.0–1.0)
    weight       : Current load weight in kg
    deflection   : Max deck deflection in cm
    vibration    : Peak vibration magnitude in g
    risk_score   : Computed physics risk score (0.0–1.0)
    features     : Optional full feature dict from extract_features_live()
    force        : If True, log even when state hasn't changed (used for first sample)

    Returns
    -------
    bool : True if an event was recorded, False if it was a duplicate (no state change).
    """
    global _last_logged_state

    with _lock:
        # Only log on state transitions (or first sample, or forced)
        if state == _last_logged_state and not force:
            return False

        _last_logged_state = state

        event = {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "state":      state,
            "confidence": round(confidence, 4),
            "weight_kg":  round(weight, 3),
            "deflection_cm": round(deflection, 3),
            "vibration_g":   round(vibration, 4),
            "risk_score":    round(risk_score, 4),
        }

        # Embed key features if provided
        if features:
            event["deflection_rate"]  = round(features.get("deflection_rate", 0.0), 4)
            event["dominant_freq_hz"] = round(features.get("dominant_frequency", 0.0), 4)
            event["spectral_energy"]  = round(features.get("vibration_spectral_energy", 0.0), 2)

        _history.append(event)

        # Rolling cap — remove oldest events when limit exceeded
        if len(_history) > MAX_EVENTS:
            _history.pop(0)

        return True


def get_history() -> list[dict]:
    """Return a snapshot of all events (newest last)."""
    with _lock:
        return list(_history)


def get_history_reversed() -> list[dict]:
    """Return events newest-first (for dashboard display)."""
    with _lock:
        return list(reversed(_history))


def clear():
    """Wipe all logged events and reset the state tracker."""
    global _last_logged_state
    with _lock:
        _history.clear()
        _last_logged_state = None


def event_count() -> int:
    """Number of events currently stored."""
    with _lock:
        return len(_history)


def last_state() -> str | None:
    """Return the most recently logged state, or None if no events yet."""
    with _lock:
        return _last_logged_state


def to_dataframe() -> "pd.DataFrame":
    """
    Convert the in-memory event log to a pandas DataFrame.
    Returns an empty DataFrame with the correct column schema when no events exist.
    """
    columns = [
        "timestamp", "state", "confidence",
        "weight_kg", "deflection_cm", "vibration_g", "risk_score",
        "deflection_rate", "dominant_freq_hz", "spectral_energy"
    ]
    events = get_history_reversed()
    if not events:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(events)

    # Ensure all expected columns exist (older events may lack optional feature cols)
    for col in columns:
        if col not in df.columns:
            df[col] = None

    return df[columns]


def to_csv_bytes() -> bytes:
    """Serialise the event log to UTF-8 CSV bytes — ready for st.download_button()."""
    return to_dataframe().to_csv(index=False).encode("utf-8")
