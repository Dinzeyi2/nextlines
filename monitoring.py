from __future__ import annotations

from prometheus_client import Counter, Histogram
import sentry_sdk

# Prometheus metrics
REQUEST_COUNT = Counter("requests_total", "Total number of prediction requests")
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Time spent processing prediction requests"
)


def init_monitoring(sentry_dsn: str | None = None) -> None:
    """Initialize Sentry SDK if DSN provided."""
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn)
