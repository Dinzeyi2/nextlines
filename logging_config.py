import logging
import json
from typing import Any, Dict

_LOG_RECORD_DEFAULT_KEYS = {
    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
    'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
    'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
    'processName', 'process', 'message'
}


class JsonFormatter(logging.Formatter):
    """Logging formatter that outputs records as JSON."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_record: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _LOG_RECORD_DEFAULT_KEYS
        }
        if extras:
            log_record.update(extras)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging() -> None:
    """Configure root logger to emit JSON logs to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
