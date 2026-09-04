"""Request deadline propagated into bounded external calls; local work is cooperative."""
from contextvars import ContextVar
import time

deadline_context = ContextVar("processing_deadline", default=None)


def remaining(default=5.0):
    deadline = deadline_context.get()
    return default if deadline is None else max(0.0, deadline-time.monotonic())
