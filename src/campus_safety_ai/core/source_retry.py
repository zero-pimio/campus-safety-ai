"""Bounded reconnect timing, independent of any video transport."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReconnectPolicy:
    """Total retries per run; successful reads never reset this budget."""

    attempts: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self):
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("reconnect attempts must be a non-negative integer")
        for value in (self.initial_backoff_seconds, self.max_backoff_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("reconnect backoffs must be finite and non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum reconnect backoff must be at least the initial backoff")

    def delay(self, attempt: int) -> float:
        """Return the capped delay before a one-based retry number."""
        if type(attempt) is not int or not 1 <= attempt <= self.attempts:
            raise ValueError("retry number exceeds the configured reconnect budget")
        delay = self.initial_backoff_seconds
        # Early saturation also avoids overflowing for very large retry budgets.
        for _ in range(attempt - 1):
            if delay == 0 or delay >= self.max_backoff_seconds / 2:
                return 0.0 if delay == 0 else self.max_backoff_seconds
            delay *= 2
        return min(delay, self.max_backoff_seconds)
