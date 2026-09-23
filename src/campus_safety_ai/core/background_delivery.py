"""One local producer and one independently connected, retrying consumer."""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.event_delivery import Destination, EventDelivery
from campus_safety_ai.core.outbox_lease import OutboxLease

logger = logging.getLogger(__name__)


class BackgroundDelivery:
    """Persist before returning from submit. Network work stays on a worker.

    A filesystem lock permits one sender per outbox on this host.
    Delivery remains at least once; receivers must implement deduplication.
    """

    def __init__(self, database: Path, destination: Destination, *, retry_base: float = 1,
                 retry_max: float = 60, poll_seconds: float = 0.1,
                 shutdown_seconds: float = 3, max_pending_records: int | None = None,
                 max_pending_bytes: int | None = None) -> None:
        for value in (retry_base, retry_max, poll_seconds, shutdown_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("delivery timing values must be finite and positive")
        if retry_max < retry_base:
            raise ValueError("retry_max must be at least retry_base")
        database = database.resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        self._lease = OutboxLease(database).acquire()
        try:
            self._producer = EventDelivery(database, destination, max_pending_records=max_pending_records,
                                           max_pending_bytes=max_pending_bytes)
        except BaseException:
            self._lease.close()
            raise
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._changed = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._shutdown_seconds = shutdown_seconds

        def work() -> None:
            consumer = None
            try:
                # SQLite's thread affinity is respected: construct and close here.
                consumer = EventDelivery(database, destination, _lease=self._lease)
                while not self._stop.is_set():
                    consumer.drain_due(retry_base=retry_base, retry_max=retry_max, should_stop=self._stop.is_set)
                    self._changed.set()
                    self._wake.wait(poll_seconds)
                    self._wake.clear()
            except BaseException as error:
                self._error = error
                self._changed.set()
            finally:
                try:
                    if consumer is not None:
                        consumer.close()
                finally:
                    self._lease.close()

        self._thread = threading.Thread(target=work, name="event-delivery", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            try:
                self._producer.close()
            finally:
                self._lease.close()
            raise

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("delivery is closed")
        if self._error is not None:
            raise RuntimeError("delivery worker failed; inspect local storage") from self._error

    def submit(self, records: list[EventRecord]) -> int:
        self._check()
        inserted = self._producer.enqueue(records)
        self._wake.set()
        return inserted

    def pending_count(self) -> int:
        self._check()
        return self._producer.pending_count()

    def retry_status(self) -> dict:
        self._check()
        return self._producer.retry_status()

    def close(self) -> None:
        if self._closed:
            return
        deadline = time.monotonic() + self._shutdown_seconds
        drain_deadline = deadline - min(0.1, self._shutdown_seconds / 2)
        try:
            # Give finite jobs a bounded opportunity to finish, never block on
            # an unavailable platform forever. Remaining events stay on disk.
            while self._error is None and self._producer.pending_count():
                remaining = drain_deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(min(remaining, 0.05))
                self._changed.clear()
            pending = self._producer.pending_count()
        finally:
            self._stop.set()
            self._wake.set()
            self._thread.join(max(0, deadline - time.monotonic()))
            self._producer.close()
            self._closed = True
        if pending:
            logger.warning("delivery stopped with %s pending events preserved in outbox", pending)
        if self._thread.is_alive():
            logger.warning("delivery request still in flight; outbox remains durable")
        if self._error is not None:
            raise RuntimeError("delivery worker failed; events remain in outbox") from self._error

    def __enter__(self) -> BackgroundDelivery:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            logger.exception("delivery shutdown failed while handling another error")
