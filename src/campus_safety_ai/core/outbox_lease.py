"""One local sender lease shared by synchronous and background delivery."""
from __future__ import annotations

import errno
import fcntl
from pathlib import Path
from typing import TextIO


class OutboxLease:
    """Hold a nonblocking filesystem lock until the sender has actually stopped.

    A background worker owns the lease for its entire lifetime, including an
    in-flight request after bounded shutdown returns. A synchronous sender owns
    one only while sending; producers can continue to enqueue without a lease.
    """

    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        self._handle: TextIO | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None and not self._handle.closed

    def acquire(self) -> OutboxLease:
        if self.held:
            raise RuntimeError("outbox sender lease is already held")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        handle = self.database.with_name(self.database.name + ".worker.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            handle.close()
            if isinstance(error, OSError) and error.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError("another delivery worker already owns this outbox") from None
            raise
        self._handle = handle
        return self

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()
