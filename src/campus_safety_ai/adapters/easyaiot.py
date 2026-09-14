"""EasyAIoT event mapping and HTTP delivery.

The core pipeline deliberately knows nothing about EasyAIoT field names or
transport details.  This module is the external seam: it maps a canonical
``EventRecord`` to EasyAIoT's ``/alert/hook`` request and implements the
``Destination`` interface used by the persistent outbox.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from campus_safety_ai.contracts import EventRecord, iso_time
from campus_safety_ai.core.event_delivery import Destination, JsonlDestination
from campus_safety_ai.settings import PlatformSettings


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".ts", ".flv", ".avi"})


class EasyAIoTDeliveryError(RuntimeError):
    """Raised when EasyAIoT cannot accept an event.

    ``EventDelivery`` intentionally leaves the outbox row undelivered when
    this error escapes, so a later flush can retry it.
    """


class JsonPoster(Protocol):
    def __call__(
        self,
        endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> int: ...


def _easyaiot_time(value: datetime) -> str:
    """Format a timezone-aware timestamp accepted by EasyAIoT's alert model."""

    return value.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _path_suffix(value: str) -> str:
    parsed = urlparse(value)
    return parsed.path.lower().rsplit("/", 1)[-1]


def _split_evidence(evidence_uris: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Choose image and recording paths without changing canonical evidence."""

    image_path = next(
        (uri for uri in evidence_uris if _path_suffix(uri).endswith(tuple(IMAGE_SUFFIXES))),
        None,
    )
    record_path = next(
        (uri for uri in evidence_uris if _path_suffix(uri).endswith(tuple(VIDEO_SUFFIXES))),
        None,
    )

    # EasyAIoT's existing alert table has one image_path field.  If a custom
    # evidence sink returns an extension-less URI, keep a non-record URI
    # visible instead of silently dropping the evidence reference.  Do not
    # mislabel a known video as an image when no snapshot exists.
    if image_path is None and evidence_uris:
        image_path = next((uri for uri in evidence_uris if uri != record_path), None)
    if record_path is None:
        record_path = next((uri for uri in evidence_uris if uri != image_path), None)
    return image_path, record_path


class EasyAIoTEventMapper:
    """Map the project event contract to EasyAIoT's alert-hook payload."""

    def __init__(
        self,
        *,
        device_name: str | None = None,
        object_label: str = "person",
        task_type: str = "realtime",
    ) -> None:
        if object_label.strip() == "":
            raise ValueError("EasyAIoT object label must be non-empty")
        normalized_task_type = "snap" if task_type == "snapshot" else task_type
        if normalized_task_type not in {"realtime", "snap"}:
            raise ValueError(f"unsupported EasyAIoT task type: {task_type}")
        self.device_name = device_name.strip() if device_name and device_name.strip() else None
        self.object_label = object_label.strip()
        self.task_type = normalized_task_type

    def to_payload(self, value: EventRecord | Mapping[str, Any]) -> dict[str, Any]:
        record = value if isinstance(value, EventRecord) else EventRecord.from_dict(dict(value))
        canonical = record.to_dict()
        image_path, record_path = _split_evidence(record.evidence_uris)
        information = {
            "source": "campus-safety-ai",
            "schemaVersion": record.schema_version,
            "eventId": record.event_id,
            "revision": record.revision,
            "phase": record.phase,
            "eventType": record.event_type,
            "severity": record.severity,
            "edgeId": record.edge_id,
            "cameraId": record.camera_id,
            "startedAt": iso_time(record.started_at),
            "observedAt": iso_time(record.observed_at),
            "endedAt": iso_time(record.ended_at) if record.ended_at else None,
            "subjectTrackKeys": list(record.subject_track_keys),
            "confidence": record.confidence,
            "modelVersion": record.model_version,
            "configVersion": record.config_version,
            "idempotencyKey": record.idempotency_key,
            "status": record.status,
            "evidenceUris": list(record.evidence_uris),
            # Keep the exact canonical object for consumers that need a
            # lossless replay rather than relying on the flattened fields.
            "eventRecord": canonical,
        }
        return {
            "object": self.object_label,
            "event": record.event_type,
            "device_id": record.camera_id,
            "device_name": self.device_name or record.camera_id,
            "region": None,
            "information": information,
            "time": _easyaiot_time(record.observed_at),
            "image_path": image_path,
            "record_path": record_path,
            "task_type": self.task_type,
            # EasyAIoT calls this correlation_id.  The per-revision value
            # preserves the project's idempotency semantics on its side.
            "correlation_id": record.idempotency_key,
        }


def _post_json(
    endpoint: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> int:
    request = Request(endpoint, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read()
            return int(response.status)
    except HTTPError as exc:
        raise EasyAIoTDeliveryError(f"EasyAIoT endpoint returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise EasyAIoTDeliveryError("EasyAIoT endpoint is unavailable") from exc


class EasyAIoTDestination:
    """Reliable-outbox destination for EasyAIoT's ``/alert/hook`` endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 10.0,
        mapper: EasyAIoTEventMapper | None = None,
        poster: JsonPoster | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("EasyAIoT endpoint must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("EasyAIoT endpoint must not contain URL credentials")
        if timeout_seconds <= 0:
            raise ValueError("EasyAIoT timeout must be positive")
        self.endpoint = endpoint
        self.token = token.strip() if token and token.strip() else None
        self.timeout_seconds = timeout_seconds
        self.mapper = mapper or EasyAIoTEventMapper()
        self.poster = poster or _post_json

    def publish(self, record: dict[str, Any]) -> None:
        payload = self.mapper.to_payload(record)
        canonical = record if isinstance(record, dict) else payload["information"]["eventRecord"]
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Event-Id": str(canonical["eventId"]),
            "X-Event-Revision": str(canonical["revision"]),
            "X-Event-Phase": str(canonical["phase"]),
            "X-Idempotency-Key": str(canonical["idempotencyKey"]),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        status = self.poster(self.endpoint, body, headers, self.timeout_seconds)
        if status < 200 or status >= 300:
            raise EasyAIoTDeliveryError(f"EasyAIoT endpoint returned HTTP {status}")


def build_platform_destination(
    settings: PlatformSettings,
    *,
    events_path: Path | None = None,
    token: str | None = None,
) -> Destination:
    """Build the configured destination while keeping secrets out of TOML."""

    if settings.destination == "jsonl":
        return JsonlDestination(events_path or settings.events)
    if settings.destination != "easyaiot":
        raise ValueError(f"unsupported platform destination: {settings.destination}")
    if not settings.easyaiot_endpoint:
        raise ValueError("EasyAIoT destination requires an endpoint")
    resolved_token = token
    if resolved_token is None and settings.easyaiot_token_env:
        resolved_token = os.environ.get(settings.easyaiot_token_env)
    return EasyAIoTDestination(
        settings.easyaiot_endpoint,
        token=resolved_token,
        timeout_seconds=settings.easyaiot_timeout_seconds,
        mapper=EasyAIoTEventMapper(
            device_name=settings.easyaiot_device_name,
            object_label=settings.easyaiot_object,
            task_type=settings.easyaiot_task_type,
        ),
    )
