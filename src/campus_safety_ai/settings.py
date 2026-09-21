from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from campus_safety_ai.core.event_analysis import IntrusionPolicy
from campus_safety_ai.core.fight_analysis import FightPolicy
from campus_safety_ai.core.parking_analysis import ParkingPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if value.get("schema_version") != "1.0":
        raise ValueError(f"unsupported or missing schema_version in {path}")
    return value


def _path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class FightModelSettings:
    backend: Literal["paddle", "torch", "onnx"]
    path: Path
    model_version: str
    device: str = "auto"


@dataclass(frozen=True)
class PlatformSettings:
    destination: Literal["jsonl", "easyaiot"]
    outbox: Path
    events: Path
    easyaiot_endpoint: str | None = None
    easyaiot_token_env: str | None = "EASYAIOT_TOKEN"
    easyaiot_timeout_seconds: float = 10.0
    easyaiot_device_name: str | None = None
    easyaiot_object: str = "person"
    easyaiot_task_type: Literal["realtime", "snap"] = "realtime"


@dataclass(frozen=True)
class VideoRuntimeSettings:
    video_adapter: Literal["opencv"]
    perception_adapter: str
    tracker_backend: Literal["simple_iou", "bytetrack"]
    precision: str
    device: str
    frame_count: int
    sample_frequency: int
    fight_event_config: Path
    fight_model_config: Path
    platform_config: Path
    evidence_enabled: bool
    evidence_dir: Path


def load_fight_policy(
    path: Path = PROJECT_ROOT / "configs/events/fight-v1.toml",
) -> FightPolicy:
    value = _read(path)
    return FightPolicy(
        edge_id=str(value.get("edge_id", "edge-dev-01")),
        start_score=float(value["start_score"]),
        end_score=float(value["end_score"]),
        confirm_seconds=float(value["confirm_seconds"]),
        clear_seconds=float(value["clear_seconds"]),
        cooldown_seconds=float(value["cooldown_seconds"]),
        config_version=str(value["config_version"]),
        behavior_label=str(value.get("behavior_label", "fighting")),
        event_type=str(value.get("event_type", "person_fighting")),
    )


def load_intrusion_policy(
    event_path: Path = PROJECT_ROOT / "configs/events/intrusion-v1.toml",
    scene_path: Path = PROJECT_ROOT / "configs/scenes/gate-02.toml",
) -> IntrusionPolicy:
    event = _read(event_path)
    scene = _read(scene_path)
    zone = scene["intrusion_zone"]
    target_labels = [str(label) for label in event.get("target_labels", ["person"])]
    return IntrusionPolicy(
        edge_id=str(event.get("edge_id", "edge-dev-01")),
        zone_id=str(zone["zone_id"]),
        polygon=tuple((float(point[0]), float(point[1])) for point in zone["polygon_normalized"]),
        enter_seconds=float(event["enter_seconds"]),
        exit_seconds=float(event["exit_seconds"]),
        cooldown_seconds=float(event["cooldown_seconds"]),
        minimum_confidence=float(event["minimum_confidence"]),
        config_version=str(event["config_version"]),
        event_type=str(event.get("event_type", "intrusion")),
        target_label=target_labels[0] if target_labels else "person",
        max_observation_gap_seconds=float(event.get("max_observation_gap_seconds", 2.0)),
    )


def load_parking_policy(event_path: Path, scene_path: Path) -> ParkingPolicy:
    event = _read(event_path)
    zone = _read(scene_path)["parking_zone"]
    return ParkingPolicy(
        edge_id=str(event.get("edge_id", "edge-dev-01")),
        zone_id=str(zone["zone_id"]),
        polygon=tuple((float(point[0]), float(point[1])) for point in zone["polygon_normalized"]),
        target_labels=tuple(str(label) for label in event.get(
            "target_labels", ["car", "truck", "bus", "motorcycle"]
        )),
        minimum_confidence=float(event.get("minimum_confidence", 0.4)),
        stationary_seconds=float(event.get("stationary_seconds", 30.0)),
        movement_threshold=float(event.get("movement_threshold", 0.02)),
        exit_seconds=float(event.get("exit_seconds", 2.0)),
        cooldown_seconds=float(event.get("cooldown_seconds", 30.0)),
        max_observation_gap_seconds=float(event.get("max_observation_gap_seconds", 2.0)),
        config_version=str(event["config_version"]),
    )


def load_fight_model_settings(
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> FightModelSettings:
    value = _read(path)
    backend = str(value["backend"])
    if backend not in {"paddle", "torch", "onnx"}:
        raise ValueError(f"unsupported fight model backend: {backend}")
    return FightModelSettings(
        backend=backend,  # type: ignore[arg-type]
        path=_path(str(value["path"]), project_root),
        model_version=str(value["model_version"]),
        device=str(value.get("device", "auto")),
    )


def load_platform_settings(
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> PlatformSettings:
    value = _read(path)
    destination = str(value["destination"])
    if destination not in {"jsonl", "easyaiot"}:
        raise ValueError(f"unsupported local destination: {destination}")
    easyaiot = value.get("easyaiot", {})
    if not isinstance(easyaiot, dict):
        raise ValueError("easyaiot platform settings must be a table")
    endpoint_value = easyaiot.get("endpoint")
    endpoint = str(endpoint_value).strip() if endpoint_value else None
    if destination == "easyaiot" and not endpoint:
        raise ValueError("EasyAIoT destination requires easyaiot.endpoint")
    timeout_seconds = float(easyaiot.get("timeout_seconds", 10.0))
    if timeout_seconds <= 0:
        raise ValueError("easyaiot.timeout_seconds must be positive")
    task_type = str(easyaiot.get("task_type", "realtime"))
    if task_type == "snapshot":
        task_type = "snap"
    if task_type not in {"realtime", "snap"}:
        raise ValueError(f"unsupported EasyAIoT task type: {task_type}")
    object_label = str(easyaiot.get("object", "person")).strip()
    if not object_label:
        raise ValueError("easyaiot.object must be non-empty")
    device_name_value = easyaiot.get("device_name")
    device_name = str(device_name_value).strip() if device_name_value else None
    token_env_value = easyaiot.get("token_env", "EASYAIOT_TOKEN")
    token_env = str(token_env_value).strip() if token_env_value else None
    return PlatformSettings(
        destination=destination,  # type: ignore[arg-type]
        outbox=_path(str(value["outbox"]), project_root),
        events=_path(str(value["events"]), project_root),
        easyaiot_endpoint=endpoint,
        easyaiot_token_env=token_env,
        easyaiot_timeout_seconds=timeout_seconds,
        easyaiot_device_name=device_name,
        easyaiot_object=object_label,
        easyaiot_task_type=task_type,  # type: ignore[arg-type]
    )


def load_video_runtime_settings(
    path: Path = PROJECT_ROOT / "configs/runtimes/pc-dev.toml",
    project_root: Path = PROJECT_ROOT,
) -> VideoRuntimeSettings:
    value = _read(path)
    video_adapter = str(value["video_adapter"])
    if video_adapter != "opencv":
        raise ValueError(f"unsupported video adapter: {video_adapter}")
    tracker_backend = str(value.get("tracker_backend", "simple_iou"))
    if tracker_backend not in {"simple_iou", "bytetrack"}:
        raise ValueError(f"unsupported tracker backend: {tracker_backend}")
    frame_count = int(value.get("frame_count", 8))
    sample_frequency = int(value.get("sample_frequency", 7))
    if frame_count <= 0 or sample_frequency <= 0:
        raise ValueError("frame_count and sample_frequency must be positive")
    return VideoRuntimeSettings(
        video_adapter="opencv",
        perception_adapter=str(value.get("perception_adapter", "ultralytics")),
        tracker_backend=tracker_backend,  # type: ignore[arg-type]
        precision=str(value.get("precision", "fp32")),
        device=str(value.get("device", "auto")),
        frame_count=frame_count,
        sample_frequency=sample_frequency,
        fight_event_config=_path(str(value["fight_event_config"]), project_root),
        fight_model_config=_path(str(value["fight_model_config"]), project_root),
        platform_config=_path(str(value["platform_config"]), project_root),
        evidence_enabled=bool(value.get("evidence_enabled", False)),
        evidence_dir=_path(str(value.get("evidence_dir", "runtime/evidence")), project_root),
    )
