from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from campus_safety_ai.contracts import BehaviorObservation, EventRecord


@dataclass(frozen=True)
class FightPolicy:
    edge_id: str
    start_score: float = 0.75
    end_score: float = 0.35
    confirm_seconds: float = 2.0
    clear_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    config_version: str = "fight-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.end_score < self.start_score <= 1:
            raise ValueError("fight thresholds must satisfy 0 <= end_score < start_score <= 1")
        if min(self.confirm_seconds, self.clear_seconds, self.cooldown_seconds) < 0:
            raise ValueError("fight durations cannot be negative")


@dataclass
class _FightState:
    positive_since: datetime | None = None
    negative_since: datetime | None = None
    cooldown_until: datetime | None = None
    event_id: str | None = None
    started_at: datetime | None = None
    subject_track_keys: tuple[str, ...] = ()
    revision: int = 0


@dataclass(frozen=True)
class _FinalObservation:
    """Duck-typed observation standing in for a synthesized stream-end record."""

    camera_id: str
    source_epoch: int
    observed_at: datetime
    score: float = 0.0
    model_version: str = "stream-finalize"


class FightEventAnalysis:
    """Turns temporal fight-classifier scores into a stable event lifecycle."""

    def __init__(self, policy: FightPolicy) -> None:
        self.policy = policy
        self._states: dict[tuple[str, int], _FightState] = {}
        self._last_sequence: dict[tuple[str, int], int] = {}

    def advance(self, observation: BehaviorObservation) -> list[EventRecord]:
        if observation.behavior != "fighting":
            return []

        stream = (observation.camera_id, observation.source_epoch)
        previous = self._last_sequence.get(stream)
        if previous is not None and observation.sequence <= previous:
            return []
        self._last_sequence[stream] = observation.sequence
        state = self._states.setdefault(stream, _FightState())

        if observation.score >= self.policy.start_score:
            state.negative_since = None
            if state.event_id is not None:
                state.subject_track_keys = self._merge_subjects(
                    state.subject_track_keys, observation.subject_track_keys
                )
                return []
            if state.cooldown_until is not None and observation.observed_at < state.cooldown_until:
                state.positive_since = None
                return []
            state.positive_since = state.positive_since or observation.window_started_at
            confirmed_for = (observation.observed_at - state.positive_since).total_seconds()
            if confirmed_for >= self.policy.confirm_seconds:
                key = (
                    f"{observation.camera_id}:{observation.source_epoch}:"
                    f"fighting:{state.positive_since.isoformat()}"
                )
                state.event_id = str(uuid5(NAMESPACE_URL, key))
                state.started_at = state.positive_since
                state.subject_track_keys = observation.subject_track_keys
                state.revision = 1
                return [self._record(observation, state, "START")]
            return []

        if observation.score <= self.policy.end_score:
            state.positive_since = None
            if state.event_id is None:
                return []
            state.negative_since = state.negative_since or observation.window_started_at
            cleared_for = (observation.observed_at - state.negative_since).total_seconds()
            if cleared_for >= self.policy.clear_seconds:
                state.revision += 1
                record = self._record(observation, state, "END")
                state.cooldown_until = observation.observed_at + timedelta(
                    seconds=self.policy.cooldown_seconds
                )
                state.event_id = None
                state.started_at = None
                state.subject_track_keys = ()
                state.negative_since = None
                return [record]
            return []

        # Scores in the hysteresis band neither confirm nor clear an event.
        if state.event_id is None:
            state.positive_since = None
        else:
            state.negative_since = None
        return []

    def finalize(self, camera_id: str, source_epoch: int, ended_at: datetime) -> list[EventRecord]:
        """Close every open fight event for one source epoch when its stream ends.

        A stream restart or replay end must never leave an event stuck in the
        OPEN state; tracks cannot survive a new source epoch (see CONTEXT.md).
        """
        stream = (camera_id, source_epoch)
        state = self._states.get(stream)
        if state is None or state.event_id is None:
            return []
        state.revision += 1
        record = self._record(
            _FinalObservation(camera_id, source_epoch, ended_at), state, "END"
        )
        state.cooldown_until = ended_at + timedelta(seconds=self.policy.cooldown_seconds)
        state.event_id = None
        state.started_at = None
        state.subject_track_keys = ()
        state.negative_since = None
        return [record]

    @staticmethod
    def _merge_subjects(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*left, *right)))

    def _record(
        self,
        observation: BehaviorObservation,
        state: _FightState,
        phase: Literal["START", "END"],
    ) -> EventRecord:
        assert state.event_id is not None and state.started_at is not None
        return EventRecord(
            schema_version="1.0",
            event_id=state.event_id,
            revision=state.revision,
            phase=phase,
            event_type="person_fighting",
            severity="critical",
            edge_id=self.policy.edge_id,
            camera_id=observation.camera_id,
            started_at=state.started_at,
            observed_at=observation.observed_at,
            ended_at=observation.observed_at if phase == "END" else None,
            subject_track_keys=state.subject_track_keys,
            confidence=observation.score,
            model_version=observation.model_version,
            config_version=self.policy.config_version,
            idempotency_key=f"{state.event_id}:{state.revision}",
            status="CLOSED" if phase == "END" else "OPEN",
        )

