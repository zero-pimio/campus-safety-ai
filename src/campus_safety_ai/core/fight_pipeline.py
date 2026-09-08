from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol
from collections.abc import Iterable, Sequence

from campus_safety_ai.contracts import BehaviorObservation, EventRecord
from campus_safety_ai.core.fight_analysis import FightEventAnalysis
from campus_safety_ai.core.fight_inference import FightClassifier


@dataclass(frozen=True)
class VideoWindow:
    frames: tuple[Any, ...]
    first_frame: int
    last_frame: int


class VideoWindowSource(Protocol):
    fps: float
    sample_frequency: int

    def __iter__(self) -> Iterable[VideoWindow]: ...


class EvidenceSink(Protocol):
    def capture(
        self, observation: BehaviorObservation, rgb_frames: Sequence[Any], sampled_fps: float
    ) -> dict[str, str] | None: ...


@dataclass(frozen=True)
class FightPipelineResult:
    observations: tuple[dict[str, Any], ...]
    events: tuple[EventRecord, ...]
    fps: float
    sample_frequency: int


class FightVideoPipeline:
    """Deep module that owns window timing, event lifecycle, and evidence capture."""

    def __init__(
        self,
        classifier: FightClassifier,
        analysis: FightEventAnalysis,
        evidence: EvidenceSink,
    ) -> None:
        self.classifier = classifier
        self.analysis = analysis
        self.evidence = evidence

    def run(
        self,
        source: VideoWindowSource,
        camera_id: str,
        source_epoch: int,
        started_at: datetime,
    ) -> FightPipelineResult:
        observations: list[dict[str, Any]] = []
        events: list[EventRecord] = []
        last_observation: BehaviorObservation | None = None
        for sequence, window in enumerate(source, start=1):
            prediction = self.classifier.predict(window.frames)
            window_started_at = started_at + timedelta(seconds=window.first_frame / source.fps)
            window_ended_at = started_at + timedelta(seconds=window.last_frame / source.fps)
            observation = BehaviorObservation(
                camera_id=camera_id,
                source_epoch=source_epoch,
                sequence=sequence,
                observed_at=window_ended_at,
                window_started_at=window_started_at,
                window_ended_at=window_ended_at,
                behavior="fighting",
                score=prediction.fight_score,
                model_version=self.classifier.model_version,
            )
            artifact = None
            evidence_error = None
            try:
                artifact = self.evidence.capture(
                    observation, window.frames, source.fps / source.sample_frequency
                )
            except Exception as error:  # Evidence loss must not suppress the safety event.
                evidence_error = f"{type(error).__name__}: {error}"
            if artifact is not None:
                observation = replace(observation, evidence_uris=tuple(artifact.values()))
            value = observation.to_dict()
            value["sampledFrameRange"] = [window.first_frame, window.last_frame]
            value["logits"] = list(prediction.logits)
            if artifact is not None:
                value["evidence"] = artifact
            if evidence_error is not None:
                value["evidenceError"] = evidence_error
            observations.append(value)
            events.extend(self.analysis.advance(observation))
            last_observation = observation

        if last_observation is not None:
            events.extend(
                self.analysis.finalize(camera_id, source_epoch, last_observation.observed_at)
            )
        return FightPipelineResult(
            tuple(observations), tuple(events), source.fps, source.sample_frequency
        )
