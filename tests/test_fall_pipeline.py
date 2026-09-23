from datetime import datetime, timedelta, UTC
from dataclasses import asdict, replace

import pytest

from campus_safety_ai.core.fall_analysis import FallEventAnalysis, FallPolicy
from campus_safety_ai.core.fall_pipeline import FallPipeline, PoseFrame, WindowPolicy

START = datetime(2026, 1, 1, tzinfo=UTC)


class Head:
    model_version = "test-head"

    def __init__(self):
        self.windows = []

    def predict(self, frames):
        self.windows.append(frames)
        return 0.9, {}


def frame(t, sequence, track="person-1", valid=True):
    return PoseFrame("camera", 1, sequence, START + timedelta(seconds=t), (), (), track, valid,
                     "associated" if valid else "missing")


def pipeline(head=None):
    return FallPipeline(head or Head(), FallEventAnalysis(FallPolicy(edge_id="test")),
                        WindowPolicy(window_seconds=1, stride_seconds=.2, sample_count=4,
                                     max_frame_gap_seconds=.3, max_buffer_frames=16))


def test_fixed_causal_window_and_bounded_storage():
    head = Head()
    flow = pipeline(head)
    for i in range(201):
        batch = flow.advance(frame(i / 10, i + 1))
        assert flow.buffer_size <= 16
        if batch.observation and batch.observation.score is not None:
            assert (batch.observation.window_ended_at - batch.observation.window_started_at).total_seconds() == 1
    assert head.windows
    for samples in head.windows:
        assert len(samples) == 4
        assert (samples[-1].observed_at - samples[0].observed_at).total_seconds() <= 1
    assert head.windows[-1][0].observed_at > START + timedelta(seconds=18)


def test_hidden_bad_frame_resets_even_between_stride_ticks():
    head = Head()
    flow = pipeline(head)
    for i in range(13):
        flow.advance(frame(i / 10, i + 1))
    before = len(head.windows)
    bad = flow.advance(frame(1.25, 14, valid=False))
    assert bad.observation.score is None
    for i in range(13, 22):
        result = flow.advance(frame(i / 10, i + 2))
        if result.observation:
            assert result.observation.score is None
    assert len(head.windows) == before


def test_track_switch_gap_and_reordered_frames_do_not_bridge():
    flow = pipeline()
    for i in range(15):
        flow.advance(frame(i / 10, i + 1))
    assert flow.advance(frame(1.5, 16, track="person-2")).observation.score is None
    assert flow.advance(frame(4, 17, track="person-2")).observation.score is None
    with pytest.raises(ValueError):
        flow.advance(frame(3, 18))


def test_unknown_head_quality_is_preserved():
    class UnknownHead(Head):
        def predict(self, frames):
            return None, {"invalid_reasons": ["insufficient_core_joints"]}
    flow = pipeline(UnknownHead())
    batches = [flow.advance(frame(i / 10, i + 1)) for i in range(13)]
    assert any(b.observation and b.observation.reason == "invalid_descriptor" for b in batches)


def test_finalize_requires_a_new_source_epoch():
    from dataclasses import replace

    flow = pipeline()
    flow.advance(frame(0, 1))
    flow.finalize()
    with pytest.raises(ValueError, match="newer source_epoch"):
        flow.advance(frame(1, 2))
    assert flow.advance(replace(frame(1, 1), source_epoch=2)).observation.score is None


@pytest.mark.parametrize("kwargs", [{"window_seconds": float("nan")}, {"max_buffer_frames": 2},
                                    {"stride_seconds": 0}, {"sample_count": True}])
def test_bad_window_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        WindowPolicy(**kwargs)


@pytest.mark.parametrize('changed', [{'window_seconds': 2}, {'stride_seconds': .2}, {'sample_count': 8},
                                   {'max_frame_gap_seconds': .2}, {'max_buffer_frames': 300}])
def test_frozen_model_refuses_any_changed_window_extraction_policy(changed):
    policy = WindowPolicy()
    head = Head()
    head.extraction_identity = {'window_policy': asdict(policy)}
    with pytest.raises(ValueError, match='window'):
        FallPipeline(head, FallEventAnalysis(FallPolicy(edge_id='test')), replace(policy, **changed))


def test_frozen_model_accepts_its_exact_window_extraction_policy():
    policy = WindowPolicy()
    head = Head()
    head.extraction_identity = {'window_policy': asdict(policy)}
    flow = FallPipeline(head, FallEventAnalysis(FallPolicy(edge_id='test')), policy)
    assert flow.advance(frame(0, 1)).observation.reason == 'warming_up'
