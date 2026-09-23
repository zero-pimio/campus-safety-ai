import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('torch')
pytest.importorskip('numpy')

import torch

from campus_safety_ai.adapters.runtimes.pose_fall import sha256
from campus_safety_ai.core.fall_analysis import FallObservation
from campus_safety_ai.core.fall_pipeline import WindowPolicy
from campus_safety_ai.training.fall_windows import ORIGIN


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'test_{name}_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trainer():
    return load_script('train_fall_windows')


def row(label=0, features=None, split='train'):
    return {'sample_id': f'{split}-sample', 'split': split, 'label': label, 'features': features}


def test_unknown_and_uncertain_windows_never_supply_a_missing_training_class(trainer):
    records = [row(0, [0.] * 32), row(1, None), row(None, [1.] * 32)]
    with pytest.raises(ValueError, match='both observed classes'):
        trainer.tensors(records)


def test_training_tensors_exclude_unknown_and_uncertain_without_relabeling(trainer):
    records = [row(0, [0.] * 32), row(None, [2.] * 32), row(1, None), row(1, [1.] * 32)]
    selected, inputs, labels = trainer.tensors(records)
    assert selected == [records[0], records[3]]
    assert inputs.shape == (2, 32)
    assert labels.tolist() == [0, 1]


def test_scoring_abstains_at_unknown_rows_without_calling_model_on_them(trainer):
    model = Mock(return_value=torch.tensor([[0., 0.], [0., 0.]]))
    records = [row(0, None), row(0, [0.] * 32), row(None, None), row(1, [1.] * 32)]
    assert trainer.scores_for(model, records) == [None, .5, None, .5]
    assert model.call_args.args[0].shape == (2, 32)
    model.reset_mock()
    assert trainer.scores_for(model, [row(0, None)]) == [None]
    model.assert_not_called()


def test_empty_classification_metrics_are_json_null_without_claiming_normal_predictions(trainer):
    records = [row(0, None), row(1, None), row(None, [0.] * 32)]
    metrics = trainer.metrics(records, [None, None, .1], .5)
    assert all(metrics[key] is None for key in ('loss', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'))
    assert metrics['labeled_known_windows'] == 0
    assert metrics['unknown_observations'] == 2
    assert all(value == 0 for value in metrics['confusion'].values())
    json.dumps(metrics, allow_nan=False)


def test_train_rejects_absent_validation_positive_before_optimizer_or_test_access(trainer, monkeypatch, tmp_path):
    manifest = {'samples': [{'sample_id': split, 'split': split} for split in
                            ('train', 'val', 'test', 'external_diagnostic')]}
    calls = []

    def collect(sample, *_):
        calls.append(sample['split'])
        split = sample['split']
        return [row(0, [0.] * 32, split), row(1, [1.] * 32 if split == 'train' else None, split)]

    optimizer = Mock(side_effect=AssertionError('fitting must not start without both validation classes'))
    monkeypatch.setattr(trainer, 'collect_windows', collect)
    monkeypatch.setattr(trainer.torch.optim, 'AdamW', optimizer)
    args = SimpleNamespace(output=tmp_path / 'candidate', cache=tmp_path / 'cache')
    with pytest.raises(ValueError, match='both observed classes'):
        trainer.train(args, manifest, WindowPolicy())
    assert calls == ['train', 'val']
    optimizer.assert_not_called()
    assert not (args.output / 'frozen.json').exists()


@pytest.fixture
def evaluation(trainer, monkeypatch, tmp_path):
    root, output, cache = (tmp_path / name for name in ('source', 'output', 'cache'))
    for directory in (root, output, cache):
        directory.mkdir()
    monkeypatch.setattr(trainer, 'ROOT', root)
    source_path = root / 'pipeline.py'
    source_path.write_text('# frozen inference source\n')
    baseline = tmp_path / 'baseline.pt'
    baseline.write_bytes(b'baseline fixture; model loading is injected')
    (output / 'best.pt').write_bytes(b'candidate fixture; model loading is injected')
    sample = {'sample_id': 'test-camera', 'split': 'test', 'subject_id': 'held-out-person',
              'timing': {'duration_s': 1.1, 'frame_timestamps_s': [0, .5, 1.]},
              'events': [{'start_s': .1, 'end_s': .9}]}
    manifest = {'samples': [sample]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    identity = {'pose_checkpoint_sha256': 'detector', 'settings': {'confidence': .5}}
    (cache / 'identity.json').write_text(json.dumps(identity))
    (cache / 'test-camera.json').write_text('{"fixture": "the collection seam is injected"}')
    window = WindowPolicy()
    protocol = {'window_policy': asdict(window), 'pose_cache_identity': {**identity, 'window_policy': asdict(window)},
                'source_sha256': {'pipeline.py': sha256(source_path)}, 'baseline_sha256': sha256(baseline)}
    trainer.write(output / 'protocol.json', protocol)
    frozen = {'manifest_sha256': sha256(manifest_path), 'baseline_sha256': sha256(baseline),
              'checkpoint_sha256': sha256(output / 'best.pt'), 'protocol_sha256': sha256(output / 'protocol.json'),
              'threshold': .5, 'event_policy': asdict(trainer.policy(.5))}
    trainer.write(output / 'frozen.json', frozen)
    args = SimpleNamespace(output=output, cache=cache, manifest=manifest_path, baseline=baseline)
    heads = Mock(side_effect=lambda path: SimpleNamespace(checkpoint_sha256=sha256(path), _model=Mock()))
    monkeypatch.setattr(trainer, 'FrozenPoseFallHead', heads)
    observation = FallObservation('test-camera', 1, 1, ORIGIN, ORIGIN, ORIGIN, None,
                                  'fixture', reason='missing_person')
    records = [{'sample_id': 'test-camera', 'split': 'test', 'label': 0, 'features': None,
                'observation': observation.to_dict()}]
    collect = Mock(return_value=records)
    monkeypatch.setattr(trainer, 'collect_windows', collect)
    return SimpleNamespace(args=args, manifest=manifest, window=window, source=source_path,
                           heads=heads, collect=collect)


@pytest.mark.parametrize('changed', ['window', 'cache_identity', 'baseline', 'source', 'manifest'])
def test_changed_frozen_dependencies_abort_before_test_collection_or_start_marker(trainer, evaluation, changed):
    fixture = evaluation
    if changed == 'window':
        fixture.window = replace(fixture.window, window_seconds=2)
    elif changed == 'cache_identity':
        (fixture.args.cache / 'identity.json').write_text('{"pose_checkpoint_sha256": "other", "settings": {}}')
    elif changed == 'baseline':
        fixture.args.baseline.write_bytes(b'another baseline')
    elif changed == 'source':
        fixture.source.write_text('# altered inference logic\n')
    else:
        fixture.args.manifest.write_text('{"samples": []}')
    with pytest.raises(ValueError, match='changed|differ'):
        trainer.evaluate(fixture.args, fixture.manifest, fixture.window)
    fixture.collect.assert_not_called()
    fixture.heads.assert_not_called()
    assert not (fixture.args.output / 'test-evaluation-started.json').exists()


def test_failed_test_collection_leaves_a_durable_marker_that_blocks_silent_reruns(trainer, evaluation):
    fixture = evaluation

    def fail(*args):
        marker = fixture.args.output / 'test-evaluation-started.json'
        assert marker.exists(), 'the irreversible test access must be recorded before collection'
        assert json.loads(marker.read_text())['cache_sha256']['test-camera'] == sha256(
            fixture.args.cache / 'test-camera.json')
        raise RuntimeError('injected failure after entering the test phase')

    fixture.collect.side_effect = fail
    with pytest.raises(RuntimeError, match='injected failure'):
        trainer.evaluate(fixture.args, fixture.manifest, fixture.window)
    assert not (fixture.args.output / 'test.json').exists()
    fixture.collect.reset_mock()
    fixture.heads.reset_mock()
    with pytest.raises(ValueError, match='already exists'):
        trainer.evaluate(fixture.args, fixture.manifest, fixture.window)
    fixture.collect.assert_not_called()
    fixture.heads.assert_not_called()


def test_existing_test_report_is_not_overwritten_even_if_start_marker_was_removed(trainer, evaluation):
    fixture = evaluation
    path = fixture.args.output / 'test.json'
    original = '{"prior": "immutable result"}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match='already exists'):
        trainer.evaluate(fixture.args, fixture.manifest, fixture.window)
    assert path.read_text() == original
    fixture.collect.assert_not_called()


def test_evaluation_with_all_unknown_observations_still_writes_misses_and_null_classification(trainer, evaluation):
    fixture = evaluation
    trainer.evaluate(fixture.args, fixture.manifest, fixture.window)
    result = json.loads((fixture.args.output / 'test.json').read_text())
    assert result['baseline_sha256'] == sha256(fixture.args.baseline)
    assert result['physical_camera_independence'] == 'unverified'
    for name in ('candidate', 'baseline'):
        assert result[name]['window_metrics']['accuracy'] is None
        assert result[name]['event_metrics']['strict']['metrics']['false_negatives'] == 1
        assert result[name]['event_metrics']['strict']['coverage']['known_seconds'] == 0
    fixture.collect.assert_called_once()


@pytest.fixture
def extraction(monkeypatch, tmp_path):
    extractor = load_script('extract_fall_web_pose')
    root, output = tmp_path / 'source', tmp_path / 'cache'
    root.mkdir()
    output.mkdir()
    monkeypatch.setattr(extractor, 'ROOT', root)
    detector = root / 'pose.pt'
    detector.write_bytes(b'pose checkpoint fixture')
    video = root / 'video.mp4'
    video.write_bytes(b'source fixture; decoder access must not happen in these cache tests')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'samples': [{'sample_id': 'clip', 'split': 'train', 'video_path': 'video.mp4',
                                               'sha256': sha256(video), 'timing': {'frame_timestamps_s': [0, .1]}}]}))
    settings = {'confidence': .5}
    monkeypatch.setattr(extractor, 'FrozenPoseFallHead', lambda _: SimpleNamespace(extraction_identity={'settings': settings}))
    monkeypatch.setattr(extractor, 'SinglePersonPose', lambda *args: SimpleNamespace())
    monkeypatch.setattr(extractor, 'OpenCvTimedFileFrames', Mock(side_effect=AssertionError('unexpected extraction')))
    identity = {'pose_checkpoint_sha256': sha256(detector), 'settings': settings,
                'runtime_source_sha256': sha256(Path(extractor.pose_fall.__file__)), 'device': 'cpu',
                'guard': 'exact SinglePersonPose.predict, every decoded frame, explicit source times'}
    identity_path = output / 'identity.json'
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    record = output / 'clip.json'
    record.write_text(json.dumps({'sample_id': 'clip', 'source_sha256': sha256(video),
                                  'identity_sha256': sha256(identity_path), 'timestamps_s': [0, .1],
                                  'frames': [{'arbitrary': 'damaged pose payload'}]}))
    prepared = output / 'prepared.json'
    prepared.write_text(json.dumps({'identity_sha256': sha256(identity_path), 'records': {'clip': sha256(record)}}))
    monkeypatch.setattr(sys, 'argv', ['extract_fall_web_pose.py', '--manifest', str(manifest), '--output', str(output),
                                    '--detector', str(detector), '--device', 'cpu', '--splits', 'train'])
    return SimpleNamespace(module=extractor, output=output, record=record, prepared=prepared)


def test_extraction_rejects_modified_registered_record_before_rebuilding_index(extraction):
    fixture = extraction
    original_index = fixture.prepared.read_bytes()
    body = json.loads(fixture.record.read_text())
    body['frames'][0]['arbitrary'] = 'modified after fingerprinting'
    fixture.record.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='modified'):
        fixture.module.main()
    assert fixture.prepared.read_bytes() == original_index


@pytest.mark.parametrize('missing', ['entire_index', 'record_entry'])
def test_extraction_does_not_silently_certify_existing_unindexed_pose_content(extraction, missing):
    fixture = extraction
    if missing == 'entire_index':
        fixture.prepared.unlink()
    else:
        prepared = json.loads(fixture.prepared.read_text())
        prepared['records'] = {}
        fixture.prepared.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='index|register|prepared|unverified|fingerprint'):
        fixture.module.main()


def test_extraction_seals_each_finished_record_before_a_later_source_failure(extraction, monkeypatch):
    fixture = extraction
    fixture.record.unlink()
    fixture.prepared.unlink()
    manifest_path = fixture.module.ROOT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['samples'].append({**manifest['samples'][0], 'sample_id': 'second'})
    manifest_path.write_text(json.dumps(manifest))

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter([SimpleNamespace(sequence=1), SimpleNamespace(sequence=2)])

    def predict(frame):
        return SimpleNamespace(sequence=frame.sequence, valid=False, reason='missing_person',
                               track_key=None, keypoints=[], box=[])

    monkeypatch.setattr(fixture.module, 'SinglePersonPose', lambda *args: SimpleNamespace(predict=predict))
    monkeypatch.setattr(fixture.module, 'OpenCvTimedFileFrames',
                        Mock(side_effect=[Source(), RuntimeError('second video failed')]))
    with pytest.raises(RuntimeError, match='second video failed'):
        fixture.module.main()
    prepared = json.loads(fixture.prepared.read_text())
    assert prepared['records'] == {'clip': sha256(fixture.record)}
    assert len(json.loads(fixture.record.read_text())['frames']) == 2
    assert not (fixture.output / 'second.json').exists()
    assert not list(fixture.output.glob('*.partial'))
