import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('numpy')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

import onnx
import torch

from campus_safety_ai.adapters.runtimes.pose_fall import sha256
from campus_safety_ai.core.fall_analysis import FallPolicy
from campus_safety_ai.core.fall_pipeline import WindowPolicy
from campus_safety_ai.training import pose_motion


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[1] / 'scripts/export_fall_window_head.py'
    spec = importlib.util.spec_from_file_location('test_export_fall_head_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_head(tmp_path):
    window = asdict(WindowPolicy())
    protocol = {'window_policy': window, 'pose_cache_identity': {'pose_checkpoint_sha256': 'fixture-detector',
                                                               'settings': {}, 'window_policy': window},
                'cache_sha256': {'val-one': 'fixture-cache-hash', 'train-one': 'other-cache-hash'}}
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    torch.manual_seed(7)
    model = pose_motion.PoseMotionClassifier(len(pose_motion.FEATURE_NAMES), 0).eval()
    model.set_feature_normalization(torch.linspace(-.2, .2, 32), torch.linspace(.5, 1.5, 32))
    checkpoint = tmp_path / 'best.pt'
    torch.save({'model_name': 'pose-motion-classifier', 'model_state': model.state_dict(),
                'feature_dim': len(pose_motion.FEATURE_NAMES), 'feature_names': list(pose_motion.FEATURE_NAMES),
                'class_names': ['no_fall_transition', 'fall_transition'], 'hidden_dim': 0, 'threshold': .9,
                'descriptor_source_sha256': sha256(Path(pose_motion.__file__)),
                'descriptor_settings': {'min_conf': .25}, 'protocol_sha256': sha256(protocol_path),
                'pose_checkpoint_sha256': 'fixture-detector'}, checkpoint)
    (tmp_path / 'frozen.json').write_text(json.dumps({'checkpoint_sha256': sha256(checkpoint),
                                                    'protocol_sha256': sha256(protocol_path), 'threshold': .9,
                                                    'event_policy': asdict(FallPolicy(edge_id='test', start_score=.9))}))
    (tmp_path / 'development-windows.json').write_text(json.dumps({
        'window_policy': window,
        'rows': [{'sample_id': 'train-one', 'split': 'train', 'features': ['train must not enter parity']},
                 {'sample_id': 'val-one', 'split': 'val', 'features': [0.] * 32},
                 {'sample_id': 'val-one', 'split': 'val', 'features': [.1] * 32},
                 {'sample_id': 'val-one', 'split': 'val', 'features': None}],
    }))
    return checkpoint


def test_export_verifies_real_runtime_dynamic_batches_without_modifying_frozen_inputs(exporter, frozen_head):
    inputs = [frozen_head, *(frozen_head.parent / name for name in
                             ('protocol.json', 'frozen.json', 'development-windows.json'))]
    before = {path: sha256(path) for path in inputs}
    report = exporter.export(frozen_head)
    assert report['status'] == 'passed'
    assert report['validation_feature_count'] == 2 and report['raw_zero_control_count'] == 1
    assert report['pose_detector_sha256'] == 'fixture-detector'
    assert report['providers'] == ['CPUExecutionProvider']
    assert report['parity']['argmax_agreement'] and report['parity']['frozen_threshold_agreement']
    assert report['parity']['maximum_logit_absolute_error'] <= 1e-4
    assert [run['batch_size'] for run in report['parity']['comparison_runs']] == [1, 3]
    assert all(run['checked_features'] == 3 for run in report['parity']['comparison_runs'])
    graph_path = frozen_head.parent / 'onnx/model.onnx'
    graph = onnx.load(graph_path)
    assert graph.opset_import[0].version == 17
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param == 'batch'
    assert graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param == 'batch'
    assert report['onnx_sha256'] == sha256(graph_path)
    assert before == {path: sha256(path) for path in inputs}
    assert not (frozen_head.parent / 'test.json').exists()


def test_export_refuses_existing_output_even_if_empty(exporter, frozen_head):
    output = frozen_head.parent / 'onnx'
    output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        exporter.export(frozen_head)
    assert list(output.iterdir()) == []


def test_bad_checkpoint_fingerprint_is_rejected_before_export_output_is_created(exporter, frozen_head):
    frozen_head.write_bytes(frozen_head.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='frozen selection'):
        exporter.export(frozen_head)
    assert not (frozen_head.parent / 'onnx').exists()


@pytest.mark.parametrize('changed', ['test_row', 'window', 'unlisted_sample', 'nonfinite'])
def test_export_refuses_untrusted_or_non_development_parity_inputs(exporter, frozen_head, changed):
    path = frozen_head.parent / 'development-windows.json'
    document = json.loads(path.read_text())
    if changed == 'test_row':
        document['rows'][1]['split'] = 'test'
    elif changed == 'window':
        document['window_policy']['sample_count'] = 8
    elif changed == 'unlisted_sample':
        document['rows'][1]['sample_id'] = 'not-in-frozen-cache'
    else:
        document['rows'][1]['features'][0] = float('nan')
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        exporter.export(frozen_head)
    assert not (frozen_head.parent / 'onnx').exists()
