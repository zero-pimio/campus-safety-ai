import copy
import json
from datetime import timedelta

import pytest

pytest.importorskip('numpy')
pytest.importorskip('torch')

from campus_safety_ai.adapters.runtimes.pose_fall import sha256
from campus_safety_ai.core.fall_analysis import FallObservation, FallPolicy
from campus_safety_ai.core.fall_pipeline import WindowPolicy
from campus_safety_ai.training.fall_windows import ORIGIN, collect_windows, event_report, validate_manifest, window_label


def interval(start, end, label):
    return {'start_s': start, 'end_s': end, 'training_label': label}


@pytest.mark.parametrize('intervals, expected', [
    ([interval(0, .8, None), interval(.8, 1, 1)], 1),
    ([interval(0, .800002, None), interval(.800002, 1, 1)], None),
    ([interval(0, 1, 0)], 0),
    ([interval(0, .4, 0), interval(.4, .6, None), interval(.6, 1, 0)], None),
    ([interval(0, .5, 0), interval(.5, 1, 0)], 0),
    ([interval(0, .9, 0), interval(.9, 1, 1)], None),
    ([interval(1, 2, 1)], None),
    ([], None),
])
def test_window_labels_keep_motion_and_uncertain_boundaries_distinct(intervals, expected):
    assert window_label(0, 1, intervals) == expected


@pytest.mark.parametrize('start,end,minimum', [(1, 1, .2), (2, 1, .2), (float('nan'), 1, .2),
                                             (0, float('inf'), .2), (0, 1, 0), (0, 1, -.1),
                                             (0, 1, float('nan')), (False, 1, .2)])
def test_window_label_rejects_invalid_time_or_motion_threshold(start, end, minimum):
    with pytest.raises(ValueError):
        window_label(start, end, [interval(0, 1, 0)], minimum)


def sample(name='sample', split='train'):
    return {'sample_id': name, 'split': split, 'subject_id': f'person-{name}',
            'source_group': f'recording-{name}', 'sha256': (name[0] * 64),
            'video_path': f'videos/{name}.mp4', 'camera_id': None,
            'timing': {'duration_s': 2.1, 'frame_timestamps_s': [i / 10 for i in range(21)]},
            'intervals': [interval(0, 2.1, 0)], 'events': []}


@pytest.mark.parametrize('identity', ['subject_id', 'source_group', 'sha256'])
@pytest.mark.parametrize('split', ['val', 'test', 'external_diagnostic'])
def test_manifest_rejects_shared_identity_across_all_reported_splits(identity, split):
    first, second = sample('alpha'), sample('beta', split)
    second[identity] = first[identity]
    with pytest.raises(ValueError, match='leakage|cross|split'):
        validate_manifest({'samples': [first, second]})


def test_manifest_accepts_same_person_and_recording_within_one_split():
    first, second = sample('alpha'), sample('beta')
    second['subject_id'] = first['subject_id']
    second['source_group'] = first['source_group']
    validate_manifest({'samples': [first, second]})


@pytest.mark.parametrize('split', ['train', 'external_diagnostic'])
def test_manifest_rejects_overlapping_intervals_even_in_external_diagnostics(split):
    item = sample(split=split)
    item['intervals'] = [interval(0, 1, 0), interval(.5, 1.5, 1)]
    with pytest.raises(ValueError, match='interval|overlap'):
        validate_manifest({'samples': [item]})


@pytest.mark.parametrize('timing', [
    {'duration_s': float('inf'), 'frame_timestamps_s': [0, 1]},
    {'duration_s': 0, 'frame_timestamps_s': [0]},
    {'duration_s': 2.1, 'frame_timestamps_s': []},
    {'duration_s': 2.1, 'frame_timestamps_s': [.1, .2]},
    {'duration_s': 2.1, 'frame_timestamps_s': [0, .2, .1]},
    {'duration_s': 2.1, 'frame_timestamps_s': [0, 0]},
    {'duration_s': 2.1, 'frame_timestamps_s': [0, float('nan')]},
    {'duration_s': 2.1, 'frame_timestamps_s': [0, 3]},
])
def test_manifest_rejects_unusable_or_unbounded_timing(timing):
    item = sample()
    item['timing'] = timing
    with pytest.raises(ValueError):
        validate_manifest({'samples': [item]})


def write_cache(tmp_path):
    item = sample()
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'detector': 'fixture', 'settings': {}}))
    points = [[0, 0, 1] for _ in range(17)]
    for index, xy in [(5, (20, 20)), (6, (30, 20)), (11, (22, 50)), (12, (28, 50))]:
        points[index] = [*xy, 1]
    frames = [{'sequence': index + 1, 'valid': True, 'reason': 'associated',
               'track_key': 'sample:1:person', 'keypoints': copy.deepcopy(points), 'box': [10, 10, 40, 80]}
              for index in range(21)]
    cache = {'sample_id': item['sample_id'], 'source_sha256': item['sha256'],
             'identity_sha256': sha256(identity), 'timestamps_s': item['timing']['frame_timestamps_s'],
             'frames': frames}
    path = tmp_path / 'sample.json'
    path.write_text(json.dumps(cache))
    seal_cache(path)
    return item, cache, path


def seal_cache(path):
    (path.parent / 'prepared.json').write_text(json.dumps({
        'identity_sha256': sha256(path.parent / 'identity.json'), 'records': {'sample': sha256(path)},
    }))


def test_collect_windows_runs_real_causal_sampler_and_descriptor(tmp_path):
    item, _, _ = write_cache(tmp_path)
    rows = collect_windows(item, tmp_path, WindowPolicy(sample_count=4))
    assert rows[0]['features'] is None
    valid = [row for row in rows if row['features'] is not None]
    assert valid and valid[0]['seconds'] >= 1
    assert all(len(row['features']) == 32 and row['label'] == 0 for row in valid)
    assert all(row['split'] == 'train' for row in rows)


@pytest.mark.parametrize('mutation', ['source', 'identity', 'identity_file', 'times', 'count', 'sequence', 'sample'])
def test_collect_windows_rejects_inconsistent_cache_provenance(tmp_path, mutation):
    item, cache, path = write_cache(tmp_path)
    if mutation == 'source':
        cache['source_sha256'] = 'different'
    elif mutation == 'identity':
        cache['identity_sha256'] = 'different'
    elif mutation == 'identity_file':
        (tmp_path / 'identity.json').write_text('{"detector": "changed"}')
    elif mutation == 'times':
        cache['timestamps_s'] = [seconds + .01 for seconds in cache['timestamps_s']]
    elif mutation == 'count':
        cache['frames'].pop()
    elif mutation == 'sequence':
        cache['frames'][2]['sequence'] = 2
    else:
        cache['sample_id'] = 'different-recording'
    path.write_text(json.dumps(cache))
    # Keep the outer body fingerprint valid to exercise the inner provenance
    # checks independently, rather than rejecting every case at the first hash.
    seal_cache(path)
    with pytest.raises(ValueError):
        collect_windows(item, tmp_path, WindowPolicy(sample_count=4))


@pytest.mark.parametrize('mutation', ['points', 'box', 'valid', 'prepared_identity', 'prepared_record'])
def test_prepared_index_detects_changed_pose_body_or_missing_fingerprint(tmp_path, mutation):
    item, cache, path = write_cache(tmp_path)
    if mutation == 'points':
        cache['frames'][0]['keypoints'][5][0] += 10
    elif mutation == 'box':
        cache['frames'][0]['box'][2] += 10
    elif mutation == 'valid':
        cache['frames'][0]['valid'] = False
    else:
        prepared_path = tmp_path / 'prepared.json'
        prepared = json.loads(prepared_path.read_text())
        if mutation == 'prepared_identity':
            prepared['identity_sha256'] = 'different'
        else:
            prepared['records'] = {}
        prepared_path.write_text(json.dumps(prepared))
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match='fingerprint'):
        collect_windows(item, tmp_path, WindowPolicy(sample_count=4))


def test_invalid_pose_resets_descriptor_history_in_training_collection(tmp_path):
    item, cache, path = write_cache(tmp_path)
    cache['frames'][11].update(valid=False, reason='multiple_people', track_key=None, keypoints=[], box=[])
    path.write_text(json.dumps(cache))
    seal_cache(path)
    rows = collect_windows(item, tmp_path, WindowPolicy(sample_count=4))
    assert any(row['features'] is not None for row in rows if row['seconds'] < 1.1)
    assert all(row['features'] is None for row in rows if row['seconds'] >= 1.1)
    unknown = next(row for row in rows if row['seconds'] == 1.1)
    assert unknown['observation']['reason'] == 'multiple_people'


def observation_row(item, time, sequence):
    at = ORIGIN + timedelta(seconds=time)
    observation = FallObservation(item['sample_id'], 1, sequence, at, at, at, None,
                                  'fixture', reason='missing_person')
    return {'sample_id': item['sample_id'], 'features': None, 'observation': observation.to_dict()}


def test_event_report_keeps_unknown_and_unobserved_separate_from_known_normal_inference():
    item = sample()
    item['events'] = [{'start_s': .5, 'end_s': 1.5}]
    times = [0, .5, 1, 1.5, 2]
    rows = [observation_row(item, t, index + 1) for index, t in enumerate(times)]
    report = event_report([item], rows, [None] * len(rows), FallPolicy(edge_id='test'))
    for key in ('strict', 'tolerance_0_5s'):
        assert report[key]['metrics']['false_negatives'] == 1
        assert report[key]['metrics']['alarm_count'] == 0
        assert report[key]['coverage']['known_seconds'] == 0
        assert report[key]['coverage']['unknown_seconds'] == 2
        assert report[key]['coverage']['unobserved_seconds'] == pytest.approx(.1)
        assert report[key]['coverage']['complete'] is False


def test_event_report_does_not_fill_a_missing_sample_with_known_normal_time():
    first, missing = sample('alpha'), sample('beta', 'test')
    rows = [observation_row(first, t, index + 1) for index, t in enumerate([0, 1, 2])]
    report = event_report([first, missing], rows, [None] * len(rows), FallPolicy(edge_id='test'))
    coverage = report['strict']['coverage']
    second = next(session for session in coverage['sessions'] if session['camera_id'] == 'beta')
    assert second['observation_count'] == second['known_seconds'] == second['unknown_seconds'] == 0
    assert second['unobserved_seconds'] == 2.1


def test_event_report_rejects_score_count_mismatch():
    with pytest.raises(ValueError, match='scores'):
        event_report([sample()], [], [0.1], FallPolicy(edge_id='test'))


@pytest.mark.parametrize('changed', ['sample', 'camera', 'epoch'])
def test_event_report_rejects_rows_from_another_split_stream_instead_of_silently_discarding_them(changed):
    item = sample()
    record = observation_row(item, 0, 1)
    if changed == 'sample':
        record['sample_id'] = 'unlisted-test-clip'
    elif changed == 'camera':
        record['observation']['cameraId'] = 'unlisted-camera'
    else:
        record['observation']['sourceEpoch'] = 2
    with pytest.raises(ValueError, match='different sample|epoch'):
        event_report([item], [record], [None], FallPolicy(edge_id='test'))
