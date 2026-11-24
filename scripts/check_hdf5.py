#!/usr/bin/env python3
"""Read-only RoboDojo HDF5 baseline audit. Never declares training readiness."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def serial(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def dump(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def strict_bool(value):
    value = serial(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def jpeg_buffer(value):
    return np.frombuffer(value.tobytes() if isinstance(value, np.ndarray) else bytes(value), np.uint8)


def quality(image):
    return {'spatial_std': float(np.std(image, axis=(0, 1)).mean()),
            'near_black_fraction': float(np.mean(np.max(image, axis=-1) < 3))}


def pose_metrics(pose):
    a = np.asarray(pose, np.float64)
    if a.ndim < 2 or a.shape[-1] != 7 or not a.size or not np.isfinite(a).all():
        return {'invalid_shape_or_values': True}
    q = a[..., 3:7]
    norms = np.linalg.norm(q, axis=-1)
    unit = q / np.maximum(norms[..., None], 1e-12)
    dots = np.sum(unit[1:] * unit[:-1], axis=-1)
    displacement = np.linalg.norm(np.diff(a[..., :3], axis=0), axis=-1)
    angles = 2 * np.arccos(np.clip(np.abs(dots), 0, 1))
    return {'quaternion_norm_min': float(norms.min()), 'quaternion_norm_max': float(norms.max()),
            'bad_norm_count': int((np.abs(norms - 1) > 1e-3).sum()),
            'zero_norm_count': int((norms < 1e-12).sum()), 'sign_flip_count': int((dots < 0).sum()),
            'sign_flip_rgb_or_control_rows': np.unique(np.nonzero(dots < 0)[0] + 1).tolist(),
            'max_translation_step_m': float(displacement.max()) if displacement.size else 0,
            'max_physical_rotation_step_rad': float(angles.max()) if angles.size else 0}


def continuous(q, anchor=None):
    q = np.asarray(q, np.float64).copy()
    norm = np.linalg.norm(q, axis=-1)
    if not np.isfinite(q).all() or np.any(np.abs(norm - 1) > 1e-3):
        raise ValueError('Cannot continuity-normalize invalid/non-unit quaternions')
    q /= norm[:, None]
    if anchor is not None and np.dot(q[0], anchor) < 0:
        q[0] *= -1
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1
    return q


def fk_check(joints, poses, side, cal):
    from scipy.spatial.transform import Rotation
    q = np.asarray(joints, np.float64)
    if q.ndim != 2 or q.shape[1] != len(cal['joint_axes']) or poses.shape != (len(q), 7):
        raise ValueError('FK shape / calibration mismatch')
    t = np.broadcast_to(np.asarray(cal['bases'][side]), (len(q), 4, 4)).copy()
    for i, (origin, axis) in enumerate(zip(cal['joint_origins'], cal['joint_axes'])):
        r = np.broadcast_to(np.eye(4), t.shape).copy()
        r[:, :3, :3] = Rotation.from_rotvec(q[:, i, None] * np.asarray(axis)).as_matrix()
        t = t @ np.asarray(origin) @ r
    truth = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    dr = Rotation.from_matrix(np.transpose(t[:, :3, :3], (0, 2, 1)) @ truth).magnitude()
    dp = np.linalg.norm(t[:, :3, 3] - poses[:, :3], axis=1)
    return {'rows': len(q), 'max_position_error_m': float(dp.max()),
            'max_orientation_error_rad': float(dr.max())}


def examine(job):
    entry, opts = job
    cv2.setNumThreads(1)
    p = Path(entry['path'])
    r = {'path': str(p), 'task': entry['task'], 'input_evidence': entry,
         'issues': [], 'numeric': {}, 'poses': {}, 'images': {}, 'fk': {},
         'image_flags': [], 'success': None, 'success_evidence': None,
         'physical_alignment_verified': False, 'simulation_replayed': False,
         'source_modified_by_checker': False, 'images_mode': opts['images']}

    def issue(code, detail, severity='error'):
        r['issues'].append({'code': code, 'detail': serial(detail), 'severity': severity})

    before = None
    try:
        before = p.stat()
        r.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns)
        with h5py.File(p, 'r') as f:
            attrs = {k: serial(v) for k, v in f.attrs.items()}
            r['attributes'] = attrs
            try:
                meta = json.loads(attrs.get('metadata', '{}'))
                if not isinstance(meta, dict):
                    raise ValueError('metadata must be a JSON object')
            except (ValueError, TypeError) as e:
                meta = {}
                issue('metadata_parse', str(e))
            r['metadata'] = meta
            r['image_offset_declared'] = attrs.get('image_offset', meta.get('image_offset'))
            r['already_aligned_marker'] = attrs.get('alignment_view_version')
            r['offset_applied_from_source'] = attrs.get('image_offset_from_source_rgb_frames')
            r['rgb_codec_declared'] = attrs.get('rgb_codec', meta.get('rgb_codec'))
            r['action_semantics_declared'] = attrs.get('action_semantics', meta.get('action_semantics'))
            r['state_semantics_declared'] = attrs.get('state_semantics', meta.get('state_semantics'))
            r['instruction'] = serial(f['instruction'][()]) if 'instruction' in f else attrs.get('instruction', meta.get('instruction'))
            for key in ('image_offset_declared', 'rgb_codec_declared', 'action_semantics_declared'):
                if r[key] is None:
                    issue('missing_declaration', key, 'review')
            if not r['instruction']:
                issue('missing_instruction', 'Check task manifest / delivery contract', 'review')
            raw_success = attrs.get('success')
            if raw_success is None and 'success' in f and f['success'].shape == ():
                raw_success = serial(f['success'][()])
            r['source_success_label_present'] = raw_success is not None
            r['source_success_label'] = raw_success
            if raw_success is not None:
                r['success'] = strict_bool(raw_success)
                r['success_evidence'] = 'HDF5 success scalar/attribute'
                if r['success'] is None:
                    issue('unrecognized_success_value', raw_success, 'review')
            if 'success' in entry:
                if r['success'] is not None and r['success'] != entry['success']:
                    issue('success_evidence_conflict', {'hdf5': r['success'], 'input': entry['success']}, 'review')
                else:
                    r['success'] = entry['success']
                    r['success_evidence'] = entry['success_evidence']
            if r['success'] is None:
                issue('success_unconfirmed', 'Consult supplied reports or existing user confirmation', 'review')
            elif r['success'] is False:
                issue('recorded_unsuccessful', r['success_evidence'], 'review')

            image_paths = []
            for cam in CAMERAS:
                found = [name for name in (f'vision/{cam}/colors', f'rgb/{cam}') if name in f]
                if len(found) != 1:
                    issue('missing_or_ambiguous_camera', {'camera': cam, 'paths': found})
                if found:
                    image_paths.append(found[0])
            terminal_paths = [f'terminal_rgb/{cam}' for cam in CAMERAS if f'terminal_rgb/{cam}' in f]
            image_set = set(image_paths + terminal_paths)
            arrays = {}
            digest = hashlib.sha256()
            r['dataset_shapes'] = {}

            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                r['dataset_shapes'][name] = {'shape': list(obj.shape), 'dtype': str(obj.dtype)}
                if name in image_set or obj.dtype.kind not in 'biufc':
                    return
                a = obj[()]
                arrays[name] = np.asarray(a)
                valid = np.isfinite(a)
                finite = np.asarray(a)[valid]
                r['numeric'][name] = {'nonfinite': int(np.size(a) - valid.sum()),
                                      'min': serial(finite.min()) if finite.size and not np.iscomplexobj(a) else None,
                                      'max': serial(finite.max()) if finite.size and not np.iscomplexobj(a) else None}
                if not valid.all():
                    issue('nonfinite', {'dataset': name, 'count': int((~valid).sum())})
                digest.update(name.encode() + str(np.shape(a)).encode() + str(obj.dtype).encode() + np.ascontiguousarray(a).tobytes())
            f.visititems(visit)
            r['numeric_sha256'] = digest.hexdigest()
            if 'state/left_ee_poses' in arrays:
                r['schema'] = 'legacy_ee_or_joint25'
                n = len(arrays['state/left_ee_poses'])
                required = [f'{g}/{s}_{field}' for g in ('state', 'action') for s in ('left', 'right')
                            for field in ('arm_joint_states', 'ee_joint_states')]
                required += ['state/left_ee_poses', 'state/right_ee_poses']
                r['rgb_hz_declared'] = serial(arrays.get('additional_info/frequency'))
            elif 'observations/qpos' in arrays:
                r['schema'] = 'dense_joint_rgb_index'
                n = len(arrays['observations/qpos'])
                required = ['observations/qpos', 'actions/joint_position', 'rgb_step_index']
                dt, stride = meta.get('dt'), meta.get('rgb_stride')
                r['control_hz_declared'] = 1 / dt if isinstance(dt, (int, float)) and dt > 0 else None
                r['rgb_hz_declared'] = 1 / (dt * stride) if isinstance(dt, (int, float)) and isinstance(stride, (int, float)) and dt > 0 and stride > 0 else None
            elif 'observations/ee' in arrays:
                r['schema'] = 'native_joint_rgb_index'
                n = len(arrays['observations/ee'])
                required = ['observations/ee', 'observations/joints', 'rgb_step_index']
                required += [f'actions/{s}_{field}' for s in ('left', 'right') for field in ('arm_joint_state', 'ee_joint_state')]
                r['control_hz_declared'] = attrs.get('control_hz')
            else:
                r['schema'] = 'unsupported'
                n, required = None, []
                issue('unsupported_schema', 'Add an adapter; do not treat this episode as passed')
            r['control_rows'] = n
            for name in required:
                if name not in arrays:
                    issue('missing_field', name)
            for name, a in arrays.items():
                if n is not None and a.ndim and name.startswith(('state/', 'action/', 'observations/', 'next_observations/', 'actions/', 'measured/')) and len(a) != n:
                    issue('numeric_length', {'dataset': name, 'actual': len(a), 'control_rows': n})
                if (name.endswith('_ee_poses') or name.endswith('/ee') or name.endswith('/ee_pose')):
                    metrics = pose_metrics(a)
                    r['poses'][name] = metrics
                    if metrics.get('invalid_shape_or_values') or metrics.get('bad_norm_count'):
                        issue('pose_invalid', {'dataset': name, **metrics})
                    elif metrics.get('sign_flip_count'):
                        issue('quaternion_sign_discontinuity', {'dataset': name, **metrics}, 'review')
                if name in ('time_s', 'timestamp', 'step', 'frame_index') and a.ndim == 1 and len(a) > 1:
                    if not np.all(np.diff(a.astype(np.float64)) > 0):
                        issue('nonincreasing_time_or_step', name)
                if name.startswith('observations/') and 'next_' + name in arrays:
                    b = arrays['next_' + name]
                    if a.shape == b.shape and len(a) > 1 and np.isfinite(a).all() and np.isfinite(b).all():
                        err = float(np.max(np.abs(a[1:].astype(float) - b[:-1].astype(float))))
                        r.setdefault('next_observation_max_error', {})[name] = err
                        if err > 1e-5:
                            issue('next_observation_discontinuity', {'dataset': name, 'max_error': err}, 'review')
            counts = [len(f[name]) for name in image_paths]
            r['rgb_frames'] = counts[0] if counts else 0
            if len(set(counts)) > 1:
                issue('camera_length_mismatch', dict(zip(image_paths, counts)))
            if 'rgb_step_index' in arrays:
                idx = arrays['rgb_step_index']
                valid_idx = (idx.ndim == 1 and len(idx) == r['rgb_frames'] and np.isfinite(idx).all()
                             and np.equal(idx, np.floor(idx)).all() and (idx >= 0).all()
                             and n is not None and (idx < n).all() and (np.diff(idx) > 0).all())
                if not valid_idx:
                    issue('invalid_rgb_step_index', {'shape': list(idx.shape), 'control_rows': n, 'rgb_frames': r['rgb_frames']})
                else:
                    r['rgb_index_summary'] = {'first': int(idx[0]) if len(idx) else None,
                                               'last': int(idx[-1]) if len(idx) else None,
                                               'strides': dict(Counter(map(int, np.diff(idx))))}
            elif n is not None and n != r['rgb_frames']:
                issue('image_state_length_without_mapping', {'rgb': r['rgb_frames'], 'control': n})
            if not r['rgb_frames']:
                issue('empty_image_sequence', 'No normal RGB frames')

            if opts.get('kinematics'):
                cal = opts['kinematics']
                pairs = []
                for side in ('left', 'right'):
                    for group in ('state', 'action'):
                        qname, pname = f'{group}/{side}_arm_joint_states', f'{group}/{side}_ee_poses'
                        if qname in arrays and pname in arrays:
                            pairs.append((group + '/' + side, arrays[qname], arrays[pname], side))
                for qname, pname, stride in (('observations/joints', 'observations/ee', None),
                                            ('observations/qpos', 'observations/ee_pose', 8)):
                    if qname in arrays and pname in arrays and arrays[pname].shape == (n, 2, 7):
                        q = arrays[qname]
                        for arm, side in enumerate(('left', 'right')):
                            qa = q[:, arm * stride:arm * stride + 6] if stride and q.shape == (n, 16) else q[:, arm] if q.shape == (n, 2, 6) else None
                            if qa is not None:
                                pairs.append((pname + '/' + side, qa, arrays[pname][:, arm], side))
                for name, q, pose, side in pairs:
                    try:
                        metric = fk_check(q, pose, side, cal)
                        r['fk'][name] = metric
                        if metric['max_position_error_m'] > opts['fk_position_tolerance'] or metric['max_orientation_error_rad'] > opts['fk_rotation_tolerance']:
                            issue('fk_mismatch', {'dataset': name, **metric})
                    except Exception as e:
                        issue('fk_check_error', {'dataset': name, 'error': str(e)})
                if not pairs:
                    issue('fk_not_adapted', 'No unambiguous paired joint/EE representation; adapt the source format', 'review')

            ee = {}
            for group in ('state', 'action'):
                names = [f'{group}/{s}_ee_poses' for s in ('left', 'right')]
                grips = [f'{group}/{s}_ee_joint_states' for s in ('left', 'right')]
                if all(k in arrays for k in names + grips) and all(arrays[k].shape == (n, 7) for k in names) and all(arrays[k].shape in ((n,), (n, 1)) for k in grips):
                    ee[group] = np.concatenate([np.c_[arrays[a], arrays[b].reshape(n, 1)] for a, b in zip(names, grips)], axis=1)
                    r[group + '_ee16_sha256'] = array_hash(ee[group])
            if set(ee) == {'state', 'action'}:
                r['state_action_ee16_sha256'] = array_hash(np.c_[ee['state'], ee['action']])
                try:
                    fixed = {k: np.asarray(v, np.float64).copy() for k, v in ee.items()}
                    for k in (3, 11):
                        fixed['state'][:, k:k+4] = continuous(fixed['state'][:, k:k+4])
                        fixed['action'][:, k:k+4] = continuous(fixed['action'][:, k:k+4], fixed['state'][0, k:k+4])
                    r['continuous_state_action_ee16_sha256'] = array_hash(np.c_[fixed['state'], fixed['action']])
                    if opts.get('stats'):
                        r['normalization'] = {}
                        for group, a in fixed.items():
                            stats = opts['stats'][group]['default']
                            mu, sd = np.asarray(stats['global_mean']), np.asarray(stats['global_std'])
                            if mu.shape != (16,) or sd.shape != (16,) or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd < 0).any():
                                raise ValueError('Expected finite EE16 mean/std arrays')
                            z = (a - mu) / (sd + 1e-8)
                            r['normalization'][group] = {'max_abs_z': float(np.abs(z).max()), 'elements_over5': int((np.abs(z) > 5).sum()), 'data_clipped': False}
                except (ValueError, KeyError, IndexError) as e:
                    issue('continuous_or_stats_not_completed', str(e), 'review')
            elif opts.get('stats'):
                issue('ee16_stats_not_adapted', 'Requires semantically verified joint/gripper projection first', 'review')

            visual = hashlib.sha256()
            for name in image_paths + terminal_paths:
                ds = f[name]
                if ds.ndim != 1:
                    issue('unsupported_image_storage', {'dataset': name, 'shape': list(ds.shape)})
                    continue
                count = len(ds)
                chosen = set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist())
                indices = range(count) if opts['images'] == 'all' else sorted(chosen) if opts['images'] == 'sample' else []
                info = {'frames': count, 'decoded_reduced': 0, 'decoded_full': 0, 'adjacent_identical': 0, 'terminal': name in terminal_paths}
                shapes = Counter()
                h, previous = hashlib.sha256(), None
                for i in indices:
                    buf = jpeg_buffer(ds[i])
                    raw = buf.tobytes()
                    h.update(len(raw).to_bytes(8, 'little') + raw)
                    if opts['images'] == 'all' and previous == raw:
                        info['adjacent_identical'] += 1
                    previous = raw
                    im = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_8) if len(buf) else None
                    if im is None:
                        issue('jpeg_decode_failed', {'dataset': name, 'rgb_frame': int(i)})
                        r['image_flags'].append({'dataset': name, 'rgb_frame': int(i), 'decode_failed': True})
                        continue
                    info['decoded_reduced'] += 1
                    metric = quality(im)
                    if i in chosen or metric['spatial_std'] < 1.2 or metric['near_black_fraction'] >= .99:
                        full = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if full is None:
                            issue('full_decode_failed', {'dataset': name, 'rgb_frame': int(i)})
                            continue
                        info['decoded_full'] += 1
                        shapes[str(full.shape)] += 1
                        metric = quality(full)
                        if metric['spatial_std'] < 1 or metric['near_black_fraction'] >= .995:
                            r['image_flags'].append({'dataset': name, 'rgb_frame': int(i), **metric,
                                                     'near_black': metric['near_black_fraction'] >= .995,
                                                     'near_constant': metric['spatial_std'] < 1})
                info['full_shapes'] = dict(shapes)
                if len(shapes) > 1:
                    issue('camera_size_changes', {'dataset': name, 'shapes': dict(shapes)}, 'review')
                info['encoded_stream_sha256'] = h.hexdigest() if opts['images'] == 'all' else None
                visual.update(name.encode() + h.digest())
                r['images'][name] = info
            if r['image_flags']:
                issue('image_candidates_require_review', len(r['image_flags']), 'review')
            r['visual_sha256'] = visual.hexdigest() if opts['images'] == 'all' and len(image_paths) == 3 else None
        r['hdf5_sha256'] = sha_file(p)
        if entry.get('expected_sha256') and r['hdf5_sha256'] != entry['expected_sha256'].lower():
            issue('source_manifest_sha_mismatch', entry['expected_sha256'])
    except Exception as e:
        issue('episode_audit_exception', f'{type(e).__name__}: {e}')
    if before is not None:
        try:
            after = p.stat()
            r['source_stat_unchanged'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            if not r['source_stat_unchanged']:
                issue('source_changed_during_check', str(p))
        except OSError as e:
            issue('source_disappeared', str(e))
    r['error_count'] = sum(x['severity'] == 'error' for x in r['issues'])
    r['review_count'] = sum(x['severity'] == 'review' for x in r['issues'])
    return serial(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='JSONL of explicit primary HDF5 paths and task names')
    parser.add_argument('--out', type=Path, required=True, help='New report directory outside raw data')
    parser.add_argument('--images', choices=('all', 'sample', 'none'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--kinematics-json', type=Path)
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--fk-position-tolerance', type=float, default=.0005)
    parser.add_argument('--fk-rotation-tolerance', type=float, default=.002)
    args = parser.parse_args()
    entries = [json.loads(s) for s in args.inputs.read_text().splitlines() if s.strip()]
    if not entries or args.workers < 1:
        parser.error('Need nonempty inputs and positive workers')
    for e in entries:
        if not e.get('path') or not e.get('task'):
            parser.error('Each entry requires path and task')
        e['path'] = str(Path(e['path']).resolve())
        if Path(e['path']).suffix.lower() not in ('.hdf5', '.h5'):
            parser.error('This checker only accepts HDF5; use the Parquet/video adapter for other formats')
        if 'success' in e and (type(e['success']) is not bool or not e.get('success_evidence')):
            parser.error('Input success requires a JSON boolean and success_evidence')
        if args.out.resolve().is_relative_to(Path(e['path']).parent):
            parser.error('Write reports outside the source episode directory')
    if len({e['path'] for e in entries}) != len(entries):
        parser.error('Duplicate input paths; first establish the primary inventory')
    opts = {'images': args.images, 'kinematics': json.loads(args.kinematics_json.read_text()) if args.kinematics_json else None,
            'stats': json.loads(args.stats.read_text()) if args.stats else None,
            'fk_position_tolerance': args.fk_position_tolerance, 'fk_rotation_tolerance': args.fk_rotation_tolerance}
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'inputs.json', entries)
    dump(args.out / 'check_config.json', {'options': opts, 'inputs_sha256': sha_file(args.inputs),
                                         'kinematics_sha256': sha_file(args.kinematics_json) if args.kinematics_json else None,
                                         'stats_sha256': sha_file(args.stats) if args.stats else None})
    records = []
    with (args.out / 'episodes.jsonl').open('x') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(examine, ((e, opts) for e in entries), chunksize=1):
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            progress = {'completed': len(records), 'total': len(entries), 'updated_at': datetime.now().astimezone().isoformat(),
                        'error_episodes': sum(bool(r['error_count']) for r in records)}
            dump(args.out / 'progress.json', progress)
            if len(records) % 25 == 0 or len(records) == len(entries):
                print(json.dumps(progress), flush=True)
    duplicate = {}
    for key in ('hdf5_sha256', 'numeric_sha256', 'visual_sha256', 'action_ee16_sha256', 'state_action_ee16_sha256', 'continuous_state_action_ee16_sha256'):
        groups = defaultdict(list)
        for r in records:
            if r.get(key):
                groups[r[key]].append(r['path'])
        duplicate[key] = [{'sha256': k, 'paths': v} for k, v in groups.items() if len(v) > 1]
    dump(args.out / 'duplicate_groups.json', duplicate)
    dump(args.out / 'image_flags.json', [{'path': r['path'], 'task': r['task'], 'flags': r['image_flags']} for r in records if r['image_flags']])
    fields = ['task', 'episodes', 'bytes', 'rgb_frames', 'control_rows', 'error_episodes', 'review_episodes', 'success_true', 'success_false', 'success_unknown', 'image_flag_episodes', 'decoded_reduced', 'decoded_full', 'fk_checked_episodes', 'stats_checked_episodes']
    with (args.out / 'task_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in sorted({r['task'] for r in records}):
            rows = [r for r in records if r['task'] == task]
            item = {'task': task, 'episodes': len(rows)}
            for key in ('bytes', 'rgb_frames', 'control_rows'):
                item[key] = sum(r.get(key) or 0 for r in rows)
            for key, predicate in [('error_episodes', lambda r: r['error_count']), ('review_episodes', lambda r: r['review_count']),
                                    ('success_true', lambda r: r['success'] is True), ('success_false', lambda r: r['success'] is False),
                                    ('success_unknown', lambda r: r['success'] is None), ('image_flag_episodes', lambda r: r['image_flags']),
                                    ('fk_checked_episodes', lambda r: r['fk']), ('stats_checked_episodes', lambda r: r.get('normalization'))]:
                item[key] = sum(bool(predicate(r)) for r in rows)
            for key in ('decoded_reduced', 'decoded_full'):
                item[key] = sum(cam[key] for r in rows for cam in r['images'].values())
            writer.writerow(item)
    report = {'status': 'baseline_scan_completed', **progress, 'images_mode': args.images,
              'review_episodes': sum(bool(r['review_count']) for r in records), 'training_ready': False,
              'limitations': ['No simulator replay or independent physical alignment verification.',
                              'Success attachments, task aliases, layout overlap and primary inventory need contract review.',
                              'Full-resolution decoding is sampled plus flagged frames; see per-camera counts.',
                              'FK/statistics only cover explicitly supplied calibration and adapted representations.',
                              'Duplicate fingerprints cover this input list only; no automatic exclusion or conversion.']}
    dump(args.out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 2 if report['error_episodes'] else 0


if __name__ == '__main__':
    sys.exit(main())
