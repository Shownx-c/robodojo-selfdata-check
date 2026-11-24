#!/usr/bin/env python3
"""Behavior checks against synthetic corruption and independent source snapshots."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
from PIL import Image

SKILL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_hdf5', SKILL / 'scripts/check_hdf5.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def calibration():
    bases = {}
    for side, x in [('left', -.3), ('right', .3)]:
        base = np.eye(4)
        base[0, 3] = x
        bases[side] = base.tolist()
    return {'bases': bases, 'joint_axes': [[0, 0, 1]] * 6, 'joint_origins': [np.eye(4).tolist()] * 6}


def images(f, prefix, count=5, black=False, corrupt=False, seed=1):
    rng = np.random.default_rng(seed)
    for cam in audit.CAMERAS:
        name = f'vision/{cam}/colors' if prefix == 'vision' else f'rgb/{cam}'
        ds = f.create_dataset(name, (count,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(count):
            im = rng.integers(10, 120, size=(48, 64, 3), dtype=np.uint8)
            im[:, :, 0] += 100
            if black and cam == 'cam_right_wrist' and i == 2:
                im[:] = 0
            encoded = cv2.imencode('.jpg', im)[1]
            if corrupt and cam == 'cam_left_wrist' and i == 3:
                encoded = np.frombuffer(b'not-a-jpeg', dtype=np.uint8)
            ds[i] = encoded


def legacy(p, flip=False, black=False, corrupt=False, seed=1):
    with h5py.File(p, 'w') as f:
        f.attrs.update(image_offset=0, rgb_codec='OpenCV RGB-direct JPEG95 4:2:0',
                       action_semantics='issued joint target FK', success='false')
        f.create_dataset('instruction', data='Put the block on the table')
        f.create_dataset('additional_info/frequency', data=25)
        for group in ('state', 'action'):
            for side, x in [('left', -.3), ('right', .3)]:
                pose = np.zeros((5, 7)); pose[:, 0] = x; pose[:, 3] = 1
                if flip:
                    pose[2, 3:] *= -1
                f.create_dataset(f'{group}/{side}_ee_poses', data=pose)
                f.create_dataset(f'{group}/{side}_arm_joint_states', data=np.zeros((5, 6)))
                f.create_dataset(f'{group}/{side}_ee_joint_states', data=np.full((5, 1), .5))
        images(f, 'vision', black=black, corrupt=corrupt, seed=seed)


def options():
    return {'images': 'all', 'kinematics': None, 'stats': None,
            'fk_position_tolerance': .0005, 'fk_rotation_tolerance': .002}


class Checkers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'; self.raw.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def scan(self, p, opts=None):
        before = hashlib.sha256(p.read_bytes()).hexdigest()
        row = audit.examine(({'path': str(p), 'task': 'test'}, opts or options()))
        self.assertEqual(before, hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertTrue(row['source_stat_unchanged'])
        return row

    def test_quaternion_fk_boolean_and_statistics(self):
        p = self.raw / 'good.hdf5'; legacy(p, flip=True)
        opts = options(); opts['kinematics'] = calibration()
        opts['stats'] = {g: {'default': {'global_mean': [0] * 16, 'global_std': [.1] * 16}} for g in ('state', 'action')}
        r = self.scan(p, opts)
        self.assertEqual(r['error_count'], 0)
        self.assertIs(r['success'], False)
        self.assertEqual(r['image_offset_declared'], 0)
        self.assertEqual(r['poses']['action/left_ee_poses']['sign_flip_count'], 2)
        self.assertAlmostEqual(r['poses']['action/left_ee_poses']['max_physical_rotation_step_rad'], 0)
        self.assertEqual(len(r['fk']), 4)
        self.assertLess(r['fk']['action/left']['max_position_error_m'], 1e-10)
        self.assertGreater(r['normalization']['action']['elements_over5'], 0)
        self.assertFalse(r['normalization']['action']['data_clipped'])
        self.assertEqual(sum(v['decoded_reduced'] for v in r['images'].values()), 15)

    def test_corruption_and_black_are_separate(self):
        p = self.raw / 'bad.hdf5'; legacy(p, black=True, corrupt=True)
        with h5py.File(p, 'r+') as f:
            f['action/left_arm_joint_states'][0, 0] = np.nan
        r = self.scan(p)
        codes = {x['code'] for x in r['issues']}
        self.assertIn('nonfinite', codes)
        self.assertIn('jpeg_decode_failed', codes)
        self.assertTrue(any(x.get('near_black') and x['rgb_frame'] == 2 for x in r['image_flags']))
        self.assertTrue(any(x.get('decode_failed') and x['rgb_frame'] == 3 for x in r['image_flags']))

    def test_dense_mapping_and_missing_evidence(self):
        p = self.raw / 'dense.hdf5'
        with h5py.File(p, 'w') as f:
            f.attrs['metadata'] = json.dumps({'dt': .004, 'rgb_stride': 10, 'instruction': 'Pick up'})
            f.create_dataset('observations/qpos', data=np.zeros((30, 16)))
            f.create_dataset('actions/joint_position', data=np.zeros((30, 16)))
            f.create_dataset('rgb_step_index', data=np.array([0, 10, 20]))
            images(f, 'rgb', count=3)
        r = self.scan(p)
        self.assertEqual(r['error_count'], 0)
        self.assertEqual(r['rgb_frames'], 3)
        self.assertEqual(r['control_rows'], 30)
        self.assertEqual(r['rgb_hz_declared'], 25)
        self.assertIsNone(r['image_offset_declared'])
        self.assertIsNone(r['success'])
        self.assertNotIn('action_ee16_sha256', r)
        with h5py.File(p, 'r+') as f:
            f['rgb_step_index'][1] = 31
        r = self.scan(p)
        self.assertIn('invalid_rgb_step_index', {x['code'] for x in r['issues']})

    def test_native_and_unknown_schema(self):
        p = self.raw / 'native.hdf5'
        with h5py.File(p, 'w') as f:
            ee = np.zeros((5, 2, 7)); ee[:, :, 3] = 1
            f.create_dataset('observations/ee', data=ee)
            f.create_dataset('observations/joints', data=np.zeros((5, 2, 6)))
            f.create_dataset('rgb_step_index', data=np.arange(5))
            for side in ('left', 'right'):
                f.create_dataset(f'actions/{side}_arm_joint_state', data=np.zeros((5, 6)))
                f.create_dataset(f'actions/{side}_ee_joint_state', data=np.zeros((5, 1)))
            images(f, 'rgb')
        self.assertEqual(self.scan(p)['error_count'], 0)
        unknown = self.raw / 'unknown.hdf5'
        with h5py.File(unknown, 'w') as f:
            images(f, 'rgb')
        self.assertIn('unsupported_schema', {x['code'] for x in self.scan(unknown)['issues']})

    def test_cli_duplicate_semantics_and_evidence_image(self):
        a, b = self.raw / 'a.hdf5', self.raw / 'b.hdf5'
        legacy(a, seed=1); legacy(b, seed=2)
        manifest = self.root / 'inputs.jsonl'
        manifest.write_text(''.join(json.dumps({'path': str(p), 'task': 'test'}) + '\n' for p in (a, b)))
        out = self.root / 'audit'
        result = subprocess.run([sys.executable, str(SKILL / 'scripts/check_hdf5.py'), '--inputs', str(manifest), '--out', str(out), '--workers', '1'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        dup = json.loads((out / 'duplicate_groups.json').read_text())
        self.assertEqual(len(dup['action_ee16_sha256']), 1)
        self.assertEqual(dup['visual_sha256'], [])
        report = json.loads((out / 'report.json').read_text())
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['completed'], 2)
        png = self.root / 'evidence.png'
        result = subprocess.run([sys.executable, str(SKILL / 'scripts/review_frames.py'), '--hdf5', str(a), '--frame', '0', '--color', 'unknown', '--out', str(png)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(png) as im:
            self.assertEqual(im.size, (960, 1742))
            pixels = np.asarray(im)
            self.assertGreater(pixels[102:150, 320:384, 0].mean(), pixels[102:150, 320:384, 2].mean())
            self.assertLess(pixels[942:990, 320:384, 0].mean(), pixels[942:990, 320:384, 2].mean())


if __name__ == '__main__':
    unittest.main(verbosity=2)
