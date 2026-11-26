import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.mdpi_evaluation import (
    aggregate_adjacent_steps, evaluate_trajectories, evaluate_windows,
    fit_train_baselines, integrate_steps, trajectory_metrics, window_plan,
    _control_inputs,
)


class DifferenceModel(torch.nn.Module):
    def forward(self, global_inputs, local_inputs):
        return torch.diff(global_inputs, dim=1)


class WindowDataset(Dataset):
    split = 'val'

    def __init__(self):
        self.positions = torch.stack([torch.arange(10) ** 2, -torch.arange(10)], dim=1).float()

    def __len__(self):
        return 3

    def __getitem__(self, index):
        points = self.positions + index * 30
        return {'global_encoder_images': points, 'local_encoder_images': points,
                'positions_yz': points}


class TrajectoryDataset:
    split = 'val'
    manifest_sha256 = 'synthetic'

    def __init__(self, valid=True):
        positions = np.column_stack([np.arange(25) ** 2, -np.arange(25)]).tolist()
        self.scans = {
            'included': {'split': 'val', 'source_scan': 'synthetic/included',
                         'source_indices': list(range(25)), 'eligible': [valid] * 25,
                         'positions_yz': positions},
            'short': {'split': 'val', 'source_scan': 'synthetic/short',
                      'source_indices': list(range(5)), 'eligible': [valid] * 5,
                      'positions_yz': positions[:5]},
        }

    def load_window(self, scan_id, rows):
        points = torch.tensor(np.asarray(self.scans[scan_id]['positions_yz'])[rows], dtype=torch.float32)
        return {'global_encoder_images': points, 'local_encoder_images': points,
                'positions_yz': points}


class EvaluationTests(unittest.TestCase):
    def test_tail_and_duplicate_edge_averaging(self):
        rows = np.arange(25)
        plan = window_plan(rows, np.ones(25, dtype=bool))
        windows = plan[0]['windows']
        self.assertEqual(windows[:, 0].tolist(), [0, 9, 15])
        predictions = np.ones((3, 9, 2))
        predictions[-1] *= 3
        steps, count = aggregate_adjacent_steps(rows, windows, predictions)
        self.assertEqual(len(steps), 24)
        self.assertTrue(np.all(count > 0))
        np.testing.assert_array_equal(count[15:18], 2)
        np.testing.assert_array_equal(steps[15:18], 2)

    def test_gap_and_short_segments(self):
        indices = np.r_[np.arange(10), np.arange(20, 25)]
        plan = window_plan(indices, np.ones(15, dtype=bool))
        self.assertEqual([row['status'] for row in plan], ['included', 'skipped_short_segment'])
        self.assertEqual(plan[1]['rows'].tolist(), list(range(10, 15)))
        with self.assertRaises(ValueError):
            aggregate_adjacent_steps(np.arange(12), np.arange(10)[None], np.ones((1, 9, 2)))

    def test_signed_integration_and_no_scale_alignment(self):
        points = integrate_steps([[1, -2], [-3, 4]])
        np.testing.assert_array_equal(points, [[0, 0, 0], [0, 1, -2], [0, -2, 2]])
        self.assertEqual(trajectory_metrics(points + 11, points)['position_rmse_nominal_mm'], 0)
        self.assertGreater(trajectory_metrics(2 * points, points)['position_rmse_nominal_mm'], 0)

    def test_window_pair_direction_and_denominator(self):
        metrics = evaluate_windows(DifferenceModel(), DataLoader(WindowDataset(), batch_size=2))
        self.assertEqual(metrics['windows'], 3)
        self.assertEqual(metrics['pairs'], 135)
        self.assertEqual(metrics['mean_pair_error_nominal_mm'], 0)
        self.assertEqual(metrics['mse_yz'], 0)
        self.assertEqual(metrics['mse_denominator_coordinates'], 2)
        self.assertGreater(metrics['baselines']['zero_motion']['mean_pair_error_nominal_mm'], 0)

    def test_baseline_fits_train_only(self):
        points = np.column_stack([np.arange(10) ** 2, -np.arange(10)]).tolist()
        data = SimpleNamespace(split='train', windows=[{'scan_id': 'train', 'frame_indices': list(range(10))}],
                               scans={'train': {'split': 'train', 'positions_yz': points}})
        baseline = fit_train_baselines(data)
        self.assertEqual(baseline['fit_scope'], 'train_only')
        np.testing.assert_array_equal(baseline['window_position_mean_yz'], np.diff(points, axis=0))
        data.split = 'val'
        with self.assertRaises(ValueError):
            fit_train_baselines(data)

    def test_trajectory_coverage_and_empty_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'report'
            report = evaluate_trajectories(DifferenceModel(), TrajectoryDataset(), output)
            self.assertEqual(report['eligible_frames'], 30)
            self.assertEqual(report['evaluated_frames'], 25)
            self.assertEqual(report['all_independently_anchored_segments']['position_rmse_nominal_mm'], 0)
            self.assertEqual(report['single_complete_eligible_segment_scans']['segments'], 1)
            self.assertTrue((output / 'trajectory_examples.png').is_file())
            self.assertIn('skipped_short_segment', (output / 'per_segment.csv').read_text())
            self.assertEqual(len(np.load(output / 'trajectories.npz').files), 4)
            with self.assertRaises(FileExistsError):
                evaluate_trajectories(DifferenceModel(), TrajectoryDataset(), output)
            empty = evaluate_trajectories(DifferenceModel(), TrajectoryDataset(valid=False), Path(temp) / 'empty')
            self.assertEqual(empty['evaluated_frames'], 0)
            self.assertIsNone(empty['all_independently_anchored_segments']['position_rmse_nominal_mm'])
            self.assertNotIn('NaN', json.dumps(empty, allow_nan=False))

    def test_identical_shuffle_applied_to_both_branches(self):
        values = torch.arange(20).reshape(2, 10, 1)
        global_inputs, local_inputs = _control_inputs(values, values * 3, 'temporal_shuffle', torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(local_inputs, global_inputs * 3))
        self.assertFalse(torch.equal(values, global_inputs))

    def test_test_cli_guard_precedes_checkpoint_loading(self):
        from scripts.evaluate_mdpi_fusion import main
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(SystemExit) as raised:
                main(['--checkpoint', str(Path(temp) / 'missing.pt'), '--output', str(Path(temp) / 'report'), '--split', 'test'])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((Path(temp) / 'report').exists())


if __name__ == '__main__':
    unittest.main()
