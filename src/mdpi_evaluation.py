"""Evaluation of the fixed-orientation MDPI translation experiment.

Distances are against the encoder/interpolated reference in nominal millimetres.
They are not calibrated probe-pose or image reconstruction accuracy metrics.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch


PAIR_INDICES = np.array([[i, j] for j in range(1, 10) for i in range(j)], dtype=np.int64)


def window_plan(source_indices, valid, length=10, max_source_gap=3):
    """Cover edges of eligible segments with stride length-1 and a full tail."""
    source_indices, valid = np.asarray(source_indices), np.asarray(valid)
    if (source_indices.ndim != 1 or source_indices.dtype.kind not in 'iu'
            or valid.shape != source_indices.shape or valid.dtype != np.bool_
            or len(source_indices) < 1 or np.any(np.diff(source_indices) <= 0)
            or length < 2 or max_source_gap < 1):
        raise ValueError('Invalid source indices, eligibility, or window settings')
    rows = np.flatnonzero(valid)
    if not len(rows):
        return []
    groups = np.split(rows, np.flatnonzero(np.diff(source_indices[rows]) > max_source_gap) + 1)
    result = []
    for segment in groups:
        if len(segment) < length:
            windows = np.empty((0, length), dtype=np.int64)
        else:
            starts = list(range(0, len(segment) - length + 1, length - 1))
            if starts[-1] != len(segment) - length:
                starts.append(len(segment) - length)
            windows = np.stack([segment[start:start + length] for start in starts])
        result.append({'rows': segment, 'windows': windows,
                       'status': 'included' if len(windows) else 'skipped_short_segment'})
    return result


def aggregate_adjacent_steps(segment_rows, windows, predicted_steps):
    """Average predictions only for identical adjacent eligible edges."""
    rows, windows = np.asarray(segment_rows), np.asarray(windows)
    steps = np.asarray(predicted_steps, dtype=np.float64)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(rows) < 2
            or np.any(np.diff(rows) <= 0) or windows.ndim != 2
            or windows.shape[1] < 2 or windows.dtype.kind not in 'iu'
            or steps.shape != (len(windows), windows.shape[1] - 1, 2)
            or not np.isfinite(steps).all()):
        raise ValueError('Invalid segment windows or adjacent predictions')
    lookup = {int(row): position for position, row in enumerate(rows)}
    total = np.zeros((len(rows) - 1, 2), dtype=np.float64)
    count = np.zeros(len(rows) - 1, dtype=np.int64)
    for window, values in zip(windows, steps):
        if any(int(row) not in lookup for row in window):
            raise ValueError('Window contains rows outside the segment')
        positions = np.array([lookup[int(row)] for row in window])
        if not np.all(np.diff(positions) == 1):
            raise ValueError('Window does not contain contiguous eligible rows')
        total[positions[:-1]] += values
        count[positions[:-1]] += 1
    if np.any(count == 0):
        raise ValueError('Window stitching leaves uncovered edges')
    return total / count[:, None], count


def integrate_steps(steps_yz):
    steps = np.asarray(steps_yz, dtype=np.float64)
    if steps.ndim != 2 or steps.shape[1] != 2 or not np.isfinite(steps).all():
        raise ValueError('Expected finite adjacent Y/Z displacements')
    points = np.zeros((len(steps) + 1, 3), dtype=np.float64)
    points[1:, 1:] = np.cumsum(steps, axis=0)
    return points


def trajectory_metrics(predicted, reference):
    """Rebase the initial point only; never fit rotation, scale, or shape."""
    predicted, reference = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (predicted.ndim != 2 or predicted.shape != reference.shape
            or predicted.shape[1] != 3 or len(predicted) < 2
            or not np.isfinite(predicted - reference).all()):
        raise ValueError('Expected matching finite N by 3 trajectories, N>=2')
    predicted, reference = predicted - predicted[0], reference - reference[0]
    point_errors = np.linalg.norm(predicted - reference, axis=1)
    step_errors = np.linalg.norm(np.diff(predicted, axis=0) - np.diff(reference, axis=0), axis=1)
    reference_length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
    return {
        'points': len(predicted), 'edges': len(predicted) - 1,
        'position_mean_error_nominal_mm': float(point_errors.mean()),
        'position_rmse_nominal_mm': float(np.sqrt(np.square(point_errors).mean())),
        'position_max_error_nominal_mm': float(point_errors.max()),
        'endpoint_drift_nominal_mm': float(point_errors[-1]),
        'adjacent_step_mean_error_nominal_mm': float(step_errors.mean()),
        'reference_path_length_nominal_mm': reference_length,
        'predicted_path_length_nominal_mm': float(np.linalg.norm(np.diff(predicted, axis=0), axis=1).sum()),
        'endpoint_drift_fraction_of_reference_length': float(point_errors[-1] / reference_length)
        if reference_length > 0 else None,
    }


def fit_train_baselines(dataset_train):
    """Fit the index-blind nine-position mean only on frozen train windows."""
    if getattr(dataset_train, 'split', None) != 'train':
        raise ValueError('Baseline fitting requires a train dataset')
    steps = []
    for window in dataset_train.windows:
        scan = dataset_train.scans[str(window['scan_id'])]
        if scan['split'] != 'train':
            raise ValueError('Non-train scan encountered in baseline fitting')
        positions = np.asarray(scan['positions_yz'], dtype=np.float64)[window['frame_indices']]
        if positions.shape != (10, 2) or not np.isfinite(positions).all():
            raise ValueError('Invalid train positions')
        steps.append(np.diff(positions, axis=0))
    if not steps:
        raise ValueError('Cannot fit baseline without train windows')
    return {'fit_scope': 'train_only', 'train_windows': len(steps),
            'window_position_mean_yz': np.mean(steps, axis=0).tolist(),
            'uses_source_index_gaps': False}


def _pair_translations(steps):
    if steps.ndim != 3 or steps.shape[1:] != (9, 2):
        raise ValueError('Model must return [batch,9,2] adjacent Y/Z steps')
    positions = torch.cat([steps.new_zeros((len(steps), 1, 2)), steps.cumsum(dim=1)], dim=1)
    pairs = torch.as_tensor(PAIR_INDICES, device=steps.device)
    return positions[:, pairs[:, 1]] - positions[:, pairs[:, 0]]


def _target_pairs(positions):
    if positions.ndim != 3 or positions.shape[1:] != (10, 2) or not torch.isfinite(positions).all():
        raise ValueError('Expected finite [batch,10,2] reference positions')
    pairs = torch.as_tensor(PAIR_INDICES, device=positions.device)
    return positions[:, pairs[:, 1]] - positions[:, pairs[:, 0]]


def _baseline_steps(baselines):
    values = np.asarray(baselines['window_position_mean_yz'], dtype=np.float64)
    if values.shape != (9, 2) or not np.isfinite(values).all() or baselines.get('fit_scope') != 'train_only':
        raise ValueError('Invalid train-only baseline')
    return values


def _control_inputs(global_inputs, local_inputs, control, generator):
    if control is None:
        return global_inputs, local_inputs
    if control == 'blank':
        return torch.zeros_like(global_inputs), torch.zeros_like(local_inputs)
    if control == 'repeat_first':
        return global_inputs[:, :1].expand_as(global_inputs), local_inputs[:, :1].expand_as(local_inputs)
    if control == 'temporal_shuffle':
        permutations = torch.stack([torch.randperm(10, generator=generator) for _ in range(len(global_inputs))])
        batch = torch.arange(len(global_inputs))[:, None]
        return global_inputs[batch, permutations], local_inputs[batch, permutations]
    raise ValueError('Unknown image sensitivity control')


@torch.inference_mode()
def evaluate_windows(model, loader, device='cpu', use_amp=False, baselines=None, control=None):
    """Window metrics; MSE is averaged over the TWO observed coordinates."""
    device = torch.device(device)
    model.eval()
    squared_error, error, pairs_count, windows_count = 0.0, 0.0, 0, 0
    baseline_errors = {'zero_motion': 0.0}
    baseline_steps = _baseline_steps(baselines) if baselines is not None else None
    if baseline_steps is not None:
        baseline_errors['constant_window_position'] = 0.0
    generator = torch.Generator().manual_seed(0)
    for batch in loader:
        global_inputs, local_inputs = _control_inputs(batch['global_encoder_images'], batch['local_encoder_images'], control, generator)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == 'cuda')):
            steps = model(global_inputs.to(device), local_inputs.to(device))
        steps = steps.float()
        if not torch.isfinite(steps).all():
            raise ValueError('Nonfinite model predictions')
        prediction = _pair_translations(steps)
        target = _target_pairs(batch['positions_yz'].to(device).float())
        difference = prediction - target
        squared_error += float(difference.square().sum())
        error += float(torch.linalg.vector_norm(difference, dim=-1).sum())
        pairs_count += target.shape[0] * target.shape[1]
        windows_count += target.shape[0]
        baseline_errors['zero_motion'] += float(torch.linalg.vector_norm(target, dim=-1).sum())
        if baseline_steps is not None:
            constant = torch.as_tensor(baseline_steps, device=device, dtype=target.dtype)[None].expand(len(target), -1, -1)
            baseline_errors['constant_window_position'] += float(torch.linalg.vector_norm(_pair_translations(constant) - target, dim=-1).sum())
    if not pairs_count:
        raise ValueError('Evaluation loader contains no windows')
    return {'windows': windows_count, 'pairs': pairs_count,
            'mean_pair_error_nominal_mm': error / pairs_count,
            'mse_yz': squared_error / (pairs_count * 2),
            'mse_denominator_coordinates': 2,
            'baselines': {name: {'mean_pair_error_nominal_mm': value / pairs_count}
                          for name, value in baseline_errors.items()},
            'image_control': control}


def evaluate_image_controls(model, loader, device='cpu', use_amp=False, baselines=None):
    if getattr(loader.dataset, 'split', None) != 'val':
        raise ValueError('Image sensitivity controls are restricted to validation')
    return {'scope': 'validation sensitivity controls; not valid-acquisition accuracy',
            'controls': {control: evaluate_windows(model, loader, device, use_amp, baselines, control)
                         for control in ('repeat_first', 'blank', 'temporal_shuffle')}}


@torch.inference_mode()
def _predict_windows(model, dataset, scan_id, windows, batch_size, device):
    result = []
    for start in range(0, len(windows), batch_size):
        samples = [dataset.load_window(scan_id, row.tolist()) for row in windows[start:start + batch_size]]
        global_inputs = torch.stack([row['global_encoder_images'] for row in samples]).to(device)
        local_inputs = torch.stack([row['local_encoder_images'] for row in samples]).to(device)
        prediction = model(global_inputs, local_inputs).float().cpu().numpy()
        if prediction.shape != (len(samples), 9, 2) or not np.isfinite(prediction).all():
            raise ValueError('Expected finite [batch,9,2] model predictions')
        result.append(prediction)
    return np.concatenate(result, axis=0)


def _aggregate_metrics(metrics):
    if not metrics:
        return {'segments': 0, 'points': 0, 'edges': 0,
                'position_mean_error_nominal_mm': None, 'position_rmse_nominal_mm': None,
                'mean_endpoint_drift_nominal_mm': None, 'adjacent_step_mean_error_nominal_mm': None}
    points, edges = sum(m['points'] for m in metrics), sum(m['edges'] for m in metrics)
    return {
        'segments': len(metrics), 'points': points, 'edges': edges,
        'position_mean_error_nominal_mm': sum(m['position_mean_error_nominal_mm'] * m['points'] for m in metrics) / points,
        'position_rmse_nominal_mm': float(np.sqrt(sum(m['position_rmse_nominal_mm'] ** 2 * m['points'] for m in metrics) / points)),
        'mean_endpoint_drift_nominal_mm': float(np.mean([m['endpoint_drift_nominal_mm'] for m in metrics])),
        'adjacent_step_mean_error_nominal_mm': sum(m['adjacent_step_mean_error_nominal_mm'] * m['edges'] for m in metrics) / edges,
    }


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def evaluate_trajectories(model, dataset, output_dir, batch_size=4, device='cpu', baselines=None):
    """Evaluate every eligible segment, with independent first-point anchors."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Choose a new evaluation output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    model.eval()
    baseline = _baseline_steps(baselines) if baselines is not None else None
    segments, scans, arrays, examples = [], [], {}, []
    model_metrics, zero_metrics, constant_metrics, complete_metrics = [], [], [], []
    eligible_frames = evaluated_frames = eligible_edges = evaluated_edges = 0
    for scan_id, scan in sorted(dataset.scans.items()):
        indices = np.asarray(scan['source_indices'], dtype=np.int64)
        valid = np.asarray(scan['eligible'], dtype=bool)
        positions = np.asarray(scan['positions_yz'], dtype=np.float64)
        if positions.shape != (len(indices), 2):
            raise ValueError('Scan reference shape does not match source indices')
        plan = window_plan(indices, valid)
        included = [part for part in plan if part['status'] == 'included']
        complete = len(plan) == 1 and len(included) == 1
        scan_metrics = []
        scan_points = scan_edges = 0
        eligible_frames += int(valid.sum())
        eligible_edges += sum(max(0, len(part['rows']) - 1) for part in plan)
        for segment_index, part in enumerate(plan):
            rows, windows = part['rows'], part['windows']
            record = {'scan_id': scan_id, 'source_scan': scan.get('source_scan', scan_id),
                      'segment': segment_index, 'status': part['status'],
                      'eligible_frames': len(rows), 'source_start': int(indices[rows[0]]),
                      'source_end': int(indices[rows[-1]]), 'inference_windows': len(windows),
                      'single_complete_eligible_segment': complete}
            if not len(windows):
                segments.append(record)
                continue
            predicted_steps = _predict_windows(model, dataset, scan_id, windows, batch_size, device)
            steps, counts = aggregate_adjacent_steps(rows, windows, predicted_steps)
            prediction = integrate_steps(steps)
            reference = np.column_stack([np.zeros(len(rows)), positions[rows]])
            reference -= reference[0]
            metrics = trajectory_metrics(prediction, reference)
            record.update(metrics)
            model_metrics.append(metrics)
            scan_metrics.append(metrics)
            if complete:
                complete_metrics.append(metrics)
            zeros = trajectory_metrics(np.zeros_like(reference), reference)
            zero_metrics.append(zeros)
            record['zero_motion_position_rmse_nominal_mm'] = zeros['position_rmse_nominal_mm']
            if baseline is not None:
                constant_steps, _ = aggregate_adjacent_steps(rows, windows, np.broadcast_to(baseline, (len(windows), 9, 2)))
                constant = trajectory_metrics(integrate_steps(constant_steps), reference)
                constant_metrics.append(constant)
                record['constant_position_rmse_nominal_mm'] = constant['position_rmse_nominal_mm']
            segments.append(record)
            scan_points += len(rows)
            scan_edges += len(rows) - 1
            key = f'scan_{scan_id}_segment_{segment_index}'
            arrays[key + '_source_indices'] = indices[rows]
            arrays[key + '_prediction_xyz'] = prediction
            arrays[key + '_reference_xyz'] = reference
            arrays[key + '_edge_prediction_counts'] = counts
            if len(examples) < 6:
                examples.append((scan_id, segment_index, prediction, reference))
        evaluated_frames += scan_points
        evaluated_edges += scan_edges
        scans.append({'scan_id': scan_id, 'source_scan': scan.get('source_scan', scan_id),
                      'eligible_frames': int(valid.sum()), 'evaluated_frames': scan_points,
                      'eligible_segments': len(plan), 'evaluated_segments': len(included),
                      'single_complete_eligible_segment': complete, **_aggregate_metrics(scan_metrics)})
    report = {
        'schema': 'dualtrack_mdpi_trajectory_report_v1', 'split': dataset.split,
        'manifest_sha256': dataset.manifest_sha256,
        'units': 'nominal_mm_against_interpolated_reference',
        'anchor_policy': 'one independent origin per eligible segment; no rotation or scale fitting; no gap bridging',
        'window_policy': {'length': 10, 'stride': 9, 'tail': 'append_final_full_window', 'max_source_gap': 3,
                          'duplicate_edges': 'mean_prediction'},
        'scans': len(scans), 'eligible_frames': eligible_frames, 'evaluated_frames': evaluated_frames,
        'eligible_edges': eligible_edges, 'evaluated_edges': evaluated_edges,
        'frame_coverage_fraction': evaluated_frames / eligible_frames if eligible_frames else None,
        'edge_coverage_fraction': evaluated_edges / eligible_edges if eligible_edges else None,
        'all_independently_anchored_segments': _aggregate_metrics(model_metrics),
        'single_complete_eligible_segment_scans': _aggregate_metrics(complete_metrics),
        'baselines': {'zero_motion': _aggregate_metrics(zero_metrics)},
        'baseline_fit_scope': baselines.get('fit_scope') if baselines else None,
    }
    if constant_metrics:
        report['baselines']['constant_window_position'] = _aggregate_metrics(constant_metrics)
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    _write_csv(output_dir / 'per_segment.csv', segments)
    _write_csv(output_dir / 'per_scan.csv', scans)
    np.savez_compressed(output_dir / 'trajectories.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if examples:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        for ax, (scan_id, segment, prediction, reference) in zip(axes.flat, examples):
            ax.plot(reference[:, 2], reference[:, 1], label='Interpolated reference')
            ax.plot(prediction[:, 2], prediction[:, 1], label='Prediction')
            ax.set_title(f'{scan_id}, segment {segment}')
            ax.set_xlabel('Z (nominal mm)')
            ax.set_ylabel('Y (nominal mm)')
            ax.axis('equal')
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(examples):]:
            ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(.5, .5, 'No eligible segment has at least 10 frames', ha='center', va='center')
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / 'trajectory_examples.png', dpi=150)
    plt.close(fig)
    return report
