# Screen Capture to Geometry Prediction Guide

This guide explains the full workflow to convert a folder of ultrasound screen-capture frames into DualTrack format, run inference, and export pose geometry (translation and rotation) to CSV.

## 1. Prerequisites

- Repository root: `/home/syahid/github/DualTrack`
- Input frame folder: `/home/syahid/github/DualTrack/data/data_custom/screen_captures`
- Recommended environment: `conda` env named `dualtrack`
- Model checkpoint used in this setup: `/home/syahid/github/DualTrack/dualtrack_final.pt`

If needed, run commands with:

```bash
cd /home/syahid/github/DualTrack
```

## 2. Convert Screen Captures to DualTrack H5 Format

Use the conversion script:

```bash
python scripts/data/convert_screen_captures_to_dualtrack_format.py \
  --input_dir /home/syahid/github/DualTrack/data/data_custom/screen_captures \
  --output_dir /home/syahid/github/DualTrack/data/data_custom/screen_captures_dualtrack \
  --sweep_id session_20260528_182438 \
  --split val
```

This creates:

- `data/data_custom/screen_captures_dualtrack/session_20260528_182438.h5`
- `data/data_custom/screen_captures_dualtrack/metadata.csv`

The H5 file contains keys required by the pipeline:

- `images`
- `tracking`
- `spacing`
- `dimensions`
- `pixel_to_image`

## 3. Run DualTrack Inference

Use the prepared inference config:

- `src/submission/cfg/screen_captures_dualtrack_final.yaml`

Run inference:

```bash
PYTHONPATH=. conda run -n dualtrack python scripts/inference/predict_screen_captures.py \
  --config src/submission/cfg/screen_captures_dualtrack_final.yaml \
  --input_h5 data/data_custom/screen_captures_dualtrack/session_20260528_182438.h5 \
  --output_h5 experiments/eval/screen_captures/prediction_session_20260528_182438.h5
```

Output file:

- `experiments/eval/screen_captures/prediction_session_20260528_182438.h5`

Main prediction datasets inside output H5:

- `pred_tracking_glob` with shape `(N-1, 4, 4)`
- `pred_tracking_loc` with shape `(N-1, 4, 4)`

## 4. Inspect H5 Contents Quickly

```bash
python - <<'PY'
import h5py
p='experiments/eval/screen_captures/prediction_session_20260528_182438.h5'
with h5py.File(p,'r') as f:
    print(list(f.keys()))
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)
PY
```

## 5. Export Translation and Rotation to CSV

Use the pose export script:

```bash
python scripts/inference/export_pose_from_prediction_h5.py \
  --input_h5 experiments/eval/screen_captures/prediction_session_20260528_182438.h5 \
  --output_csv experiments/eval/screen_captures/prediction_session_20260528_182438_pose.csv
```

CSV output:

- `experiments/eval/screen_captures/prediction_session_20260528_182438_pose.csv`

Columns include:

- `frame_idx`
- Global pose: `glob_tx`, `glob_ty`, `glob_tz`, `glob_rx_deg`, `glob_ry_deg`, `glob_rz_deg`
- Local pose: `loc_tx`, `loc_ty`, `loc_tz`, `loc_rx_deg`, `loc_ry_deg`, `loc_rz_deg`

Rotation is exported as Euler angles in `xyz` order, in degrees.

## 6. Notes About Geometry

- Translation values are taken from each 4x4 transform matrix `[0:3, 3]`.
- Rotation values are converted from matrix to Euler `xyz` degrees.
- `pred_tracking_glob` is the cumulative trajectory in the model prediction space.
- `pred_tracking_loc` is frame-to-frame local relative motion represented as transforms.

## 7. Troubleshooting

- If `ModuleNotFoundError: No module named 'src'`, run with `PYTHONPATH=.` from repo root.
- If `ModuleNotFoundError` for libraries such as `omegaconf`, run inside the `dualtrack` conda environment.
- If GPU is unavailable, set `device: cpu` in `src/submission/cfg/screen_captures_dualtrack_final.yaml`.
