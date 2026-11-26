"""Evaluate the original-JPG MDPI fusion model; validation is the default."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--image-controls', action='store_true')
    args = parser.parse_args(argv)
    if args.split == 'test' and not args.allow_test:
        parser.error('Test inference requires explicit --allow-test')
    if args.image_controls and args.split != 'val':
        parser.error('--image-controls is restricted to validation')
    if args.batch_size < 1 or args.threads < 1:
        parser.error('--batch-size and --threads must be positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    import torch
    from torch.utils.data import DataLoader
    from src.datasets.mdpi_original import MDPIOriginalDataset
    from src.mdpi_training import load_run_checkpoint
    from src.mdpi_evaluation import evaluate_trajectories, evaluate_image_controls, evaluate_windows

    torch.set_num_threads(args.threads)
    model, checkpoint, manifest = load_run_checkpoint(args.checkpoint, manifest_path=args.manifest, device=args.device)
    dataset = MDPIOriginalDataset(manifest, split=args.split, allow_test=args.allow_test)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    use_amp = bool(checkpoint.get('config', {}).get('amp', False))
    window_metrics = evaluate_windows(model, loader, args.device, use_amp=use_amp,
                                      baselines=checkpoint.get('baselines'))
    report = evaluate_trajectories(model, dataset, args.output, args.batch_size, args.device,
                                   baselines=checkpoint.get('baselines'))
    (args.output / 'window_metrics.json').write_text(json.dumps(window_metrics, indent=2, allow_nan=False) + '\n')
    if args.image_controls:
        controls = evaluate_image_controls(model, loader, args.device, use_amp=use_amp,
                                           baselines=checkpoint.get('baselines'))
        (args.output / 'image_controls.json').write_text(json.dumps(controls, indent=2, allow_nan=False) + '\n')
    provenance = {'checkpoint': str(args.checkpoint.resolve()), 'manifest': str(Path(manifest).resolve()),
                  'split': args.split, 'allow_test': args.allow_test,
                  'checkpoint_sha256': _file_sha256(args.checkpoint),
                  'manifest_sha256': dataset.manifest_sha256}
    (args.output / 'evaluation_config.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    main()
