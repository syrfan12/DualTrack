"""Check or fine-tune pretrained DualTrack fusion/head on original MDPI JPGs."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from src.datasets.mdpi_original import MDPIOriginalDataset
from src.models.mdpi_fusion import build_mdpi_fusion
from src.mdpi_training import seed_everything, check_weights, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dualtrack_finetunes/mdpi_original_fusion.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate train/val metadata and source bytes (default)")
    mode.add_argument("--train", action="store_true", help="Explicitly start full optimization")
    mode.add_argument("--smoke-step", action="store_true", help="One train-window optimizer update, no checkpoint saved")
    parser.add_argument("--check-weights", action="store_true", help="Add a real train-window forward to check-only")
    parser.add_argument("--report", type=Path, help="Save check/smoke report to a new JSON file")
    parser.add_argument("--resume", type=Path, help="Resume into a NEW output directory")
    parser.add_argument("--skip-trajectory-eval", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if config.pop("schema", None) != "mdpi_original_fusion_config_v1":
        parser.error("Unsupported configuration schema")
    for key in ("manifest", "pretrained", "output", "device", "batch_size", "num_workers"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    for key in ("manifest", "pretrained", "output"):
        path = Path(config[key])
        config[key] = str((path if path.is_absolute() else ROOT / path).resolve())
    if min(config["batch_size"], config["cpu_threads"], config["max_epochs"], config["patience"]) < 1:
        parser.error("Batch size, CPU threads, max epochs and patience must be positive")
    if config["num_workers"] < 0 or config["grid_spacing"] < 1:
        parser.error("Invalid worker/sampling configuration")
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    if args.report and (args.train or args.report.exists()):
        parser.error("--report requires check/smoke mode and a new output file")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    if args.train:
        result = train(config, resume=args.resume, skip_trajectory_eval=args.skip_trajectory_eval)
    else:
        dataset = MDPIOriginalDataset(config["manifest"], "train")
        val = MDPIOriginalDataset(config["manifest"], "val")
        result = {"mode": "dataset_check", "train_windows": len(dataset), "val_windows": len(val),
                  "train_scans": len(dataset.scans), "val_scans": len(val.scans),
                  "manifest_sha256": dataset.manifest_sha256}
        if args.check_weights or args.smoke_step:
            model = build_mdpi_fusion(config["pretrained"], config["grid_spacing"], config["device"])
            result["weights"] = check_weights(model, dataset, config, optimize=args.smoke_step)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    console = copy.deepcopy(result)
    if "weights" in console:
        console["weights"]["scope"].pop("trainable_names", None)
    print(json.dumps(console, indent=2, allow_nan=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            handle.write(encoded + "\n")
    return result


if __name__ == "__main__":
    main()
