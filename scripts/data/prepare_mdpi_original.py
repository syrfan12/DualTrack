"""Prepare original-JPEG MDPI metadata, preserving the frozen v4 split/windows."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.datasets.mdpi_original import (
    DEFAULT_OUTPUT, DEFAULT_PREPROCESSING, DEFAULT_REFERENCE, DEFAULT_SOURCE_ROOT,
    prepare_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--reference-h5", type=Path, default=DEFAULT_REFERENCE,
                        help="Frozen split/label metadata only; resized pixels are never read")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    parser.add_argument("--global-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    args = parser.parse_args()
    spec = copy.deepcopy(DEFAULT_PREPROCESSING)
    if args.local_crop is not None:
        spec["local_crop_xywh"] = args.local_crop
    if args.global_roi is not None:
        spec["global_roi_xywh"] = args.global_roi
    manifest, report = prepare_dataset(
        args.source_root, args.reference_h5, args.output, spec,
        progress=lambda scan_id, source, count: print(f"{scan_id}: {source}: {count} original JPEGs checked", flush=True))
    print(json.dumps({"manifest": str(manifest), **report}, indent=2))


if __name__ == "__main__":
    main()
