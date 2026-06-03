from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conversion CSV for TUS-REC validation data with separate "
            "frames and transfs files."
        )
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("data/val/frames"),
        help="Path to frames directory (expects scan_id/*.h5).",
    )
    parser.add_argument(
        "--tforms_dir",
        type=Path,
        default=Path("data/val/transfs"),
        help="Path to transfs directory (expects matching scan_id/*.h5).",
    )
    parser.add_argument(
        "--output_csv_path",
        "-o",
        type=Path,
        default=Path("data/val/val_meta_for_convert_clean.csv"),
        help="Path to generated CSV for converter input.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split value for all rows (default: train).",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Write paths relative to current working directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any file pair is missing or has invalid H5 keys.",
    )
    return parser.parse_args()


def make_sweep_id(frame_file: Path, frames_dir: Path) -> str:
    rel = frame_file.relative_to(frames_dir)
    return f"{rel.parent.name}_{rel.stem}"


def maybe_relativize(path: Path, relative_paths: bool) -> Path:
    if not relative_paths:
        return path
    try:
        return path.relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def has_dataset_key(path: Path, key: str) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return key in handle
    except OSError:
        return False


def main() -> None:
    args = parse_args()

    frames_dir = args.frames_dir.resolve()
    tforms_dir = args.tforms_dir.resolve()
    output_csv_path = args.output_csv_path.resolve()

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir not found or not a directory: {frames_dir}")
    if not tforms_dir.is_dir():
        raise FileNotFoundError(f"tforms_dir not found or not a directory: {tforms_dir}")

    frame_files = sorted(p for p in frames_dir.rglob("*.h5") if p.is_file())
    if not frame_files:
        raise FileNotFoundError(f"No .h5 files found under: {frames_dir}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: dict[str, int] = {}
    rows: list[dict[str, str]] = []

    skipped_missing = 0
    skipped_invalid = 0

    for frame_file in frame_files:
        rel = frame_file.relative_to(frames_dir)
        tforms_file = tforms_dir / rel

        if not tforms_file.exists():
            skipped_missing += 1
            msg = f"Missing tforms pair: {tforms_file}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}")
            continue

        if not has_dataset_key(frame_file, "frames"):
            skipped_invalid += 1
            msg = f"Missing key 'frames' in: {frame_file}"
            if args.strict:
                raise ValueError(msg)
            print(f"[skip] {msg}")
            continue

        if not has_dataset_key(tforms_file, "tforms"):
            skipped_invalid += 1
            msg = f"Missing key 'tforms' in: {tforms_file}"
            if args.strict:
                raise ValueError(msg)
            print(f"[skip] {msg}")
            continue

        sweep_id = make_sweep_id(frame_file, frames_dir)
        if sweep_id in seen_ids:
            seen_ids[sweep_id] += 1
            sweep_id = f"{sweep_id}__{seen_ids[sweep_id]}"
        else:
            seen_ids[sweep_id] = 0

        frame_path = maybe_relativize(frame_file.resolve(), args.relative_paths)
        tforms_path = maybe_relativize(tforms_file.resolve(), args.relative_paths)

        rows.append(
            {
                "sweep_id": sweep_id,
                "raw_tus_rec_frames_path": str(frame_path),
                "raw_tus_rec_tforms_path": str(tforms_path),
                "split": args.split,
            }
        )

    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "sweep_id",
                "raw_tus_rec_frames_path",
                "raw_tus_rec_tforms_path",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated CSV: {output_csv_path}")
    print(f"Valid sweep pairs: {len(rows)}")
    print(f"Skipped missing pairs: {skipped_missing}")
    print(f"Skipped invalid H5 keys: {skipped_invalid}")


if __name__ == "__main__":
    main()
