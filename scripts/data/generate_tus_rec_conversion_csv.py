from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an input CSV for convert_tus_rec_format_to_dualtrack_format.py "
            "from all .h5 files under a datasets directory."
        )
    )
    parser.add_argument(
        "--datasets_dir",
        "-d",
        type=Path,
        default=Path("datasets"),
        help="Path to the datasets directory containing .h5 files.",
    )
    parser.add_argument(
        "--output_csv_path",
        "-o",
        type=Path,
        default=Path("datasets/tus_rec_to_dualtrack_input.csv"),
        help="Path to the generated CSV file.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split value written for every row (default: train).",
    )
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Write paths in CSV relative to current working directory.",
    )
    return parser.parse_args()


def make_sweep_id(h5_path: Path) -> str:
    return f"{h5_path.parent.name}_{h5_path.stem}"


def main() -> None:
    args = parse_args()
    datasets_dir = args.datasets_dir.resolve()
    output_csv_path = args.output_csv_path.resolve()

    if not datasets_dir.exists() or not datasets_dir.is_dir():
        raise FileNotFoundError(f"datasets_dir not found or not a directory: {datasets_dir}")

    h5_files = sorted(p for p in datasets_dir.rglob("*.h5") if p.is_file())
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under: {datasets_dir}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: dict[str, int] = {}
    rows: list[dict[str, str]] = []

    for h5_path in h5_files:
        sweep_id = make_sweep_id(h5_path)
        if sweep_id in seen_ids:
            seen_ids[sweep_id] += 1
            sweep_id = f"{sweep_id}__{seen_ids[sweep_id]}"
        else:
            seen_ids[sweep_id] = 0

        sweep_path = h5_path.resolve()
        if args.relative_paths:
            try:
                sweep_path = sweep_path.relative_to(Path.cwd().resolve())
            except ValueError:
                # Fallback to absolute path if file is outside current working directory.
                pass

        rows.append(
            {
                "sweep_id": sweep_id,
                "raw_tus_rec_sweep_path": str(sweep_path),
                "split": args.split,
            }
        )

    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["sweep_id", "raw_tus_rec_sweep_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated CSV: {output_csv_path}")
    print(f"Total sweeps: {len(rows)}")


if __name__ == "__main__":
    main()