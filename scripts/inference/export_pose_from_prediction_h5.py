from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def pose_matrices_to_table(mats: np.ndarray, prefix: str) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(mats):
        rot = t[:3, :3]
        trans = t[:3, 3]
        euler_xyz_deg = R.from_matrix(rot).as_euler("xyz", degrees=True)
        rows.append(
            {
                "frame_idx": i + 1,
                f"{prefix}_tx": float(trans[0]),
                f"{prefix}_ty": float(trans[1]),
                f"{prefix}_tz": float(trans[2]),
                f"{prefix}_rx_deg": float(euler_xyz_deg[0]),
                f"{prefix}_ry_deg": float(euler_xyz_deg[1]),
                f"{prefix}_rz_deg": float(euler_xyz_deg[2]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = ArgumentParser(description="Export predicted translation and rotation from DualTrack prediction H5")
    parser.add_argument("--input_h5", required=True, help="Path to prediction .h5 file")
    parser.add_argument("--output_csv", required=True, help="Path to output csv")
    args = parser.parse_args()

    input_h5 = Path(args.input_h5)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as f:
        if "pred_tracking_glob" not in f and "pred_tracking_loc" not in f:
            raise KeyError("No pred_tracking_glob or pred_tracking_loc datasets found")

        df = None
        if "pred_tracking_glob" in f:
            glob = f["pred_tracking_glob"][:]
            df_glob = pose_matrices_to_table(glob, "glob")
            df = df_glob if df is None else df.merge(df_glob, on="frame_idx", how="outer")

        if "pred_tracking_loc" in f:
            loc = f["pred_tracking_loc"][:]
            df_loc = pose_matrices_to_table(loc, "loc")
            df = df_loc if df is None else df.merge(df_loc, on="frame_idx", how="outer")

    assert df is not None
    df = df.sort_values("frame_idx").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
