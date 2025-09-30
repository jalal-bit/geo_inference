import os
import glob
import argparse
import pandas as pd
from pathlib import Path


def process_file(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}")
        return

    # Skip if missing required columns
    if not {"is_job", "is_traffic"}.issubset(df.columns):
        print(f"[SKIP] {csv_path} missing required columns")
        return

    base = csv_path.with_suffix("")  # remove .csv
    out_dir = csv_path.parent

    # --- Split into two groups
    df_positive = df[(df["is_job"] == 1) | (df["is_traffic"] == 1)]
    df_other = df[(df["is_job"] == 0) & (df["is_traffic"] == 0)]

    # --- Save only if non-empty
    if not df_positive.empty:
        out_path = out_dir / f"{base.name}_pos.csv"
        df_positive.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}")

    if not df_other.empty:
        out_path = out_dir / f"{base.name}_other.csv"
        df_other.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Top-level directory with state/county CSV files"
    )
    args = parser.parse_args()

    # Find all CSVs recursively
    csvs = glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True)
    print(f"[INFO] Found {len(csvs)} CSVs under {args.data_dir}")

    for csv_path in sorted(csvs):
        process_file(Path(csv_path))


if __name__ == "__main__":
    main()
