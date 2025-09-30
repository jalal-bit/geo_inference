import os
import glob
import argparse
import pandas as pd
from pathlib import Path


def process_file(csv_path: Path):
    """Read one CSV and return aggregated counts by state/county"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}")
        return None

    # Skip if missing required columns
    if not {"is_job", "is_traffic"}.issubset(df.columns):
        print(f"[SKIP] {csv_path} missing required columns")
        return None

    # Parse state/county from folder structure
    parts = csv_path.parts
    # Assuming .../state/county/filename.csv
    state = parts[-3]
    county = parts[-2]

    total = len(df)
    is_job_count = df["is_job"].sum()
    is_traffic_count = df["is_traffic"].sum()

    return {
        "state": state,
        "county": county,
        "raw_count": total,
        "is_job_count": is_job_count,
        "is_traffic_count": is_traffic_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Top-level directory with state/county CSV files"
    )
    parser.add_argument(
        "--county_out",
        type=str,
        default="county_level_counts.csv",
        help="Output CSV file for county + state totals"
    )
    parser.add_argument(
        "--state_out",
        type=str,
        default="state_level_counts.csv",
        help="Output CSV file for state totals"
    )
    args = parser.parse_args()

    # Find all CSVs recursively
    csvs = glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True)
    print(f"[INFO] Found {len(csvs)} CSVs under {args.data_dir}")

    records = []
    for csv_path in sorted(csvs):
        rec = process_file(Path(csv_path))
        if rec:
            records.append(rec)

    if not records:
        print("[ERROR] No valid CSVs found with required columns.")
        return

    df = pd.DataFrame(records)

    # --- County-level proportions
    df["job_prop"] = df["is_job_count"] / df["raw_count"]
    df["traffic_prop"] = df["is_traffic_count"] / df["raw_count"]

    # --- State-level aggregation
    state_df = df.groupby("state").agg(
        raw_count=("raw_count", "sum"),
        is_job_count=("is_job_count", "sum"),
        is_traffic_count=("is_traffic_count", "sum"),
    ).reset_index()
    state_df["job_prop"] = state_df["is_job_count"] / state_df["raw_count"]
    state_df["traffic_prop"] = state_df["is_traffic_count"] / state_df["raw_count"]

    # --- Merge state totals into county-level results
    df = df.merge(state_df, on="state", suffixes=("", "_state"))

    # --- Sorting
    df = df.sort_values(
        by=["raw_count_state", "raw_count"], ascending=[False, False]
    )
    state_df = state_df.sort_values(by="raw_count", ascending=False)

    # Save both CSVs
    df.to_csv(args.county_out, index=False)
    state_df.to_csv(args.state_out, index=False)

    print(f"[OK] Wrote county-level results to {args.county_out}")
    print(f"[OK] Wrote state-level results to {args.state_out}")


if __name__ == "__main__":
    main()
