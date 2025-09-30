import os
import glob
import argparse
import pandas as pd
from pathlib import Path


def process_county(pos_path: Path, other_path: Path):
    """Compute county-level stats using _pos and _other CSVs"""
    try:
        pos_df = pd.read_csv(pos_path)
        other_df = pd.read_csv(other_path)
    except Exception as e:
        print(f"[WARN] Could not read {pos_path} / {other_path}: {e}")
        return None

    # Parse state/county from folder structure (.../state/county/*.csv)
    parts = pos_path.parts
    state = parts[-3]
    county_folder = parts[-2]

    # FIPS is the last underscore-split part
    tokens = county_folder.split("_")
    if len(tokens) > 1:
        fips = tokens[-1]
        county_name = "_".join(tokens[:-1])
    else:
        county_name, fips = county_folder, None

    # Compute counts
    pos_count = len(pos_df)
    other_count = len(other_df)
    total = pos_count + other_count

    # job/traffic counts come from _pos file
    is_job_count = pos_df["is_job"].sum() if "is_job" in pos_df.columns else 0
    is_traffic_count = pos_df["is_traffic"].sum() if "is_traffic" in pos_df.columns else 0

    return {
        "state": state,
        "county_name": county_name,
        "fips": fips,
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

    # Find all *_pos.csv files
    pos_files = glob.glob(os.path.join(args.data_dir, "**", "*_pos.csv"), recursive=True)
    print(f"[INFO] Found {len(pos_files)} *_pos.csv files")

    records = []
    for pos_path in sorted(pos_files):
        pos_path = Path(pos_path)
        other_path = pos_path.with_name(pos_path.stem.replace("_pos", "_other") + ".csv")

        if not other_path.exists():
            print(f"[SKIP] Missing {other_path}, skipping {pos_path}")
            continue

        rec = process_county(pos_path, other_path)
        if rec:
            records.append(rec)

    if not records:
        print("[ERROR] No valid county records found")
        return

    df = pd.DataFrame(records)

    # --- County proportions
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
    df = df.sort_values(by=["raw_count_state", "raw_count"], ascending=[False, False])
    state_df = state_df.sort_values(by="raw_count", ascending=False)

    # Save
    df.to_csv(args.county_out, index=False)
    state_df.to_csv(args.state_out, index=False)

    print(f"[OK] Wrote county-level results to {args.county_out}")
    print(f"[OK] Wrote state-level results to {args.state_out}")


if __name__ == "__main__":
    main()
