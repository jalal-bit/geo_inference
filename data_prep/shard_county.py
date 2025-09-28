import os
import glob
import argparse
from pathlib import Path

import pandas as pd

SHARD_PREFIX = ".shard"  # suffix marker for shard files


def shard_dataframe(df, num_shards):
    """Split DataFrame into roughly equal shards."""
    shards = []
    n = len(df)
    per = n // num_shards
    extras = n % num_shards
    start = 0
    for r in range(num_shards):
        end = start + per + (1 if r < extras else 0)
        shards.append(df.iloc[start:end].copy())
        start = end
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing input CSV files")
    parser.add_argument("--num_shards", type=int, required=True,
                        help="How many shards to split each CSV into")
    args = parser.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*", "*.csv")))
    if len(csvs) == 0:
        csvs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*.csv")))

    print(f"Found {len(csvs)} CSV files to shard")
    for csv_path in csvs:
        csv_path = Path(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path} (error reading: {e})")
            continue

        # Skip if already labeled
        if all(c in df.columns for c in ["is_job", "is_traffic"]):
            print(f"Skipping {csv_path} (already labeled)")
            continue

        # Split into shards and save
        shards = shard_dataframe(df, args.num_shards)
        for r, shard_df in enumerate(shards):
            sp = csv_path.with_suffix(csv_path.suffix + f"{SHARD_PREFIX}{r}.csv")
            shard_df.to_csv(sp, index=False)
            print(f"  Wrote shard {r}: {sp} ({len(shard_df)} rows)")


if __name__ == "__main__":
    main()
