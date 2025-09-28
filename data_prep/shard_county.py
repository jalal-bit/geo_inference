import os
import math
import argparse
import pandas as pd

def shard_dataset(input_path, output_dir, world_size, file_prefix="shard"):
    # Load dataset (adjust if JSON/CSV/parquet/etc.)
    df = pd.read_csv(input_path)

    # Compute shard sizes
    total_rows = len(df)
    shard_size = math.ceil(total_rows / world_size)

    os.makedirs(output_dir, exist_ok=True)

    shards = []
    for rank in range(world_size):
        start_idx = rank * shard_size
        end_idx = min((rank + 1) * shard_size, total_rows)
        shard_df = df.iloc[start_idx:end_idx]

        shard_path = os.path.join(output_dir, f"{file_prefix}_{rank}.csv")
        shard_df.to_csv(shard_path, index=False)
        shards.append(shard_path)

        print(f"Shard {rank}: rows {start_idx}–{end_idx} → {shard_path}")

    return shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset (CSV)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save shards")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of ranks (GPUs)")
    parser.add_argument("--file_prefix", type=str, default="shard", help="Prefix for shard files")
    args = parser.parse_args()

    shard_dataset(args.input, args.output_dir, args.world_size, args.file_prefix)
