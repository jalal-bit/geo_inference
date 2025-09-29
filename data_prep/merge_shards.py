import os
import glob
import argparse
from pathlib import Path
import pandas as pd

PRED_SUFFIX = ".preds.csv"


def merge_shards( args):

    # collect base CSVs (without shard suffix)
    csvs = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True))
    csvs = [c for c in csvs if ".shard_" not in c and not c.endswith(PRED_SUFFIX)]
    print("csvs", csvs)
    print(f"[main] merging predictions for {len(csvs)} base CSVs")

    for csv_path in csvs:
        csv_path = Path(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[main] failed reading {csv_path}: {e}")
            continue

        base_no_ext = os.path.splitext(str(csv_path))[0]
        shard_paths = sorted(glob.glob(f"{base_no_ext}.shard_*.csv"))
        pred_paths = [Path(sp + PRED_SUFFIX) for sp in shard_paths]

        preds_list = []
        for sp, pp in zip(shard_paths, pred_paths):
            if pp.exists():
                pf = pd.read_csv(pp)
            else:
                shard_len = len(pd.read_csv(sp))
                pf = pd.DataFrame([{"is_job": 0, "is_traffic": 0}] * shard_len)
            preds_list.append(pf)

        merged_preds = pd.concat(preds_list, ignore_index=True)
        df_out = df.copy()
        df_out["is_job"] = merged_preds["is_job"].astype(int)
        df_out["is_traffic"] = merged_preds["is_traffic"].astype(int)

        backup = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup.exists():
            csv_path.rename(backup)
        df_out.to_csv(csv_path, index=False)
        print(f"[main] wrote labeled CSV {csv_path} (backup at {backup})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../data/raw/eda_test/geo_inference_output")

    return p.parse_args()


def main():
    args = parse_args()

    # Phase 1: worker labeling
    merge_shards(args)


if __name__ == "__main__":
    main()