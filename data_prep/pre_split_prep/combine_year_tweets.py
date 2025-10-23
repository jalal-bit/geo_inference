import os
import sys
import pandas as pd

def combine_tweets(root_folder, start_year, end_year):
    """
    Combines yearly US and Non-US tweet CSVs from subfolders into two master CSVs,
    selecting and ordering specific columns.
    """
    # Desired column order
    selected_cols = [
        "cleaned", "country", "country_code", "state_name", "state_id",
        "county_name", "fips", "city_name", "place_fips", "neighborhood",
        "name", "place_geoid", "REGION", "DIVISION", "state_abbr", "created_at", "geo_lat", "geo_long"
    ]

    us_tweets = []
    non_us_tweets = []

    for year in range(start_year, end_year + 1):
        folder_path = os.path.join(root_folder, f"tweets_{year}")

        if not os.path.isdir(folder_path):
            print(f"⚠️ Folder not found: {folder_path}, skipping...")
            continue

        us_file = os.path.join(folder_path, f"us_tweets_combined_{year}.csv")
        non_us_file = os.path.join(folder_path, f"non_us_tweets_combined_{year}.csv")

        # Read and filter each file if exists
        if os.path.exists(us_file):
            print(f"  Reading {us_file}")
            df_us = pd.read_csv(us_file,engine="python", on_bad_lines='skip')
            df_us = df_us[[c for c in selected_cols if c in df_us.columns]]
            us_tweets.append(df_us)
        else:
            print(f"⚠️ Missing: {us_file}")

        if os.path.exists(non_us_file):
            print(f"  Reading {non_us_file}")
            df_non_us = pd.read_csv(non_us_file, engine="python", on_bad_lines='skip')
            df_non_us = df_non_us[[c for c in selected_cols if c in df_non_us.columns]]
            non_us_tweets.append(df_non_us)
        else:
            print(f"⚠️ Missing: {non_us_file}")

    # Combine and save directly in current folder
    if us_tweets:
        us_combined = pd.concat(us_tweets, ignore_index=True)
        # Ensure consistent columns and order
        for col in selected_cols:
            if col not in us_combined.columns:
                us_combined[col] = None
        us_combined = us_combined[selected_cols]
        us_combined.to_csv(f"all_us_tweets_combined_{start_year}_{end_year}.csv", index=False)
        print("  Combined US tweets saved as all_us_tweets_combined.csv")
    else:
        print("  No US tweet files found.")


    if non_us_tweets:
        non_us_combined = pd.concat(non_us_tweets, ignore_index=True)
        for col in selected_cols:
            if col not in non_us_combined.columns:
                non_us_combined[col] = None
        non_us_combined = non_us_combined[selected_cols]
        non_us_combined.to_csv(f"all_non_us_tweets_combined_{start_year}_{end_year}.csv", index=False)
        print("  Combined Non-US tweets saved as all_non_us_tweets_combined.csv")
    else:
        print("  No Non-US tweet files found.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_tweets.py <start_year> <end_year>")
        sys.exit(1)

    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])

    root_folder = os.getcwd()  # use current working directory
    combine_tweets(root_folder, start_year, end_year)


