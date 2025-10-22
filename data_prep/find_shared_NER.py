import geopandas as gpd
import pandas as pd
import spacy
import json
import re
from shapely.geometry import Point
from tqdm import tqdm
from collections import Counter, defaultdict
import sys
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--county_folder",
        type=str,
        required=True,
        help="Directory to write state and county split folders"
    )
    return parser.parse_args()

def main_cli():
    args = parse_args()

    data_folder = "../data"
    raw_folder = "raw"
    eda_folder = args.county_folder
    output_dir = os.path.join(data_folder, raw_folder, eda_folder)
    os.makedirs(output_dir, exist_ok=True)

    data_file = "df_training.csv"
    raw_data_path = os.path.join(data_folder, raw_folder, data_file)
    train_df = pd.read_csv(raw_data_path)
    print(f"[INFO] Loaded {len(train_df):,} total rows from {data_file}")

    # Filter US tweets and drop duplicates
    train_df_us = train_df[train_df['is_us'] == 1]
    train_df_us_unique = train_df_us.drop_duplicates(subset=['cleaned'], keep='first')
    print(f"[INFO] US tweets after deduplication: {len(train_df_us_unique):,}")

    # Keep only counties with enough tweets
    county_counts = train_df_us_unique.groupby(['fips', 'county_name']).size().reset_index(name='count')
    filtered_counties = county_counts[county_counts['count'] >= 25]
    tweets = train_df_us_unique.merge(filtered_counties[['fips', 'county_name']], on=['fips', 'county_name'], how='inner')
    print(f"[INFO] Counties with >=25 tweets: {tweets['fips'].nunique():,}")

    # Paths
    COUNTY_SHP = "tl_2024_us_county/tl_2024_us_county.shp"
    county_shp_path = os.path.join(data_folder, raw_folder, COUNTY_SHP)
    OUTPUT_PATH = "shared_ner_neighbors_by_state.json"
    output_ner_path = os.path.join(data_folder, raw_folder, OUTPUT_PATH)

    # --- CLEAN DATA ---
    tweets["fips"] = tweets["fips"].astype(str).str.zfill(5)

    JOB_KEYWORDS = ["hiring", "job", "click", "apply", "career", "vacancy", "position", "recruit", "opportunity"]
    pattern = re.compile(r"\b(" + "|".join(JOB_KEYWORDS) + r")\b", flags=re.IGNORECASE)

    before = len(tweets)
    tweets = tweets[~tweets["cleaned"].str.contains(pattern, na=False)]
    after = len(tweets)
    print(f"[INFO] Filtered {before-after:,} job-related tweets ({(before-after)/before:.2%})")
    print(f"[INFO] Tweets remaining: {after:,}")

    # Aggregate tweets by county
    print("[INFO] Aggregating tweets by county...")
    county_texts = (
        tweets.groupby(["fips", "state_name"])["cleaned"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    print(f"[INFO] Aggregated {len(county_texts):,} counties with text")

    # Load county shapefile
    print("[INFO] Loading county shapefile...")
    counties = gpd.read_file(county_shp_path)[["STATEFP", "COUNTYFP", "GEOID", "geometry", "NAME"]]
    counties["GEOID"] = counties["GEOID"].astype(str)
    merged = counties.merge(county_texts, left_on="GEOID", right_on="fips", how="inner")
    merged = merged.to_crs(epsg=5070)
    print(f"[INFO] Counties merged with tweets: {len(merged):,}")

    print("[DEBUG] Checking FIPS formatting before merge...")
    print("Tweet FIPS samples:", tweets["fips"].astype(str).head(10).tolist())
    print("County GEOID samples:", counties["GEOID"].astype(str).head(10).tolist())
    print(f"Unique tweet FIPS count: {tweets['fips'].nunique()}")
    print(f"Unique GEOID count: {counties['GEOID'].nunique()}")


    # Compute distance from US center
    us_center = merged.unary_union.centroid
    merged["dist_to_center"] = merged.geometry.centroid.distance(us_center)
    merged = merged.sort_values("dist_to_center").reset_index(drop=True)

    # Build neighbor map
    print("[INFO] Computing county adjacency...")
    neighbor_map = {}
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        geom = row.geometry
        neighbors = merged.loc[merged.geometry.touches(geom), "fips"].tolist()
        neighbor_map[row.fips] = neighbors

    # Load spaCy model
    print("[INFO] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.max_length = 10_000_000

    # Extract NERs sequentially
    print("[INFO] Extracting NERs per county...")
    county_freqs = {}
    for fips, text in tqdm(merged[["fips", "cleaned"]].itertuples(index=False, name=None), total=len(merged)):
        doc = nlp(text)
        ents = [ent.text.strip() for ent in doc.ents]
        freq = Counter(ents)
        county_freqs[fips] = freq
        print(f"[DEBUG] County {fips} NERs extracted: {len(freq)} entities")

    # Compare neighbors
    print("[INFO] Comparing neighboring counties across states...")
    results = []
    seen_pairs = set()
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        fips = row.fips
        state_name = row.state_name
        freq_dict = county_freqs.get(fips)
        if not freq_dict:
            continue

        for nfips in neighbor_map.get(fips, []):
            pair = tuple(sorted((fips, nfips)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            neighbor_row = merged.loc[merged.fips == nfips]
            if neighbor_row.empty:
                continue

            n_state = neighbor_row["state_name"].values[0]
            if n_state == state_name:
                continue

            neighbor_freq = county_freqs.get(nfips)
            if not neighbor_freq:
                continue

            shared = set(freq_dict.keys()).intersection(neighbor_freq.keys())
            if not shared:
                continue

            shared_data = [
                {"ner": ner, "count_county": freq_dict[ner], "count_neighbor": neighbor_freq[ner]}
                for ner in shared
            ]

            results.append({
                "state_name": state_name,
                "county_fips": fips,
                "county_name": row["NAME"],
                "neighbor_state": n_state,
                "neighbor_fips": nfips,
                "neighbor_name": neighbor_row["NAME"].values[0],
                "shared_ners": shared_data
            })
            print(f"[DEBUG] Found {len(shared_data)} shared NERs between {fips} and {nfips}")

    # Postprocessing: add reverse links
    print("[INFO] Adding reverse neighbor entries...")
    reverse_entries = []
    for r in results:
        reverse_entries.append({
            "state_name": r["neighbor_state"],
            "county_fips": r["neighbor_fips"],
            "county_name": r["neighbor_name"],
            "neighbor_state": r["state_name"],
            "neighbor_fips": r["county_fips"],
            "neighbor_name": r["county_name"],
            "shared_ners": r["shared_ners"]
        })
    results.extend(reverse_entries)

    # Group by state and county
    print("[INFO] Grouping results by state and county...")
    state_map = defaultdict(lambda: defaultdict(list))
    for r in results:
        county_info = {
            "neighbor_fips": r["neighbor_fips"],
            "neighbor_name": r["neighbor_name"],
            "neighbor_state": r["neighbor_state"],
            "shared_ners": r["shared_ners"],
        }
        state_map[r["state_name"]][r["county_fips"]].append(county_info)

    # Save JSON output
    print(f"[INFO] Saving output to {output_ner_path} ...")
    with open(output_ner_path, "w") as f:
        json.dump(state_map, f, indent=2)

    print(f"[INFO] Done. Grouped results for {len(state_map)} states saved to {output_ner_path}")


if __name__ == "__main__":
    main_cli()
