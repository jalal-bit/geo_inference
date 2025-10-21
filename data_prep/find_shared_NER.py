import geopandas as gpd
import pandas as pd
import spacy
import json
import re
from shapely.geometry import Point
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
import sys
import argparse
import os

# === CONFIG ===





def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--county_folder", type=str, required=True,help="Directory to write state and county split folders")

    return p.parse_args()

def main_cli():
    """SLURM / accelerate entrypoint."""

    args = parse_args()

    data_folder="../data"
    raw_folder="raw"
    eda_folder=args.county_folder

    output_dir=os.path.join(data_folder,raw_folder,eda_folder)

    os.makedirs(output_dir, exist_ok=True)


    data_file="df_training.csv"
    
    raw_data_path=os.path.join(data_folder,raw_folder,data_file)
    train_df = pd.read_csv(raw_data_path)

    train_df_us=train_df[train_df['is_us']==1]
    train_df_us_unique=train_df_us.drop_duplicates(subset=['cleaned'], keep='first')

    county_counts = train_df_us_unique.groupby(['fips', 'county_name']).size().reset_index(name='count')

    # Filter to keep groups with count >= 20
    filtered_counties = county_counts[county_counts['count'] >= 25]

    # Merge the filtered counties back with the original dataframe to get the desired rows
    tweets = train_df_us_unique.merge(filtered_counties[['fips', 'county_name']], on=['fips', 'county_name'], how='inner')

   

    
    COUNTY_SHP = "tl_2024_us_county/tl_2024_us_county.shp"
    county_shp_path=raw_data_path=os.path.join(data_folder,raw_folder,COUNTY_SHP)

    OUTPUT_PATH = "shared_ner_neighbors_by_state.json"
    output_ner_path=raw_data_path=os.path.join(data_folder,raw_folder,OUTPUT_PATH)

    # === LOAD AND CLEAN DATA ===
    print("Loading tweets...")

    tweets["fips"] = tweets["fips"].astype(str).str.zfill(5)

    # --- FILTER JOB-RELATED TWEETS ---
    JOB_KEYWORDS = ["hiring", "job", "click", "apply", "career", "vacancy", "position", "recruit", "opportunity"]
    pattern = re.compile(r"\b(" + "|".join(JOB_KEYWORDS) + r")\b", flags=re.IGNORECASE)

    before = len(tweets)
    tweets = tweets[~tweets["cleaned"].str.contains(pattern, na=False)]
    after = len(tweets)
    print(f"Filtered out {before - after:,} job-related tweets ({(before - after) / before:.2%}).")

    # --- AGGREGATE TWEETS BY COUNTY ---
    print("Aggregating tweets by county...")
    county_texts = (
        tweets.groupby(["fips", "state"])["tweet"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )

    # --- LOAD COUNTY SHAPEFILE ---
    print("Loading county shapefile...")
    counties = gpd.read_file(county_shp_path)[["STATEFP", "COUNTYFP", "GEOID", "geometry", "NAME"]]
    counties["GEOID"] = counties["GEOID"].astype(str)

    merged = counties.merge(county_texts, left_on="GEOID", right_on="fips", how="inner")
    merged = merged.to_crs(epsg=5070)

    # --- COMPUTE DISTANCE FROM CENTER ---
    us_center = merged.unary_union.centroid
    merged["dist_to_center"] = merged.geometry.centroid.distance(us_center)
    merged = merged.sort_values("dist_to_center").reset_index(drop=True)

    # --- BUILD NEIGHBOR MAP ---
    print("Computing county adjacency...")
    neighbor_map = {}
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        geom = row.geometry
        neighbors = merged.loc[merged.geometry.touches(geom), "fips"].tolist()
        neighbor_map[row.fips] = neighbors

    # === LOAD SPACY MODEL ===
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.max_length = 10_000_000

    def extract_ners_with_freq(args):
        fips, text = args
        doc = nlp(text)
        ents = [ent.text.strip() for ent in doc.ents]
        freq = Counter(ents)
        return fips, freq

    # --- PARALLEL NER EXTRACTION ---
    print("Extracting NERs per county...")
    with Pool(cpu_count()) as pool:
        county_freqs = dict(
            tqdm(
                pool.imap(
                    extract_ners_with_freq,
                    merged[["fips", "tweet"]].itertuples(index=False, name=None),
                ),
                total=len(merged),
            )
        )

    # === MAIN COMPARISON LOOP ===
    print("Comparing neighboring counties across states...")
    results = []
    seen_pairs = set()

    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        fips = row.fips
        state_name = row.state
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

            n_state = neighbor_row["state"].values[0]
            if n_state == state_name:
                continue  # ✅ Skip same-state comparisons

            neighbor_freq = county_freqs.get(nfips)
            if not neighbor_freq:
                continue

            shared = set(freq_dict.keys()).intersection(neighbor_freq.keys())
            if not shared:
                continue

            shared_data = [
                {
                    "ner": ner,
                    "count_county": freq_dict[ner],
                    "count_neighbor": neighbor_freq[ner],
                }
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

    # === POST-PROCESSING: ADD REVERSE LINKS ===
    print("Postprocessing to include reverse neighbor data...")
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

    # === GROUP BY STATE AND COUNTY ===
    print("Grouping results by state and county...")
    state_map = defaultdict(lambda: defaultdict(list))

    for r in results:
        county_info = {
            "neighbor_fips": r["neighbor_fips"],
            "neighbor_name": r["neighbor_name"],
            "neighbor_state": r["neighbor_state"],
            "shared_ners": r["shared_ners"],
        }
        state_map[r["state_name"]][r["county_fips"]].append(county_info)

    # === SAVE JSON OUTPUT ===
    print(f"Saving output to {output_ner_path} ...")
    with open(output_ner_path, "w") as f:
        json.dump(state_map, f, indent=2)

    print(f"✅ Done. Grouped results for {len(state_map)} states saved to {output_ner_path}")

