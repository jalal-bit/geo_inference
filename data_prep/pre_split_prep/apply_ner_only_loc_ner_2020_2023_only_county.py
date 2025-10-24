import pandas as pd
import spacy
import re
import os
import sys
from ftlangdetect import detect
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

# Load spaCy model
nlp = spacy.load("en_core_web_sm",disable=["parser", "tagger"])

# Load bad words from LDNOOBW
def load_bad_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}




def filter_english_fasttext_langdetect(df, text_col="tweet_text",min_score=0.65):
    """
    Filters a Pandas DataFrame for English tweets using fasttext-langdetect.

    Args:
        df (pd.DataFrame): Input DataFrame containing tweets.
        text_col (str): Column name containing text.
        min_score (float): Minimum confidence score to consider a tweet as English.

    Returns:
        pd.DataFrame: Filtered DataFrame with only English tweets.
    """
    def is_english(text):
        try:
            text = text.replace("\n"," ")
            result = detect(text)  # Returns a dictionary with 'lang' and 'score'
            return result["lang"] == "en" and result["score"] >= min_score
        except Exception as e:
            # Code to handle the exception
            print("An error occurred:", e)
            return False  # If detection fails, assume non-English

    df["is_english"] = df[text_col].astype(str).apply(is_english)

    return df[df["is_english"]].drop(columns=["is_english"])
# Clean a tweet
def clean_tweet(text, bad_words):
    if not isinstance(text, str): return ''
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    words = text.split()
    words = [word for word in words if word.lower() not in bad_words]
    return ' '.join(words).strip()

bad_words = load_bad_words("../../List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/en")
# === Load shapefiles ===
county_shp = "../../tl_2024_us_county/tl_2024_us_county.shp"
state_shp = "../../tl_2024_us_state/tl_2024_us_state.shp"

# Drop extra columns immediately
drop_county_cols = ['STATEFP','COUNTYFP','COUNTYNS','GEOIDFQ','NAMELSAD','LSAD','CLASSFP','MTFCC','CSAFP','CBSAFP','METDIVFP','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON']
drop_state_cols = ['STATENS','GEOID','GEOIDFQ','LSAD','MTFCC','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON']

counties = gpd.read_file(county_shp).to_crs("EPSG:4326").drop(columns=drop_county_cols)
states = gpd.read_file(state_shp).to_crs("EPSG:4326").drop(columns=drop_state_cols)



def process_texts(texts):
    """
    Process a batch of texts with spaCy using nlp.pipe() and return whether each has a location entity.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=1000):  # Efficient processing
        has_location = any(ent.label_ in ("GPE", "LOC", "FAC") for ent in doc.ents)
        results.append(has_location)
    return results
def process_large_csv(input_path, us_output_path, non_us_output_path, text_column="tweet_text", lat_col="geo_lat",long_col="geo_long" ,chunksize=100000):
    """
    Reads a large CSV file in chunks, applies NER using spaCy, and writes the results to a new CSV.
    """
    reader = pd.read_csv(input_path, chunksize=chunksize)

    first_us, first_non_us = True, True
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i+1}...")
        if text_column not in chunk.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV.")
        if lat_col not in chunk.columns or long_col not in chunk.columns:
            raise ValueError(f"Skipping {filename} — missing lat/long column")

        print(f" Initial rows: {len(chunk)}")
            # Drop rows with missing or invalid lat/lon
        chunk = chunk.dropna(subset=[lat_col, long_col])
        chunk = chunk[(chunk[lat_col] != 0) & (chunk[long_col] != 0)]
        print(f" After lat/lon check: {len(chunk)}")
        if chunk.empty:
            continue

        chunk=filter_english_fasttext_langdetect(chunk) # Keep only English tweets
        print(f" After language filtering: {len(chunk)}")
        if chunk.empty:
            continue



        chunk['cleaned'] = chunk[text_column].apply(lambda x: clean_tweet(x, bad_words))
        # Drop empty cleaned tweets
        chunk = chunk[chunk['cleaned'] != '']

        print(f" After cleaning: {len(chunk)}")
        if chunk.empty:
            continue

        mask = process_texts(chunk['cleaned'].astype(str))
        filtered_chunk = chunk[mask]

        print(f"After NLP filtering: {len(filtered_chunk)}")
        if filtered_chunk.empty:
            continue

        filtered_chunk=filtered_chunk.drop_duplicates(subset=['geo_lat', 'geo_long', 'cleaned'])
        # Create point geometry from lat/lon
        geometry = [Point(xy) for xy in zip(filtered_chunk[long_col], filtered_chunk[lat_col])]
        geo_df = gpd.GeoDataFrame(filtered_chunk, geometry=geometry, crs="EPSG:4326")
        # Spatial join: match to county
        geo_with_counties = gpd.sjoin(geo_df, counties, how="left", predicate="intersects")
        geo_with_counties['county_name'] = geo_with_counties['NAME']
        geo_with_counties['fips'] = geo_with_counties['GEOID']
        geo_with_counties = geo_with_counties.drop(columns=['index_right','NAME','GEOID'])

        # === Spatial join with states ===
        geo_with_states = gpd.sjoin(geo_with_counties, states, how="left", predicate="intersects")
        geo_with_states['state_name'] = geo_with_states['NAME']
        geo_with_states['state_abbr'] = geo_with_states['STUSPS']
        geo_with_states['state_id'] = geo_with_states['STATEFP']

        # Drop unneeded columns
        geo_with_states = geo_with_states.drop(columns=[
            'geometry', 'index_right', 'NAME', 'STUSPS','STATEFP'])


        # Separate US and non-US tweets
        us_tweets = geo_with_states.dropna(subset=["state_name"])
        non_us_tweets = geo_with_states[geo_with_states["state_name"].isna()]

        print(f"   US tweets: {len(us_tweets)}")
        print(f"Non-US tweets: {len(non_us_tweets)}")


        # Write US tweets
        if not us_tweets.empty:
            us_tweets.to_csv(us_output_path, index=False, header=first_us, mode="a")
            first_us = False
        



