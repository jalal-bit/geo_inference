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
    #words = [word for word in words if word.lower() not in bad_words]
    return ' '.join(words).strip()

bad_words = load_bad_words("../../List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/en")
# === Load shapefiles ===
county_shp = "../../tl_2024_us_county/tl_2024_us_county.shp"
state_shp = "../../tl_2024_us_state/tl_2024_us_state.shp"
place_shp = "../../us_places_2024/us_places_2024/us_places_2024_merged.shp"  

# Drop extra columns immediately
drop_county_cols = ['STATEFP','COUNTYFP','COUNTYNS','GEOIDFQ','NAMELSAD','LSAD','CLASSFP','MTFCC','CSAFP','CBSAFP','METDIVFP','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON']
drop_state_cols = ['STATENS','GEOID','GEOIDFQ','LSAD','MTFCC','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON']
drop_place_cols = ['STATEFP','PLACENS','GEOIDFQ','NAMELSAD','LSAD','CLASSFP','PCICBSA','MTFCC','FUNCSTAT','ALAND','AWATER','INTPTLAT','INTPTLON','STATE_ABBR']

counties = gpd.read_file(county_shp).to_crs("EPSG:4326").drop(columns=drop_county_cols)
states = gpd.read_file(state_shp).to_crs("EPSG:4326").drop(columns=drop_state_cols)
places = gpd.read_file(place_shp).to_crs("EPSG:4326").drop(columns=drop_place_cols) 






def process_texts(texts):
    """
    Process a batch of texts with spaCy using nlp.pipe() and return entities excluding GPE, LOC, and FAC.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=1000):  # Efficient processing
        ents = [ent.text for ent in doc.ents]
        results.append(ents)
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

        entities = process_texts(chunk['cleaned'].astype(str))
        mask = [len(e) > 0 for e in entities]  # True if tweet has at least one non-location entity
        filtered_chunk = chunk[mask]
        chunk["non_location_ents"] = entities  # optional: save them

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
             'index_right', 'NAME', 'STUSPS','STATEFP'])

        # === Spatial join: Places last (intersects) ===
        geo_with_places = gpd.sjoin(geo_with_states, places, how="left", predicate="intersects")
        geo_with_places['city_name'] = geo_with_places['NAME']
        geo_with_places['place_fips'] = geo_with_places['PLACEFP']
        geo_with_places['place_geoid'] = geo_with_places['GEOID']
        geo_with_places = geo_with_places.drop(columns=['index_right','NAME','GEOID','geometry'])



                # === Handle neighborhoods ===
        def extract_neighborhood(row):
            if pd.notna(row.get('place_type')) and row['place_type'].lower() == 'neighborhood':
                return row.get('name', None)
            return None

        geo_with_places['neighborhood'] = geo_with_places.apply(extract_neighborhood, axis=1)


        
        # Separate US and non-US tweets
        us_tweets = geo_with_places.dropna(subset=["state_name"])
        non_us_tweets = geo_with_places[geo_with_places["state_name"].isna()]

        print(f"🇺🇸 US tweets: {len(us_tweets)}")
        print(f"Non-US tweets: {len(non_us_tweets)}")


        # Write US tweets
        if not us_tweets.empty:
            us_tweets.to_csv(us_output_path, index=False, header=first_us, mode="a")
            first_us = False

        # Write Non-US tweets
        if not non_us_tweets.empty:
            non_us_tweets.to_csv(non_us_output_path, index=False, header=first_non_us, mode="a")
            first_non_us = False



if __name__ == "__main__":
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
       print(f"The input file doesn not exist {input_file}")
       sys.exit(1)
    
    name_without_ext = Path(input_file).stem
    us_output_path = f"us_tweets_NER_{name_without_ext}.csv"
    non_us_output_path=f"non_us_tweets_NER_{name_without_ext}.csv"

    print(f"Processing: {input_file}")

    process_large_csv(input_file, us_output_path=us_output_path, non_us_output_path=non_us_output_path, text_column="tweet_text", chunksize=100000)
    # print(f"Saved filtered data to {output_file}")




