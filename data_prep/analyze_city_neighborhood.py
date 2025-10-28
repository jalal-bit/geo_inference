import pandas as pd
import re
import argparse
from pathlib import Path

# === Job filtering pattern ===
JOB_KEYWORDS = ["hiring", "job", "click", "apply", "career", "vacancy", "position", "recruit", "opportunity"]
pattern = re.compile(r"\b(" + "|".join(JOB_KEYWORDS) + r")\b", flags=re.IGNORECASE)


def load_and_prepare(path):
    df = pd.read_csv(path)

    # Normalize blanks to NaN
    df['city_name'] = df['city_name'].replace("", pd.NA)
    df['neighborhood'] = df['neighborhood'].replace("", pd.NA)

    # Helper flags
    df["has_city"] = df['city_name'].notna()
    df["has_neigh"] = df['neighborhood'].notna()

    return df


def compute_stats(df):
    # === Group by state ===
    state_stats = df.groupby(['state_name']).agg(
        total=('cleaned', 'count'),
        count_with_city=('has_city', 'sum'),
        count_with_neigh=('has_neigh', 'sum'),
    ).reset_index()

    state_stats['pct_with_city'] = (state_stats['count_with_city'] / state_stats['total'] * 100).round(2)
    state_stats['pct_with_neigh'] = (state_stats['count_with_neigh'] / state_stats['total'] * 100).round(2)

    # === Group by state + county ===
    county_stats = df.groupby(['state_name', 'county_name', 'fips']).agg(
        total=('cleaned', 'count'),
        count_with_city=('has_city', 'sum'),
        count_with_neigh=('has_neigh', 'sum'),
    ).reset_index()

    county_stats['pct_with_city'] = (county_stats['count_with_city'] / county_stats['total'] * 100).round(2)
    county_stats['pct_with_neigh'] = (county_stats['count_with_neigh'] / county_stats['total'] * 100).round(2)

    return state_stats, county_stats


def filter_jobs(df):
    """Remove job-related tweets based on keywords."""
    return df[~df['cleaned'].str.contains(pattern, na=False)]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze city/neighborhood coverage in geocoded tweets.")
    parser.add_argument("input_file", help="Path to the US tweets CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file

    if not Path(input_file).exists():
        print(f"❌ Input file does not exist: {input_file}")
        exit(1)
    
    

    df = load_and_prepare(input_file)
    df=df.drop_duplicates(subset=['fips','cleaned'], keep='first')

    base = Path(input_file).stem  # filename without extension

    # Ordinary version
    state_stats, county_stats = compute_stats(df)
    state_stats.to_csv(f"analysis_state_{base}.csv", index=False)
    county_stats.to_csv(f"analysis_state_counties_{base}.csv", index=False)

    # Without job postings
    df_no_jobs = filter_jobs(df)
    state_stats_nj, county_stats_nj = compute_stats(df_no_jobs)
    state_stats_nj.to_csv(f"analysis_state_no_jobs_{base}.csv", index=False)
    county_stats_nj.to_csv(f"analysis_state_counties_no_jobs_{base}.csv", index=False)

    print("✅ Analysis files generated:")
    print(f"  - analysis_state_{base}.csv")
    print(f"  - analysis_state_counties_{base}.csv")
    print(f"  - analysis_state_no_jobs_{base}.csv")
    print(f"  - analysis_state_counties_no_jobs_{base}.csv")
