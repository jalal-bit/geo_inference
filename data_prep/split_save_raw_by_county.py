import pandas as pd
import os

import argparse







def split_by_county_and_save(train_df_us_unique_filtered, output_dir,should_split=True,all_columns=False):


    print("Before grouping:")
    print("Total rows:", len(train_df_us_unique_filtered))
    print("Unique states:", train_df_us_unique_filtered['state_name'].nunique())
    print("Unique states values:", train_df_us_unique_filtered['state_name'].unique())
    print("Unique FIPS", train_df_us_unique_filtered['fips'].nunique())
    print("Unique Stated IDS", train_df_us_unique_filtered['state_id'].nunique())
    print("Unique Stated IDS values", train_df_us_unique_filtered['state_id'].unique())
    print("Unique Stated missing states ID", train_df_us_unique_filtered['state_id'].isna().sum())
    print("Missing state_name:", train_df_us_unique_filtered['state_name'].isna().sum())
    print("top Unique Stated missing states names", train_df_us_unique_filtered[train_df_us_unique_filtered['state_name'].isna()].head(1))

    print("top Unique Stated missing states ID", train_df_us_unique_filtered[train_df_us_unique_filtered['state_id'].isna()].head(10))


    print("Missing FIPS:", train_df_us_unique_filtered['fips'].isna().sum())
    print("Missing county_name:", train_df_us_unique_filtered['county_name'].isna().sum())

    print("\nTop 10 FIPS counts:")
    print(train_df_us_unique_filtered['fips'].value_counts().head(10))

    # Group by state and county
    grouped = train_df_us_unique_filtered.groupby(['state_id', 'state_name', 'fips', 'county_name'])

    # Define the output directory
    output_base_dir = os.path.join(output_dir, 'geo_inference_output')

    # Iterate through each group
    for (state_id, state_name, fips, county_name), group_df in grouped:
        # Create state directory
        state_dir = os.path.join(output_base_dir, state_name)
        os.makedirs(state_dir, exist_ok=True)

        # Create county directory
        # Sanitize county_name for directory creation
        county_dir_name = f"{county_name}_{int(fips)}"
        county_dir = os.path.join(state_dir, county_dir_name)
        os.makedirs(county_dir, exist_ok=True)

        # Select required columns

        if all_columns:
            output_df=group_df
        else:
            output_df = group_df[['cleaned', 'state_id', 'state_name', 'county_name', 'fips']]

        # Define the base filename
        base_filename = f"{state_name}_{county_name}".replace(" ", "_").replace("/", "_")

        # Split and save if file exceeds 500 rows
        if should_split:
            if len(output_df) > 500:
                num_splits = (len(output_df) + 499) // 500  # Calculate number of splits
                for i in range(num_splits):
                    split_df = output_df.iloc[i * 500:(i + 1) * 500]
                    output_filename = os.path.join(county_dir, f"{base_filename}_split{i+1}.csv")
                    split_df.to_csv(output_filename, index=False)
                    print(f"Saved {output_filename}")
            else:
                output_filename = os.path.join(county_dir, f"{base_filename}.csv")
                output_df.to_csv(output_filename, index=False)
                print(f"Saved {output_filename}")
        else:
                output_filename = os.path.join(county_dir, f"{base_filename}.csv")
                output_df.to_csv(output_filename, index=False)
                print(f"Saved {output_filename}")


    print("Processing complete.")




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--county_folder", type=str, required=True,help="Directory to write state and county split folders")
    p.add_argument("--input_data", type=str, required=True,help="Directory to read data from")
    p.add_argument('--not_split_over_500', action='store_false', help='Enable verbose output.')
    p.add_argument('--keep_all_columns', action='store_true', help='Keep all columns.')


    

    return p.parse_args()






def main_cli():
    """SLURM / accelerate entrypoint."""

    args = parse_args()

    data_folder="../data"
    raw_folder="raw"
    eda_folder=args.county_folder
    should_split=args.not_split_over_500
    keep_all_columns=args.keep_all_columns

    output_dir=os.path.join(data_folder,raw_folder,eda_folder)

    os.makedirs(output_dir, exist_ok=True)


    data_file=args.input_data
    
    raw_data_path=os.path.join(data_folder,raw_folder,data_file)
    train_df = pd.read_csv(raw_data_path)


    print(f"input data: {data_file} --- county_output {eda_folder} --- should_split {should_split} --- keep_all_columns {keep_all_columns} ")

    
    if 'is_us' in train_df.columns:
        train_df_us=train_df[train_df['is_us']==1]
    else:
        train_df_us=train_df

    train_df_us_unique=train_df_us.drop_duplicates(subset=['fips','cleaned'], keep='first')

    county_counts = train_df_us_unique.groupby(['fips', 'county_name']).size().reset_index(name='count')

    # Filter to keep groups with count >= 25
    filtered_counties = county_counts[county_counts['count'] >= 25]

    # Merge the filtered counties back with the original dataframe to get the desired rows
    train_df_us_unique_filtered = train_df_us_unique.merge(filtered_counties[['fips', 'county_name']], on=['fips', 'county_name'], how='inner')

    print("train df unique shape",train_df_us_unique_filtered.shape)

    # # Stratified split
    split_by_county_and_save(train_df_us_unique_filtered,output_dir,should_split,keep_all_columns)
    



if __name__ == "__main__":
    main_cli()




