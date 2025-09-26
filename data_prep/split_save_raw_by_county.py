import pandas as pd
import os






def split_by_county_and_save(train_df_us_unique_filtered, output_dir):
    import os

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
        output_df = group_df[['cleaned', 'state_id', 'state_name', 'county_name', 'fips']]

        # Define the base filename
        base_filename = f"{state_name}_{county_name}".replace(" ", "_").replace("/", "_")

        # Split and save if file exceeds 500 rows
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

    print("Processing complete.")











def main_cli():
    """SLURM / accelerate entrypoint."""

    data_folder="../data"
    raw_folder="raw"
    eda_folder="eda2"

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
    train_df_us_unique_filtered = train_df_us_unique.merge(filtered_counties[['fips', 'county_name']], on=['fips', 'county_name'], how='inner')

    print("train df unique shape",train_df_us_unique_filtered.shape)

    # # Stratified split
    split_by_county_and_save(train_df_us_unique_filtered,output_dir)
    



if __name__ == "__main__":
    main_cli()




