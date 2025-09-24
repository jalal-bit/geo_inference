import pandas as pd
from sklearn.model_selection import train_test_split
import os
from prepare_raw_data_for_generative import format_target
from split_raw_stratified_data import stratified_split




def prepare_for_input_output(df):
    """
    Convert dataframe into instruction-input-output format for base LLMs.
    """
    prompts, targets = [], []

    for _, row in df.iterrows():
        text = row["cleaned"]
        target = format_target(row)

       

        prompts.append(text)
        targets.append(target)

    return pd.DataFrame({"prompt": prompts, "target": targets})



def sample_and_chunk_splits(df_train, df_val, df_test, output_dir="output", sample_size=100000, chunk_size=1000):
    """
    Sample 100k rows from each split (train/val/test), stratify by is_us,
    then chunk into 1000-row CSVs and save.
    """
   
    print("len df train",len(df_train))
    print("len df valid",len(df_val))
    print("len df test",len(df_test))
    

    def stratified_sample(df, strat_col, n, random_state=42):
        """Sample n rows stratified by a column."""
        frac = n / len(df)
        sampled = (
            df.groupby(strat_col, group_keys=False)
              .apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=random_state))
        )
        # Ensure exactly n
        return sampled.sample(n, random_state=random_state)

    # Apply stratified sampling
    df_train = stratified_sample(df_train, "is_us", sample_size)
    df_val   = stratified_sample(df_val, "is_us", sample_size)
    df_test  = stratified_sample(df_test, "is_us", sample_size)

    df_train=prepare_for_input_output(df_train)
    df_val=prepare_for_input_output(df_val)
    df_test=prepare_for_input_output(df_test)

    # Save chunks
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, start in enumerate(range(0, len(split_df), chunk_size)):
            chunk = split_df.iloc[start:start+chunk_size]
            chunk.to_csv(os.path.join(split_dir, f"{split_name}_{i+1:03d}.csv"), index=False)

    return df_train, df_val, 










def main_cli():
    """SLURM / accelerate entrypoint."""

    data_folder="../data"
    raw_folder="raw"
    eda_folder="eda"


    data_file="df_training.csv"
    
    raw_data_path=os.path.join(data_folder,raw_folder,data_file)
    final_df = pd.read_csv(raw_data_path)

    # # Stratified split
    df_train, df_valid , df_test = stratified_split(final_df)
    
    sample_and_chunk_splits(df_train, df_valid, df_test,output_dir=os.path.join(data_folder,raw_folder,eda_folder))



if __name__ == "__main__":
    main_cli()




