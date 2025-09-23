
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def stratified_split(df, test_size=0.2, val_size=0.1):
    """Stratify US by state_id, Non-US by is_us."""
    df_us = df[df["is_us"] == 1]
    df_non_us = df[df["is_us"] == 0]

    # US splits (stratify on state_id)
    df_us_trainval, df_us_test = train_test_split(
        df_us, test_size=test_size, stratify=df_us["state_id"], random_state=42
    )
    df_us_train, df_us_val = train_test_split(
        df_us_trainval,
        test_size=val_size / (1 - test_size),
        stratify=df_us_trainval["state_id"],
        random_state=42,
    )

    # Non-US splits (stratify on is_us)
    df_non_us_trainval, df_non_us_test = train_test_split(
        df_non_us, test_size=test_size, stratify=df_non_us["is_us"], random_state=42
    )
    df_non_us_train, df_non_us_val = train_test_split(
        df_non_us_trainval,
        test_size=val_size / (1 - test_size),
        stratify=df_non_us_trainval["is_us"],
        random_state=42,
    )

    # Combine and shuffle
    df_train = pd.concat([df_us_train, df_non_us_train]).sample(frac=1, random_state=42)
    df_val = pd.concat([df_us_val, df_non_us_val]).sample(frac=1, random_state=42)
    df_test = pd.concat([df_us_test, df_non_us_test]).sample(frac=1, random_state=42)

    return df_train, df_val, df_test

def main_cli():
    """SLURM / accelerate entrypoint."""

    data_folder="../data"
    raw_folder="raw"


    data_file="df_training.csv"

    raw_data_path=os.path.join(data_folder,raw_folder,data_file)
    final_df = pd.read_csv(raw_data_path)

    # # Stratified split
    df_train, df_valid , df_test = stratified_split(final_df)
    
    df_train_save_path=os.path.join(data_folder,raw_folder,"train.csv")
    df_valid_save_path=os.path.join(data_folder,raw_folder,"valid.csv")
    df_test_save_path=os.path.join(data_folder,raw_folder,"test.csv")
    
    
    df_train.to_csv(df_train_save_path)
    df_valid.to_csv(df_valid_save_path)
    #df_test.to_csv(df_test_save_path)



if __name__ == "__main__":
    main_cli()