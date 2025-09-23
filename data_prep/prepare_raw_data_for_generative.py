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


def format_target(row):
    """Return formatted location string for US or Non-US rows."""
    if row["is_us"] == 0:
        return "country: Non-US"

    # Prefer names if available, else fallback to IDs
    state_name = (
        str(row["state_name"])
        if "state_name" in row and pd.notna(row["state_name"])
        else str(row.get("state_id", "Unknown"))
    )
    county_name = (
        str(row["county_name"])
        if "county_name" in row and pd.notna(row["county_name"])
        else str(row.get("fips", "Unknown"))
    )
    return f"country: US | state: {state_name} | county: {county_name}"



def prepare_for_generative_original(df):
    """
    Convert dataframe into instruction-input-output format for base LLMs.
    """
    prompts, targets = [], []

    for _, row in df.iterrows():
        text = row["cleaned"]
        target = format_target(row)

        prompt = (
            "Instruction: Determine the country, state, and county from the following text.\n"
            f"Input: \"{text}\"\n"
            "Output:"
        )

        prompts.append(prompt)
        targets.append(target)

    return pd.DataFrame({"prompt": prompts, "target": targets})





def prepare_for_generative_with_one_shot_prompt(df,us_example,non_us_example):
    """
    Convert dataframe into instruction-input-output format for base LLMs.
    """
    prompts, targets = [], []

    for _, row in df.iterrows():
        text = row["cleaned"]
        target = format_target(row)
        is_us=row["is_us"]
        if is_us:
            prompt=f"{us_example}\nNow: Determine the country, state, and county from the following text.\n Input: \"{text}\"\n Output:"
        else:
            prompt=f"{non_us_example}\nNow: Determine only the country from the following text.\n Input: \"{text}\"\n Output:"
                

        prompts.append(prompt)
        targets.append(target)

    return pd.DataFrame({"prompt": prompts, "target": targets})



def prepare_for_generative_with_few_shot_prompt(df,examples):
    """
    Convert dataframe into instruction-input-output format for base LLMs.
    """
    prompts, targets = [], []

    for _, row in df.iterrows():
        text = row["cleaned"]
        target = format_target(row)
        is_us=row["is_us"]
        if examples:
            if is_us:
                prompt=f"{examples}\nNow: Determine the country, state, and county from the following text.\n Input: \"{text}\"\n Output:"
            else:
                prompt=f"{examples}\nNow: Determine only the country from the following text.\n Input: \"{text}\"\n Output:"
                

        prompts.append(prompt)
        targets.append(target)

    return pd.DataFrame({"prompt": prompts, "target": targets})



