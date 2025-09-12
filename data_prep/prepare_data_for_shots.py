import pandas as pd
import os
from prepare_data_for_generative import stratified_split, prepare_for_generative_with_one_shot_prompt, prepare_for_generative_with_few_shot_prompt


def format_target(row):
    """Return formatted location string for US or Non-US rows."""
    if row["is_us"] == 0:
        return f"country: {row['country']}"

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


data_folder="../data"
raw_folder="raw"
one_shot_folder="one_shot"
few_shot_folder="few_shot"

data_file="df_training.csv"

raw_data_path=os.path.join(data_folder,raw_folder,data_file)
final_df = pd.read_csv(raw_data_path)

# # Stratified split
df_train, df_valid , df_test = stratified_split(final_df)





train_us = df_train[df_train['is_us'] == True].sample(n=50, random_state=42)
train_non_us = df_train[df_train['is_us'] == False].sample(n=50, random_state=42)
df_train_sampled = pd.concat([train_us, train_non_us]).reset_index(drop=True)

# Sample from df_valid
valid_us = df_valid[df_valid['is_us'] == True].sample(n=50, random_state=42)
valid_non_us = df_valid[df_valid['is_us'] == False].sample(n=50, random_state=42)
df_valid_sampled = pd.concat([valid_us, valid_non_us]).reset_index(drop=True)


train_us = df_train[df_train['is_us'] == True].sample(n=50, random_state=42)
train_non_us = df_train[df_train['is_us'] == False].sample(n=50, random_state=42)
df_train_sampled = pd.concat([train_us, train_non_us]).reset_index(drop=True)

# Sample from df_valid
valid_us = df_valid[df_valid['is_us'] == True].sample(n=50, random_state=42)
valid_non_us = df_valid[df_valid['is_us'] == False].sample(n=50, random_state=42)
df_valid_sampled = pd.concat([valid_us, valid_non_us]).reset_index(drop=True)


df_train_sampled["label"]=df_train_sampled.apply(lambda x: format_target(x), axis=1 )
df_valid_sampled["label"]=df_valid_sampled.apply(lambda x: format_target(x), axis=1 )

df_train_sampled=df_train_sampled[['cleaned','label','is_us']]
df_valid_sampled=df_valid_sampled[['cleaned','label','is_us']]

final_concat_data = pd.concat([df_train_sampled[df_train_sampled['is_us']==1], df_train_sampled[df_train_sampled['is_us']==0],df_valid_sampled[df_valid_sampled['is_us']==1],df_valid_sampled[df_valid_sampled['is_us']==0]], axis=0,ignore_index=True)


examples_one_shot_us_template = (
    "Instruction: Determine the country, state, and county from the following text.\n"
    "Example:\n"
    "Input: \"{0}\"\n"
    "Output: {1}\n\n"
    "Now, complete the task for the following input."
)

examples_one_shot_nonus_template = (
    "Instruction: Determine only the country from the following text.\n"
    "Example:\n"
    "Input: \"{0}\"\n"
    "Output: {1}\n\n"
    "Now, complete the task for the following input."
)

examples_few_shot_mixed = (
    "Instruction: Determine the country, state, and county (if US) or just the country (if non-US) from the following text.\n"
    "Examples:\n"
    "Input: \"{0}\"\n"
    "Output: {1}\n\n"
    "Input: \"{2}\"\n"
    "Output: {3}\n\n"
    "Input: \"{4}\"\n"
    "Output: {5}\n\n"
    "Now, complete the task for the following input."
)

selected_one_shot_pair=[(0,55),(2,57),(4,61),(7,71),(20,94),(145,153),(142,157),(132,171),(104,199),(137,175)]

selected_few_shot_pair=[(0,55,146),(2,57,144),(4,61,139),(7,71,115),(20,94,149),(145,153,21),(142,157,45),(132,171,49),(104,199,40),(137,175,14)]


one_shot_dict={}
for i,p in enumerate(selected_one_shot_pair):
    t_us_idx= p[0]
    t_non_us_idx=p[1]
    us_cleand_temp=final_concat_data.iloc[t_us_idx]['cleaned']
    us_label_temp=final_concat_data.iloc[t_us_idx]['label']

    no_us_cleaned_temp=final_concat_data.iloc[t_non_us_idx]['cleaned']
    no_us_label_temp=final_concat_data.iloc[t_non_us_idx]['label']

    us_example_final=examples_one_shot_us_template.format(us_cleand_temp,us_label_temp)
    non_us_example_final=examples_one_shot_nonus_template.format(no_us_cleaned_temp,no_us_label_temp)
    one_shot_dict[i]=[us_example_final,non_us_example_final]


few_shot_dict={}
for i,p in enumerate(selected_few_shot_pair):
    t_us_idx= p[0]
    t_non_us_idx=p[1]
    t_us_idx_2= p[2]
    
    us_cleand_temp=final_concat_data.iloc[t_us_idx]['cleaned']
    us_label_temp=final_concat_data.iloc[t_us_idx]['label']

    no_us_cleaned_temp=final_concat_data.iloc[t_non_us_idx]['cleaned']
    no_us_label_temp=final_concat_data.iloc[t_non_us_idx]['label']

    us_cleand_temp2=final_concat_data.iloc[t_us_idx_2]['cleaned']
    us_label_temp2=final_concat_data.iloc[t_us_idx_2]['label']

    few_shot_dict[i]=examples_few_shot_mixed.format(us_cleand_temp,us_label_temp,no_us_cleaned_temp,no_us_label_temp,us_cleand_temp2,us_label_temp2)





run_num=0
run_name=f"run_{run_num}"
one_shot_save_folder=os.path.join(data_folder,raw_folder,one_shot_folder,run_name)
few_shot_save_folder=os.path.join(data_folder,raw_folder,few_shot_folder,run_name)
us_prompt=one_shot_dict[run_num][0]
non_us_prompt=one_shot_dict[run_num][1]
few_shot_prompt=few_shot_dict[run_num]
print(us_prompt)
print(non_us_prompt)
print(few_shot_prompt)

test_generative_one_shot = prepare_for_generative_with_one_shot_prompt(df_test,us_prompt,non_us_prompt)
test_generative_few_shot = prepare_for_generative_with_few_shot_prompt(df_test,few_shot_prompt)


os.makedirs(one_shot_save_folder, exist_ok=True)
os.makedirs(few_shot_save_folder, exist_ok=True)

one_shot_final_path_csv=os.path.join(one_shot_save_folder,"test_one_shot.csv")
few_shot_final_path_csv=os.path.join(few_shot_save_folder,"test_few_shot.csv")
test_generative_one_shot.to_csv(one_shot_final_path_csv, index=False)
test_generative_few_shot.to_csv(few_shot_final_path_csv, index=False)

one_shot_prompt_together=us_prompt+non_us_prompt
with open(os.path.join(one_shot_save_folder,"one_shot_prompt.txt"),'w') as f:
    f.write(one_shot_prompt_together)
with open(os.path.join(few_shot_save_folder,"few_shot_prompt.txt"),'w') as f:
    f.write(few_shot_prompt)