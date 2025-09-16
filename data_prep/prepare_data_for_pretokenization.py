import pandas as pd
import os,sys
from pre_tokenize_data import preprocess_and_save
from transformers import AutoTokenizer
from dotenv import load_dotenv
import argparse


def load_tokenizer(model_name,hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pre-trained and Finetuned Models")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--shot_type", type=str, choices=["few_shot", "one_shot", "zero_shot"], default=None, help="Specify the shot type: few_shot, one_shot, or zero_shot.")
    parser.add_argument("--run_num", type=int, default=0, help="Run number (non-negative integer, default is 0).")
    parser.add_argument("--max_length", type=int, default=512, help="Max Length for Tokenization")
    return parser.parse_args()

def main_cli():
    """SLURM / accelerate entrypoint."""
    args = parse_args()
    run_num = args.run_num  # replace with your model

    shot_type=args.shot_type
    model_name=args.model_name
    max_length=args.max_length
    data_folder="../data"
    

    if shot_type:
        if shot_type=='few_shot' or shot_type=='one_shot':
            run_num=args.run_num
            if run_num>=0:
                input_path=f"{data_folder}/raw/{shot_type}/run_{run_num}/test_{shot_type}.csv"
                test_df_save_path=f"{data_folder}/{model_name}/pretokenized/{shot_type}/run_{run_num}/test_pretoken"

            else:
                print("Error: --run_num must be a non-negative integer.")
                sys.exit(1)
        elif shot_type=='zero_shot':
            input_path=f"{data_folder}/raw/test.csv"
            test_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/test_pretoken"
            
    else:
        input_path=f"{data_folder}/raw/test.csv"
        test_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/test_pretoken"
    

    print("model name",model_name)
    print("shot type",shot_type)
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("hf_token loaded")

    tokn=load_tokenizer(model_name,hf_token)
    print(test_df_save_path)

    preprocess_and_save(input_path,tokn,test_df_save_path,text_column="prompt",target_column="target",is_train=False,max_length=max_length,num_proc=8)


if __name__ == "__main__":
    main_cli()
