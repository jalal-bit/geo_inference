import os
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
    parser.add_argument("--max_length", type=int, default=512, help="Max Length for Tokenization")
    return parser.parse_args()

def main_cli():
    """SLURM / accelerate entrypoint."""
    args = parse_args()

    model_name=args.model_name
    max_length=args.max_length
    data_folder="../data"
    


    train_input_path=f"{data_folder}/raw/train.csv"
    valid_input_path=f"{data_folder}/raw/valid.csv"

    train_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/train_pretoken"
    val_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/val_pretoken"
    

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("hf_token loaded")

    tokn=load_tokenizer(model_name,hf_token)
    print("train df path",train_df_save_path)
    print("valid df path",val_df_save_path)

    preprocess_and_save(train_input_path,tokn,train_df_save_path,text_column="prompt",target_column="target",is_train=True,max_length=max_length,num_proc=8)
    preprocess_and_save(valid_input_path,tokn,val_df_save_path,text_column="prompt",target_column="target",is_train=False,max_length=max_length,num_proc=8)


if __name__ == "__main__":
    main_cli()
