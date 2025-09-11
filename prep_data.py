from datasets import load_from_disk
from transformers import AutoTokenizer
from pathlib import Path


from data_preperation import preprocess_and_save
import argparse



def load_tokenizer(model_name, hf_token):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer






def parse_args():
    parser = argparse.ArgumentParser(description="Save Pretokenized Data")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    return parser.parse_args()



def main_cli():
    """SLURM / accelerate entrypoint."""

    

    args = parse_args()
    model_name = args.model_name  # replace with your model
    hf_token=args.hf_token

    train_df_load_path="data/train.csv"
    train_df_save_path=f"data/{model_name}/train_pretoken"

    valid_df_load_path="data/val.csv"
    valid_df_save_path=f"data/{model_name}/val_pretoken"

    test_df_load_path="data/test.csv"
    test_df_save_path=f"data/{model_name}/test_pretoken"

    tokenizer= load_tokenizer(model_name, hf_token)



    if Path(train_df_save_path).exists():
        print("train data set exists")
        train_ds=load_from_disk(train_df_save_path)
    else:
       print("train data set does not exists")
       train_ds = preprocess_and_save(train_df_load_path,tokenizer,train_df_save_path,text_column="prompt",target_column="target",is_train=True,max_length=512,num_proc=8)

    if Path(valid_df_save_path).exists():
       print("valid data set exists")
       val_ds=load_from_disk(valid_df_save_path)
    else:
       print("valid data set does not exist")
       val_ds=preprocess_and_save(valid_df_load_path,tokenizer,valid_df_save_path,text_column="prompt",target_column="target",is_train=False,max_length=512,num_proc=8)



    if Path(test_df_save_path).exists():
       print("test data set exists")
       test_ds=load_from_disk(test_df_save_path)
    else:
       print("test data set does not exist")
       test_ds=preprocess_and_save(test_df_load_path,tokenizer,test_df_save_path,text_column="prompt",target_column="target",is_train=False,max_length=512,num_proc=8)





if __name__ == "__main__":
    main_cli()
