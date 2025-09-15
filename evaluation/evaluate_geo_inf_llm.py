
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from pathlib import Path
import wandb
import argparse
from datasets import load_from_disk

from datetime import datetime
import os,json
from dotenv import load_dotenv
import sys


from collators import DataCollatorForEval
from metrics import compute_validation_metrics
from load import load_model_else, load_model_gemma

def evaluate(model, val_loader, tokenizer, accelerator,max_new_tokens=60, log_first_batch=False):
    """
    Multi-GPU safe:
    - generate locally
    - slice continuations using left-padding prompt lengths
    - gather lists of strings with gather_object
    - compute metrics on main process only
    """
    model.eval()
    gen_cfg = {"num_beams": 1, "do_sample": False}
    gen_cfg["max_new_tokens"]=max_new_tokens
    unwrapped_model = accelerator.unwrap_model(model)

    # We'll collect everything on main at the end
    all_pred_texts = []
    all_gold_texts = []
    total_steps_per_epoch = len(val_loader)
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)

            gen_out = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg
            )

   
            cutoff = attention_mask.size(1)  # fixed length of input sequence
            local_pred_texts = [
                tokenizer.decode(gen_out[i, cutoff:], skip_special_tokens=True)
                for i in range(gen_out.size(0))
            ]

            local_gold_texts = batch["targets"]

            # Gather lists of strings across processes
            gathered_preds = gather_object(local_pred_texts)
            gathered_golds = gather_object(local_gold_texts)

            # Only main accumulates to avoid OOM/duplication
            if accelerator.is_main_process:
                # gather_object returns a list-of-lists (one per process); flatten it
                if isinstance(gathered_preds[0], list):
                    for sub in gathered_preds:
                        all_pred_texts.extend(sub)
                    for sub in gathered_golds:
                        all_gold_texts.extend(sub)
                else:
                    # single-process case
                    all_pred_texts.extend(gathered_preds)
                    all_gold_texts.extend(gathered_golds)

                if log_first_batch and step == 0:
                    print("\n[Val Sample]")
                    print("PROMPT  :", batch["prompts"][0][:200].replace("\n", " \\n "))
                    print("PREDICT :", local_pred_texts[0])
                    print("GOLD    :", local_gold_texts[0])

                if step % 500 == 0 or step == total_steps_per_epoch - 1:
                    remaining = total_steps_per_epoch - step - 1
                    accelerator.print(
                        f"Step {step+1}/{total_steps_per_epoch} "
                        f"({remaining} steps left) "
                    )

    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_validation_metrics(all_pred_texts, all_gold_texts)

    accelerator.wait_for_everyone()
    return metrics

def test_geo_inf_model(model_name,checkpoint_path,test_dataset,batch_size,checkpoint_folder="checkpoints/",max_new_tokens=25,wandb_project=None,run_note=None,wandb_config=None):
  
    # Optim + sched
    accelerator = Accelerator(mixed_precision="bf16")

    if isinstance(model_name, str):  # Ensure model_name is a string
        load_model = load_model_gemma if "gemma" in model_name else load_model_else
    else:
        raise ValueError("model_name must be a string")

    
    load_dotenv()  # will read .env automatically

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("hf_token loaded")

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        print("wandb_key loaded")

    model,tokenizer=load_model(model_name,hf_token=hf_token,using_accelerator=True,checkpoint_folder=checkpoint_folder,checkpoint_path=checkpoint_path)
    


    print(f"Setting max new token generation: {max_new_tokens}")


    
    test_collator = DataCollatorForEval(tokenizer)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator,num_workers=4, pin_memory=True)


    # Prepare everything for distributed
    model, test_loader = accelerator.prepare(model, test_loader)

 

    # --- Create output directories ---
    
    temp_folder= f"{model_name}_{checkpoint_path}/best"if checkpoint_path else model_name
    test_result_output_drive= os.path.join(checkpoint_folder, temp_folder)
    if accelerator.is_main_process:
        os.makedirs(test_result_output_drive, exist_ok=True)
        
        if wandb_project and wandb_key:
            wandb.login(key=wandb_key)
            
            if run_note and checkpoint_path:
               run_name=f"{model_name}_{checkpoint_path}_{run_note}_evaluation"
            else:
                if checkpoint_path:
                     run_name=f"{model_name}_{checkpoint_path}_evaluation"
                elif run_note:
                     run_name=f"{model_name}_{run_note}_evaluation"
                else:
                     run_name=f"{model_name}_evaluation"
            if wandb_config:
                wandb.init(project=wandb_project,name=run_name,config=wandb_config)
            else:
                wandb.init(project=wandb_project,name=run_name)


    accelerator.wait_for_everyone()


    # Testing
    metrics = evaluate(model, test_loader, tokenizer, accelerator, max_new_tokens=max_new_tokens, log_first_batch=True)

    # accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if wandb_project and wandb_key:
            county_f1=metrics.get("county_f1", -1.0)
            county_acc=metrics.get("county_acc", -1.0)
            county_precision=metrics.get("county_precision", -1.0)
            county_recall=metrics.get("county_recall", -1.0)

            state_f1=metrics.get("state_f1", -1.0)
            state_precision=metrics.get("state_precision", -1.0)
            state_acc=metrics.get("state_acc", -1.0)
            state_recall=metrics.get("state_recall", -1.0)

            us_f1=metrics.get("us_f1", -1.0)
            us_acc=metrics.get("us_acc", -1.0)
            us_precision=metrics.get("us_precision", -1.0)
            us_recall=metrics.get("us_recall", -1.0)


            wandb.log({"test/county_f1": county_f1, "test/county_recall": county_recall, "test/county_acc": county_acc,"test/county_precision": county_precision, "test/state_f1": state_f1, "test/state_precision": state_precision, "test/state_acc": state_acc, "test/state_recall": state_recall, "test/us_f1": us_f1, "test/us_recall": us_recall, "test/us_acc": us_acc,"test/us_precision": us_precision})

        
        

        if run_note:
             file_path = os.path.join(test_result_output_drive, f"test_metrics_{run_note}.json")
        else:
            file_path = os.path.join(test_result_output_drive, "test_metrics.json")

        # If file exists, append timestamp
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                test_result_output_drive,
                f"test_metrics_{run_note}_{timestamp}.json" if run_note else f"test_metrics_{timestamp}.json"
            )

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=2)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if wandb_project and wandb_key:
            wandb.finish() 

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pre-trained and Finetuned Models")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_folder",type=str,default="checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--shot_type", type=str, choices=["few_shot", "one_shot", "zero_shot"], default=None, help="Specify the shot type: few_shot, one_shot, or zero_shot.")
    parser.add_argument("--run_num", type=int, default=0, help="Run number (non-negative integer, default is 0).")
    return parser.parse_args()



def main_cli():
    """SLURM / accelerate entrypoint."""
    args = parse_args()
    model_name = args.model_name  # replace with your model
    checkpoint_path=args.checkpoint_path
    if not checkpoint_path or checkpoint_path=="":
        checkpoint_path=None
    checkpoint_folder=args.checkpoint_folder
    batch_size=args.batch_size
    max_new_tokens=args.max_new_tokens
    wandb_project=args.wandb_project
    shot_type=args.shot_type
    run_note=None
    run_config=None
    test_df_save_path=None
    data_folder="../data"

    

    if shot_type:
        if shot_type=='few_shot' or shot_type=='one_shot':
            run_num=args.run_num
            if run_num>=0:
                test_df_save_path=f"{data_folder}/{model_name}/pretokenized/{shot_type}/run_{run_num}/test_pretoken"
                shot_path=f"{data_folder}/raw/one_shot/run_{run_num}/{shot_type}_prompt.txt"
                with open(shot_path,'r') as f:
                    shot_prompt=f.read()
                if shot_prompt:
                    run_config={"Prompt example":shot_prompt}
                run_note=f"{shot_type}-prompt_{run_num}"

            else:
                print("Error: --run_num must be a non-negative integer.")
                sys.exit(1)
        elif shot_type=='zero_shot':
            run_note=f"{shot_type}-prompt"
            test_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/test_pretoken"
    else:
        test_df_save_path=f"{data_folder}/{model_name}/pretokenized/original_data/test_pretoken"

    

    if Path(test_df_save_path).exists():
        print("test data set exists")
        test_dataset=load_from_disk(test_df_save_path)
    else:
        print("test data set does not exist",{test_df_save_path})
        sys.exit(1)
    

    if test_dataset!=None:

       test_geo_inf_model(model_name,checkpoint_path,test_dataset, batch_size, checkpoint_folder,max_new_tokens,wandb_project,run_note,run_config)

    else:
        print("Not starting testing, test pretokenized does not exist")



if __name__ == "__main__":
    main_cli()