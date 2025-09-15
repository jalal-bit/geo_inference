import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_gemma(model_name, hf_token, using_accelerator=False, checkpoint_folder=None,checkpoint_path=None):
    
    # Determine the model source (either base model or checkpoint directory)
    
    if checkpoint_folder and checkpoint_path:
        model_source=os.path.join(checkpoint_folder,f"{model_name}_{checkpoint_path}/best")
    else:
        model_source = model_name
        

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    # Load model from checkpoint or model name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16, 
        device_map=None if using_accelerator else "auto",
        token=hf_token,
        attn_implementation="eager"
    )
    

    if checkpoint_path:
        print(f"✅ Loaded model and tokenizer from checkpoint: {checkpoint_path}/best")
    else:
        print(f"✅ Loaded base model: {model_name}")

    return model, tokenizer



def load_model_else(model_name, hf_token, using_accelerator=False, checkpoint_folder=None,checkpoint_path=None):
    
    # Determine the model source (either base model or checkpoint directory)
    if checkpoint_folder and checkpoint_path:
        model_source=os.path.join(checkpoint_folder,f"{model_name}_{checkpoint_path}/best")
    else:
        model_source = model_name



    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    

    # Load model from checkpoint or model name
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,  # L40S -> use bf16
        device_map=None if using_accelerator else "auto",
        token=hf_token,
    )

    if "llama" in model_name.lower():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, "config") and model.config is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    if checkpoint_path:
        print(f"✅ Loaded model and tokenizer from checkpoint: {checkpoint_path}/best")
    else:
        print(f"✅ Loaded base model: {model_name}")

    return model, tokenizer