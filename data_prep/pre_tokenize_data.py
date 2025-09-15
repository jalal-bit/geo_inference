from datasets import load_dataset
from datasets import load_from_disk
import os


def preprocess_and_save(
    csv_path,
    tokenizer,
    save_path,
    text_column="prompt",
    target_column="target",
    is_train=True,
    max_length=512,
    num_proc=8
):
    
    dataset = load_dataset("csv", data_files=csv_path, split="train")

    def tokenize_fn_train(batch):
        inputs = []
        prompt_lens = []
        for prompt, target in zip(batch[text_column], batch[target_column]):
            # combine prompt + target
            text = prompt + target
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,  # no padding here!
            )
            inputs.append(enc["input_ids"])
            # store how long the prompt was
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
            prompt_lens.append(prompt_len)

        return {
            "input_ids": inputs,
            "prompt_len": prompt_lens,
        }

    def tokenize_fn_eval(batch):
        encodings = tokenizer(
            batch[text_column],
            max_length=max_length,
            truncation=True,
            padding=False,  # no padding here
        )
        return {
            "input_ids": encodings["input_ids"],
            "prompt": batch[text_column],
            "target": batch[target_column],
        }

    func = tokenize_fn_train if is_train else tokenize_fn_eval
    tokenized = dataset.map(func, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)

    os.makedirs(save_path, exist_ok=True)
    tokenized.save_to_disk(save_path)
    print(f"✅ Tokenized dataset saved at {save_path}")
    return tokenized


