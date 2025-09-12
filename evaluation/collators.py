import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollatorForEval:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id,padding_side="left")

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids[:, -self.max_length :],
            "attention_mask": attention_mask[:, -self.max_length :],
            "prompts": [x["prompt"] for x in batch],
            "targets": [x["target"] for x in batch],
        }