import torch
from torch.utils.data import Dataset
import pandas as pd

from data.tokenizer import Tokenizer

class CausalLMDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        with open(src_path, "r") as f:
            src_texts = f.readlines()
            self.src_ids = tokenizer.encode(src_texts, add_bos=False, add_eos=False)
        with open(tgt_path, "r") as f:
            tgt_texts = f.readlines()
            self.tgt_ids = tokenizer.encode(tgt_texts, add_bos=True, add_eos=True)
    
    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, index):
        src = self.src_ids[index]
        tgt = self.tgt_ids[index]
        return {
            "fused_input_ids": torch.tensor(src + tgt[:-1], dtype=torch.long),
            "fused_target_ids": torch.tensor([self.tokenizer.pad_id] * len(src) + tgt[1:], dtype=torch.long),
            "input_ids": torch.tensor(src + tgt[:1], dtype=torch.long),
            "target_ids": tgt
        }
