import sentencepiece as spm
from pathlib import Path
from typing import List
import pandas as pd
import os

import config

def train_tokenizer():
    all_texts = []
    with open(config.TRAIN_SRC_PATH, "r") as f:
        all_texts.extend(f.readlines())
    with open(config.TRAIN_TGT_PATH, "r") as f:
        all_texts.extend(f.readlines())
    with open("_temp_texts.txt", "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t.strip() + "\n")
    
    spm.SentencePieceTrainer.Train(
        input=f"_temp_texts.txt",
        model_prefix=config.SPM_MODEL_PATH.split(".model")[0],
        vocab_size=config.VOCAB_SIZE,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
        minloglevel=2
    )

    os.remove("_temp_texts.txt")

class Tokenizer:
    def __init__(self):
        if not Path(config.SPM_MODEL_PATH).exists():
            train_tokenizer()
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(config.SPM_MODEL_PATH)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
    
    def encode(self, texts: str | List[str], add_bos=True, add_eos=True):
        ids = self.sp.Encode(texts, out_type=int)

        if add_bos:
            if isinstance(texts, str):
                ids = [self.bos_id] + ids
            else:
                ids = [[self.bos_id] + i for i in ids]
        
        if add_eos:
            if isinstance(texts, str):
                ids = ids + [self.eos_id]
            else:
                ids = [i + [self.eos_id] for i in ids]
        
        return ids

    def decode(self, ids: List[int] | List[List[int]]):
        return self.sp.Decode(ids)