import torch
from torch.utils.data import Dataset
import pandas as pd
import re

from data.tokenizer import Tokenizer
import config

class CausalLMDataset(Dataset):
    MAX_LEN = 500

    def __init__(self, src_path: str, tgt_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.src_ids = []
        self.tgt_ids = []

        with open(src_path) as f:
            src_texts = [x.strip() for x in f]

        with open(tgt_path) as f:
            tgt_texts = [x.strip() for x in f]

        assert len(src_texts) == len(tgt_texts)

        enc_src = lambda x: tokenizer.encode([x], add_bos=False, add_eos=False, add_cls=False)[0]
        enc_tgt = lambda x: tokenizer.encode([x], add_bos=True, add_eos=True, add_cls=False)[0]

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_ids = enc_src(src_text)
            tgt_ids = enc_tgt(tgt_text)

            if len(src_ids) + len(tgt_ids) - 1 <= self.MAX_LEN:
                self.src_ids.append(src_ids)
                self.tgt_ids.append(tgt_ids)
                continue

            src_sents = self._split_sentences(src_text)
            tgt_sents = self._split_sentences(tgt_text)

            if len(src_sents) != len(tgt_sents):
                continue

            cur_src, cur_tgt = [], []

            for s_src, s_tgt in zip(src_sents, tgt_sents):
                trial_src = cur_src + [s_src]
                trial_tgt = cur_tgt + [s_tgt]

                if len(enc_src(" ".join(trial_src))) + len(enc_tgt(" ".join(trial_tgt))) - 1 <= self.MAX_LEN:
                    cur_src, cur_tgt = trial_src, trial_tgt
                    continue

                if cur_src:
                    self.src_ids.append(enc_src(" ".join(cur_src)))
                    self.tgt_ids.append(enc_tgt(" ".join(cur_tgt)))

                cur_src, cur_tgt = [s_src], [s_tgt]

            if cur_src:
                src_ids = enc_src(" ".join(cur_src))
                tgt_ids = enc_tgt(" ".join(cur_tgt))

                if len(src_ids) + len(tgt_ids) - 1 <= self.MAX_LEN:
                    self.src_ids.append(src_ids)
                    self.tgt_ids.append(tgt_ids)

    def _split_sentences(self, text):
        text = text.strip()

        sents = re.split(r'(?<=[.!?;:])\s+', text)

        sents = [
            s.strip()
            for s in sents
            if len(s.strip()) > 0
        ]

        return sents
    
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


class Seq2SeqDataset(Dataset):
    MAX_LEN = 250

    def __init__(self, src_path: str, tgt_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.src_ids = []
        self.tgt_ids = []

        with open(src_path) as f:
            src_texts = [x.strip() for x in f]

        with open(tgt_path) as f:
            tgt_texts = [x.strip() for x in f]

        assert len(src_texts) == len(tgt_texts)

        enc_src = lambda x: tokenizer.encode([x], add_bos=False, add_eos=False, add_cls=True)[0]
        enc_tgt = lambda x: tokenizer.encode([x], add_bos=True, add_eos=True, add_cls=False)[0]

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_ids = enc_src(src_text)
            tgt_ids = enc_tgt(tgt_text)

            if max(len(src_ids), len(tgt_ids)) <= self.MAX_LEN:
                self.src_ids.append(src_ids)
                self.tgt_ids.append(tgt_ids)
                continue

            src_sents = self._split_sentences(src_text)
            tgt_sents = self._split_sentences(tgt_text)

            if len(src_sents) != len(tgt_sents):
                continue

            cur_src, cur_tgt = [], []

            for s_src, s_tgt in zip(src_sents, tgt_sents):
                trial_src = cur_src + [s_src]
                trial_tgt = cur_tgt + [s_tgt]

                if max(
                    len(enc_src(" ".join(trial_src))),
                    len(enc_tgt(" ".join(trial_tgt))),
                ) <= self.MAX_LEN:
                    cur_src, cur_tgt = trial_src, trial_tgt
                    continue

                if cur_src:
                    self.src_ids.append(enc_src(" ".join(cur_src)))
                    self.tgt_ids.append(enc_tgt(" ".join(cur_tgt)))

                cur_src, cur_tgt = [s_src], [s_tgt]

            if cur_src:
                src_ids = enc_src(" ".join(cur_src))
                tgt_ids = enc_tgt(" ".join(cur_tgt))

                if max(len(src_ids), len(tgt_ids)) <= self.MAX_LEN:
                    self.src_ids.append(src_ids)
                    self.tgt_ids.append(tgt_ids)

    def _split_sentences(self, text):
        return [
            s.strip()
            for s in re.split(r'(?<=[.!?;:])\s+', text.strip())
            if s.strip()
        ]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, index):
        src = self.src_ids[index]
        tgt = self.tgt_ids[index]

        return {
            "encoder_input_ids": torch.tensor(src, dtype=torch.long),
            "decoder_input_ids": torch.tensor(tgt[:-1], dtype=torch.long),
            "target_ids": torch.tensor(tgt[1:], dtype=torch.long),
        }

def auto_dataset(src_path, tgt_path, tokenizer):
    if config.MODEL_TYPE == "seq2seq":
        return Seq2SeqDataset(src_path, tgt_path, tokenizer)
    elif config.MODEL_TYPE == "causal_lm":
        return CausalLMDataset(src_path, tgt_path, tokenizer)
    else:
        raise ValueError(f"Don't support {config.MODEL_TYPE}.")