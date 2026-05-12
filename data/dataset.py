import torch
from torch.utils.data import Dataset
import pandas as pd
import re

from data.tokenizer import Tokenizer

class CausalLMDataset(Dataset):
    MAX_LEN = 500

    def __init__(self, src_path: str, tgt_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        with open(src_path, "r") as f:
            src_texts = f.readlines()

        with open(tgt_path, "r") as f:
            tgt_texts = f.readlines()

        assert len(src_texts) == len(tgt_texts)

        self.src_ids = []
        self.tgt_ids = []

        dropped = 0
        split_added = 0

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_text = src_text.strip()
            tgt_text = tgt_text.strip()

            src_ids = tokenizer.encode(
                [src_text],
                add_bos=False,
                add_eos=False
            )[0]

            tgt_ids = tokenizer.encode(
                [tgt_text],
                add_bos=True,
                add_eos=True
            )[0]

            fused_len = len(src_ids) + len(tgt_ids) - 1

            # =========================
            # Normal sample
            # =========================

            if fused_len <= self.MAX_LEN:
                self.src_ids.append(src_ids)
                self.tgt_ids.append(tgt_ids)
                continue

            # =========================
            # Try split by punctuation
            # =========================

            src_sents = self._split_sentences(src_text)
            tgt_sents = self._split_sentences(tgt_text)

            # sentence count mismatch -> drop
            if len(src_sents) != len(tgt_sents):
                dropped += 1
                continue

            current_src = []
            current_tgt = []

            for s_src, s_tgt in zip(src_sents, tgt_sents):
                trial_src = " ".join(current_src + [s_src])
                trial_tgt = " ".join(current_tgt + [s_tgt])

                trial_src_ids = tokenizer.encode(
                    [trial_src],
                    add_bos=False,
                    add_eos=False
                )[0]

                trial_tgt_ids = tokenizer.encode(
                    [trial_tgt],
                    add_bos=True,
                    add_eos=True
                )[0]

                trial_len = len(trial_src_ids) + len(trial_tgt_ids) - 1

                # still safe -> keep accumulating
                if trial_len <= self.MAX_LEN:
                    current_src.append(s_src)
                    current_tgt.append(s_tgt)

                else:
                    # flush current chunk
                    if len(current_src) > 0:
                        final_src = " ".join(current_src)
                        final_tgt = " ".join(current_tgt)

                        self.src_ids.append(
                            tokenizer.encode(
                                [final_src],
                                add_bos=False,
                                add_eos=False
                            )[0]
                        )

                        self.tgt_ids.append(
                            tokenizer.encode(
                                [final_tgt],
                                add_bos=True,
                                add_eos=True
                            )[0]
                        )

                        split_added += 1

                    # start new chunk
                    current_src = [s_src]
                    current_tgt = [s_tgt]

            # flush remaining
            if len(current_src) > 0:
                final_src = " ".join(current_src)
                final_tgt = " ".join(current_tgt)

                final_src_ids = tokenizer.encode(
                    [final_src],
                    add_bos=False,
                    add_eos=False
                )[0]

                final_tgt_ids = tokenizer.encode(
                    [final_tgt],
                    add_bos=True,
                    add_eos=True
                )[0]

                final_len = len(final_src_ids) + len(final_tgt_ids) - 1

                # if even split chunk still too long -> drop
                if final_len <= self.MAX_LEN:
                    self.src_ids.append(final_src_ids)
                    self.tgt_ids.append(final_tgt_ids)
                    split_added += 1
                else:
                    dropped += 1

        print("=" * 50)
        print(f"Final samples : {len(self.src_ids)}")
        print(f"Split added   : {split_added}")
        print(f"Dropped       : {dropped}")
        print("=" * 50)

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
