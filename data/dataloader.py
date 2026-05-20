import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import Tokenizer
from data.dataset import CausalLMDataset
import config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

def causal_lm_collate_fn(batch):
    fused_input_ids = [item["fused_input_ids"] for item in batch]
    fused_target_ids = [item["fused_target_ids"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    return {
        "fused_input_ids": pad_sequence(fused_input_ids, batch_first=True, padding_value=config.PAD_ID),
        "fused_target_ids": pad_sequence(fused_target_ids, batch_first=True, padding_value=config.PAD_ID),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_ID),
        "target_ids": target_ids,
        "lengths": torch.tensor([ip.size(0) for ip in input_ids], dtype=torch.long),
    }

def build_dataloader(tokenizer: Tokenizer, mode="train"):
    if mode == "train":
        src_path = config.TRAIN_SRC_PATH
        tgt_path = config.TRAIN_TGT_PATH
    elif mode == "dev":
        src_path = config.DEV_SRC_PATH
        tgt_path = config.DEV_TGT_PATH
    else:
        src_path = config.TEST_SRC_PATH
        tgt_path = config.TEST_TGT_PATH
    dataset = CausalLMDataset(src_path, tgt_path, tokenizer)

    generator = torch.Generator()
    if config.SEED is not None:
        generator.manual_seed(config.SEED)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=causal_lm_collate_fn,
        generator=generator,
        worker_init_fn=seed_worker
    )