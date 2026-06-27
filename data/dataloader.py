import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import Tokenizer
from data.dataset import auto_dataset
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
        "fused_lengths": torch.tensor([x.size(0) for x in fused_input_ids], dtype=torch.long),
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=config.PAD_ID),
        "target_ids": target_ids,
        "lengths": torch.tensor([ip.size(0) for ip in input_ids], dtype=torch.long),
    }

def seq2seq_collate_fn(batch):
    encoder_input_ids = [item["encoder_input_ids"] for item in batch]
    decoder_input_ids = [item["decoder_input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    
    return {
        "encoder_input_ids": pad_sequence(encoder_input_ids, batch_first=True, padding_value=config.PAD_ID),
        "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=config.PAD_ID),
        "target_ids": pad_sequence(target_ids, batch_first=True, padding_value=config.PAD_ID),
        "encoder_input_lengths": torch.tensor([ip.size(0) for ip in encoder_input_ids], dtype=torch.long)
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
    dataset = auto_dataset(src_path, tgt_path, tokenizer)

    generator = torch.Generator()
    if config.SEED is not None:
        generator.manual_seed(config.SEED)

    if config.MODEL_TYPE == "seq2seq":
        collate_fn = seq2seq_collate_fn
    elif config.MODEL_TYPE == "causal_lm":
        collate_fn = causal_lm_collate_fn
    else:
        raise ValueError(f"Don't support {config.MODEL_TYPE}.")

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker
    )