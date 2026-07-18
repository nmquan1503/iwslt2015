import torch
import argparse
import os

from selective_attention.models import (
    CausalLM, CausalLMConfig,
    Seq2SeqLM, Seq2SeqLMConfig
)
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import CausalLMTrainer, Seq2SeqTrainer
import config

def train():
    if config.SEED is not None:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    tokenizer = Tokenizer()
    train_loader = build_dataloader(tokenizer, mode="train")
    dev_loader = build_dataloader(tokenizer, mode="dev")
    device = "cuda"
    if config.MODEL_TYPE == "causal_lm":
        model = CausalLM(CausalLMConfig(
            vocab_size=config.VOCAB_SIZE,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
            attn_log_gate_penalty=config.ATTN_LOG_GATE_PENALTY,
            ssm_state_dim=config.SSM_STATE_DIM,
            ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
            ssm_num_groups=config.SSM_NUM_GROUPS,
            ssm_chunk_size=config.SSM_CHUNK_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            device=device
        )).to("cuda")
    else:
        model = Seq2SeqLM(Seq2SeqLMConfig(
            vocab_size=config.VOCAB_SIZE,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
            attn_log_gate_penalty=config.ATTN_LOG_GATE_PENALTY,
            ssm_state_dim=config.SSM_STATE_DIM,
            ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
            ssm_num_groups=config.SSM_NUM_GROUPS,
            ssm_chunk_size=config.SSM_CHUNK_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            device=device
        )).to("cuda")
    model.warmup(config.BATCH_SIZE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    if config.MODEL_TYPE == "seq2seq":
        trainer = Seq2SeqTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion
        )
    elif config.MODEL_TYPE == "causal_lm":
        trainer = CausalLMTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion
        )

    trainer.train()

if __name__ == "__main__":
    train()