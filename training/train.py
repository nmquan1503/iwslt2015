import torch
import argparse

from selective_attention.models import CausalLM, CausalLMConfig
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer
import config

def train():
    tokenizer = Tokenizer()
    train_loader = build_dataloader(tokenizer, mode="train")
    dev_loader = build_dataloader(tokenizer, mode="dev")
    model = CausalLM(CausalLMConfig(
        vocab_size=config.VOCAB_SIZE,
        model_dim=config.MODEL_DIM,
        head_dim=config.HEAD_DIM,
        ssm_state_dim=config.SSM_STATE_DIM,
        ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
        ssm_num_groups=config.SSM_NUM_GROUPS,
        ssm_chunk_size=config.SSM_CHUNK_SIZE,
        mlconv_radius=config.MLCONV_RADIUS,
        num_layers=config.NUM_LAYERS
    ))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.train()

if __name__ == "__main__":
    train()