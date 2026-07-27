import torch
from tqdm import tqdm
import pandas as pd
import time

from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from minimal_attention.models import (
    CausalLM, CausalLMConfig,
    Seq2SeqLM, Seq2SeqLMConfig
)
from minimal_attention.inference import GenerationConfig, AnalysisConfig
import config

def compute_gate_threshold():
    tokenizer = Tokenizer()
    dev_loader = build_dataloader(tokenizer, mode="dev")
    device = "cuda"

    if config.MODEL_TYPE == "causal_lm":
        model = CausalLM(
            CausalLMConfig(
                vocab_size=config.VOCAB_SIZE,
                model_dim=config.MODEL_DIM,
                head_dim=config.HEAD_DIM,
                attn_log_gate_penalty=config.ATTN_LOG_GATE_PENALTY,
                ssm_state_dim=config.SSM_STATE_DIM,
                ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
                ssm_num_groups=config.SSM_NUM_GROUPS,
                ssm_chunk_size=config.SSM_CHUNK_SIZE,
                num_layers=config.NUM_LAYERS,
                device=device
            )
        ).to(device)
    elif config.MODEL_TYPE == "seq2seq":
        model = Seq2SeqLM(
            Seq2SeqLMConfig(
                vocab_size=config.VOCAB_SIZE,
                model_dim=config.MODEL_DIM,
                head_dim=config.HEAD_DIM,
                attn_log_gate_penalty=config.ATTN_LOG_GATE_PENALTY,
                ssm_state_dim=config.SSM_STATE_DIM,
                ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
                ssm_num_groups=config.SSM_NUM_GROUPS,
                ssm_chunk_size=config.SSM_CHUNK_SIZE,
                num_layers=config.NUM_LAYERS,
                device=device
            )
        ).to(device)
    else:
        raise ValueError(f"Don't support {config.MODEL_TYPE}")

    model.warmup(config.BATCH_SIZE)
    model.load_state_dict(
        torch.load(
            config.BEST_MODEL_PATH,
            map_location=device,
        )
    )
    model.eval()

    gen_cfg = GenerationConfig(
        attn_gate_thresholds=config.ATTN_GATE_THRESHOLDS,
        cross_attn_gate_thresholds=config.CROSS_ATTN_GATE_THRESHOLDS,
        enc_attn_gate_thresholds=config.ENC_ATTN_GATE_THRESHOLDS,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        max_new_tokens=config.MAX_NEW_TOKENS,
        cache_update_interval=config.CACHE_UPDATE_INTERVAL,
    )
    analysis_cfg = AnalysisConfig(gate_attn_num_bins=config.ATTN_GATE_NUM_BINS)

    if config.MODEL_TYPE == "causal_lm":
        inputs = [batch["input_ids"] for batch in dev_loader]
        threshold = model.compute_attn_gate_threshold(
            inputs, config.ATTN_MASS_THRESHOLD, gen_cfg, analysis_cfg
        )
        print("Gate threshold:")
        print(threshold)
    elif config.MODEL_TYPE == "seq2seq":
        inputs = [batch["encoder_input_ids"] for batch in dev_loader]
        enc_threshold, cross_threshold, dec_threshold = model.compute_attn_gate_threshold(
            inputs, config.ATTN_MASS_THRESHOLD, gen_cfg, analysis_cfg
        )
        print("Encoder gate threshold:")
        print(enc_threshold)
        print("=" * 10)
        print("Cross gate threshold:")
        print(cross_threshold)
        print("=" * 10)
        print("Decoder gate threshold:")
        print(dec_threshold)
    
if __name__ == "__main__":
    compute_gate_threshold()