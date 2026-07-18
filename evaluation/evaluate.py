import torch
from tqdm import tqdm
import pandas as pd
import time

from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from selective_attention.models import (
    CausalLM, CausalLMConfig,
    Seq2SeqLM, Seq2SeqLMConfig
)
from selective_attention.inference import GenerationConfig, AnalysisConfig
import config
from evaluation.metrics import compute_bleu, compute_rouge, compute_comet

def _generate_preds_causal_lm():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(tokenizer, mode="test")

    device = "cuda"
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
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        max_new_tokens=config.MAX_NEW_TOKENS,
        cache_update_interval=config.CACHE_UPDATE_INTERVAL,
    )

    analysis_cfg = AnalysisConfig()
    num_heads = config.MODEL_DIM // config.HEAD_DIM
    num_bins = 100
    global_attn_mass = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_attn_count = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_gate_freq = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]

    all_inputs, all_preds, all_refs = [], [], []

    total_kept_ratio_sum = 0.0
    total_batches = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            if config.ANALYSIS:
                pred_ids, stats_dict = model.generate(input_ids, gen_cfg, analysis_cfg)
                batch_stats = stats_dict["layers"]
                for layer_idx in range(config.NUM_LAYERS):
                    layer_stats = batch_stats[layer_idx]["causal_attn_gate_analysis"]
                    global_attn_mass[layer_idx] += layer_stats["attn_mass"]
                    global_attn_count[layer_idx] += layer_stats["attn_count"]
                    global_gate_freq[layer_idx] += layer_stats["gate_freq"]
                token_kept_ratio = stats_dict["overall"]["token_kept_ratio"]
                total_kept_ratio_sum += token_kept_ratio
                total_batches += 1
            else:
                pred_ids = model.generate(input_ids, gen_cfg)

            pred_ids = pred_ids.cpu()

            for inp, pred, tgt in zip(
                input_ids.cpu(),
                pred_ids,
                batch["target_ids"],
            ):
                pred = pred.tolist()

                if tokenizer.bos_id in pred:
                    pred = pred[pred.index(tokenizer.bos_id) + 1:]

                if tokenizer.eos_id in pred:
                    pred = pred[:pred.index(tokenizer.eos_id)]

                all_inputs.append(tokenizer.decode(inp.tolist()))
                all_preds.append(tokenizer.decode(pred))
                all_refs.append(tokenizer.decode(tgt))
    
    if config.ANALYSIS:
        layers_stats = []
        for mass, count, freq in zip(global_attn_mass, global_attn_count, global_gate_freq):
            layers_stats.append({
                "causal_attn_gate_analysis": {
                    "attn_mass": mass,
                    "attn_count": count,
                    "gate_freq": freq
                }
            })
        torch.save({
            "layers": layers_stats,
            "num_bins": num_bins
        }, "gate_attn_stats.pt")
        print("Đã lưu gate_attn_stats.pt")
        if total_batches > 0:
            avg_kept_ratio = total_kept_ratio_sum / total_batches
            print(f"Token kept ratio trung bình: {avg_kept_ratio:.4f}")

    return (
        all_inputs,
        all_preds,
        all_refs,
        torch.cuda.max_memory_allocated(),
    )

def _generate_preds_seq2seq():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(tokenizer, mode="test")

    device = "cuda"
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
    model.warmup(config.BATCH_SIZE)

    model.load_state_dict(
        torch.load(
            config.BEST_MODEL_PATH,
            map_location=device,
        )
    )
    model.eval()

    gen_cfg = GenerationConfig(
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
        max_new_tokens=config.MAX_NEW_TOKENS,
        cache_update_interval=config.CACHE_UPDATE_INTERVAL,
        enc_attn_gate_thresholds=config.ENC_ATTN_GATE_THRESHOLDS,
        attn_gate_thresholds=config.ATTN_GATE_THRESHOLDS,
        cross_attn_gate_thresholds=config.CROSS_ATTN_GATE_THRESHOLDS
    )

    analysis_cfg = AnalysisConfig()
    num_heads = config.MODEL_DIM // config.HEAD_DIM
    num_bins = 100
    global_enc_attn_mass = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_enc_attn_count = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_enc_gate_freq = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]

    global_dec_attn_mass = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_dec_attn_count = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_dec_gate_freq = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]

    global_cross_attn_mass = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_cross_attn_count = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]
    global_cross_gate_freq = [torch.zeros(num_heads, num_bins, device=device) for _ in range(config.NUM_LAYERS)]

    all_inputs, all_preds, all_refs = [], [], []

    total_kept_ratio_sum = 0.0
    total_batches = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["encoder_input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            if config.ANALYSIS:
                pred_ids, stats_dict = model.generate(input_ids, gen_cfg, analysis_cfg)
                batch_stats = stats_dict["layers"]
                for layer_idx in range(config.NUM_LAYERS):
                    # Encoder stats
                    layer_enc_stats = batch_stats[layer_idx]["non_causal_attn_gate_analysis"]
                    global_enc_attn_mass[layer_idx] += layer_enc_stats["attn_mass"]
                    global_enc_attn_count[layer_idx] += layer_enc_stats["attn_count"]
                    global_enc_gate_freq[layer_idx] += layer_enc_stats["gate_freq"]

                    # Decoder stats
                    layer_dec_stats = batch_stats[layer_idx]["causal_attn_gate_analysis"]
                    global_dec_attn_mass[layer_idx] += layer_dec_stats["attn_mass"]
                    global_dec_attn_count[layer_idx] += layer_dec_stats["attn_count"]
                    global_dec_gate_freq[layer_idx] += layer_dec_stats["gate_freq"]

                    # Cross-attention stats
                    layer_cross_stats = batch_stats[layer_idx]["cross_attn_gate_analysis"]
                    global_cross_attn_mass[layer_idx] += layer_cross_stats["attn_mass"]
                    global_cross_attn_count[layer_idx] += layer_cross_stats["attn_count"]
                    global_cross_gate_freq[layer_idx] += layer_cross_stats["gate_freq"]
                token_kept_ratio = stats_dict["overall"]["token_kept_ratio"]
                total_kept_ratio_sum += token_kept_ratio
                total_batches += 1
            else:
                pred_ids = model.generate(input_ids, gen_cfg)

            pred_ids = pred_ids.cpu()

            for inp, pred, tgt in zip(
                input_ids.cpu(),
                pred_ids,
                target_ids,
            ):
                pred = pred.tolist()
                tgt = tgt.tolist()

                if tokenizer.bos_id in pred:
                    pred = pred[pred.index(tokenizer.bos_id) + 1:]

                if tokenizer.eos_id in pred:
                    pred = pred[:pred.index(tokenizer.eos_id)]

                all_inputs.append(tokenizer.decode(inp.tolist()))
                all_preds.append(tokenizer.decode(pred))
                all_refs.append(tokenizer.decode(tgt))

    if config.ANALYSIS:
        layers_stats = []

        for layer_idx in range(config.NUM_LAYERS):
            layers_stats.append({
                "non_causal_attn_gate_analysis": {
                    "attn_mass": global_enc_attn_mass[layer_idx],
                    "attn_count": global_enc_attn_count[layer_idx],
                    "gate_freq": global_enc_gate_freq[layer_idx],
                },
                "causal_attn_gate_analysis": {
                    "attn_mass": global_dec_attn_mass[layer_idx],
                    "attn_count": global_dec_attn_count[layer_idx],
                    "gate_freq": global_dec_gate_freq[layer_idx],
                },
                "cross_attn_gate_analysis": {
                    "attn_mass": global_cross_attn_mass[layer_idx],
                    "attn_count": global_cross_attn_count[layer_idx],
                    "gate_freq": global_cross_gate_freq[layer_idx],
                },
            })

        torch.save(
            {
                "layers": layers_stats,
                "num_bins": num_bins,
            },
            "gate_attn_stats.pt",
        )

        print("Đã lưu gate_attn_stats.pt")
        if total_batches > 0:
            avg_kept_ratio = total_kept_ratio_sum / total_batches
            print(f"Token kept ratio trung bình: {avg_kept_ratio:.4f}")

    return (
        all_inputs,
        all_preds,
        all_refs,
        torch.cuda.max_memory_allocated(),
    )

def _write_preds(all_inputs, all_preds, all_refs):
    df = pd.DataFrame({
        "source": all_inputs,
        "target": all_refs,
        "prediction": all_preds,
    })
    df.to_csv(config.PREDS_PATH, index=False)

def evaluate():
    if config.MODEL_TYPE == "causal_lm":
        all_inputs, all_preds, all_refs, peak_mem = _generate_preds_causal_lm()
    elif config.MODEL_TYPE == "seq2seq":
        all_inputs, all_preds, all_refs, peak_mem = _generate_preds_seq2seq()
    else:
        raise ValueError(f"Don't support {config.MODEL_TYPE}")

    _write_preds(all_inputs, all_preds, all_refs)

    metrics = {
        **compute_bleu(all_preds, all_refs),
        **compute_rouge(all_preds, all_refs),
        **compute_comet(all_inputs, all_preds, all_refs, config.COMET_BATCH_SIZE, config.COMET_NUM_GPUS)
    }

    print("\n===== QUALITY =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n===== MEMORY =====")
    print(f"Peak memory: {peak_mem / 1024**3:.4f} GB")
    
if __name__ == "__main__":
    evaluate()