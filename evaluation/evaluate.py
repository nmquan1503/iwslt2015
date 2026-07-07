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
from evaluation.metrics import compute_bleu, compute_rouge

def _generate_preds_causal_lm():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(tokenizer, mode="test")

    device = "cuda"
    model = CausalLM(
        CausalLMConfig(
            vocab_size=config.VOCAB_SIZE,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
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
    global_sums = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_counts = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]

    all_inputs, all_preds, all_refs = [], [], []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            pred_ids, batch_stats = model.generate(input_ids, gen_cfg, analysis_cfg)
            pred_ids = pred_ids.cpu()

            for layer_idx in range(config.NUM_LAYERS):
                layer_stats = batch_stats[layer_idx]["causal_attn_gate_analysis"]
                global_sums[layer_idx] += layer_stats["sum"].cpu()
                global_counts[layer_idx] += layer_stats["count"].cpu()

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
    
    layers_stats = []
    for s, c in zip(global_sums, global_counts):
        layers_stats.append({
            "causal_attn_gate_analysis": {"sum": s, "count": c}
        })
    torch.save({
        "layers": layers_stats,
        "num_bins": num_bins
    }, "gate_attn_stats.pt")
    print("Đã lưu gate_attn_stats.pt")

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
    global_enc_sums = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_enc_counts = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_dec_sums = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_dec_counts = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_cross_sums = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]
    global_cross_counts = [torch.zeros(num_heads, num_bins, device=device) for _ in range(num_layers = config.NUM_LAYERS)]

    all_inputs, all_preds, all_refs = [], [], []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["encoder_input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            pred_ids, batch_stats = model.generate(input_ids, gen_cfg, analysis_cfg)
            pred_ids = pred_ids.cpu()

            for layer_idx in range(config.NUM_LAYERS):
                layer_enc_stats = batch_stats[layer_idx]["non_causal_attn_gate_analysis"]
                global_enc_sums[layer_idx] += layer_enc_stats["sum"].cpu()
                global_enc_counts[layer_idx] += layer_enc_stats["count"].cpu()
                
                layer_dec_stats = batch_stats[layer_idx]["causal_attn_gate_analysis"]
                global_dec_sums[layer_idx] += layer_dec_stats["sum"].cpu()
                global_dec_counts[layer_idx] += layer_dec_stats["count"].cpu()

                layer_cross_stats = batch_stats[layer_idx]["cross_attn_gate_analysis"]
                global_cross_sums[layer_idx] += layer_cross_stats["sum"].cpu()
                global_cross_counts[layer_idx] += layer_cross_stats["count"].cpu()

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

    layers_stats = []

    for layer_idx in range(config.NUM_LAYERS):
        layers_stats.append({
            "non_causal_attn_gate_analysis": {
                "sum": global_enc_sums[layer_idx],
                "count": global_enc_counts[layer_idx],
            },
            "causal_attn_gate_analysis": {
                "sum": global_dec_sums[layer_idx],
                "count": global_dec_counts[layer_idx],
            },
            "cross_attn_gate_analysis": {
                "sum": global_cross_sums[layer_idx],
                "count": global_cross_counts[layer_idx],
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
    }

    print("\n===== QUALITY =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n===== MEMORY =====")
    print(f"Peak memory: {peak_mem / 1024**3:.4f} GB")
    
if __name__ == "__main__":
    evaluate()