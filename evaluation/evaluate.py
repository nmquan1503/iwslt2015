import torch
from tqdm import tqdm
import pandas as pd
import time

from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from selective_attention.models import CausalLM, CausalLMConfig
from selective_attention.inference import GenerationConfig, CausalBlockCache, InferenceState
import config
from evaluation.metrics import compute_bleu, compute_rouge

def _generate_preds_causal_lm():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(tokenizer, mode="test")
    device = "cuda"
    model = CausalLM(CausalLMConfig(
        vocab_size=config.VOCAB_SIZE,
        model_dim=config.MODEL_DIM,
        head_dim=config.HEAD_DIM,
        ssm_state_dim=config.SSM_STATE_DIM,
        ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
        ssm_num_groups=config.SSM_NUM_GROUPS,
        ssm_chunk_size=config.SSM_CHUNK_SIZE,
        num_layers=config.NUM_LAYERS
    )).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_inputs, all_preds, all_refs = [], [], []
    ttft_list, tps_list = [], []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"]

            gen_cfg = GenerationConfig(
                attn_gate_thresholds=config.ATTN_GATE_THRESHOLDS,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                max_new_tokens=config.MAX_NEW_TOKENS,
                cache_update_interval=config.CACHE_UPDATE_INTERVAL
            )

            # ===== cache + state (GIỮ GIỐNG generate) =====
            cache = [CausalBlockCache() for _ in range(config.NUM_LAYERS)]
            lengths = (input_ids != gen_cfg.pad_token_id).sum(dim=1)
            state = InferenceState(lengths)

            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            # ================= PREFILL (TTFT START) =================
            torch.cuda.synchronize()
            t0 = time.time()

            if gen_cfg.attn_gate_thresholds is None:
                gen_cfg.attn_gate_thresholds = [0.0] * model.cfg.num_layers

            last_indices = lengths - 1
            logits = model.forward(
                input_ids,
                lengths,
                gen_cfg.attn_gate_thresholds,
                cache
            )
            logits = logits[torch.arange(batch_size, device=device), last_indices]

            torch.cuda.synchronize()
            t1 = time.time()

            ttft_list.append(t1 - t0)

            # ================= DECODING =================
            seq_ids = torch.full(
                (batch_size, seq_len + gen_cfg.max_new_tokens),
                fill_value=gen_cfg.pad_token_id,
                dtype=input_ids.dtype,
                device=device
            )
            seq_ids[:, :seq_len] = input_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            decode_start = time.time()

            for _ in range(gen_cfg.max_new_tokens):

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)
                if torch.any((top2_vals[:, 0] - top2_vals[:, 1]).abs() < 1e-6):
                    print("⚠️ argmax tie / near-tie detected")
                    print("gap:", (top2_vals[:, 0] - top2_vals[:, 1]))

                seq_ids[:, seq_len + state.step] = next_token
                finished |= (next_token == gen_cfg.eos_token_id)

                if finished.all():
                    break

                logits = model.step(next_token, cache, state, gen_cfg)
                state.update()

            torch.cuda.synchronize()
            decode_end = time.time()

            # ===== TPS =====
            decode_time = decode_end - decode_start
            num_tokens = batch_size * gen_cfg.max_new_tokens
            if decode_time > 0:
                tps_list.append(num_tokens / decode_time)

            # ===== EOS TRUNCATE (COPY Y HỆT generate) =====
            eos_mask = (seq_ids == gen_cfg.eos_token_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, gen_cfg.eos_token_id, seq_ids)

            # ===== decode =====
            seq_ids = seq_ids.cpu()
            input_ids = input_ids.cpu()

            for inp, pred, tgt in zip(input_ids, seq_ids, target_ids):
                inp = inp.tolist()
                pred = pred.tolist()

                if tokenizer.bos_id in pred:
                    bos_idx = pred.index(tokenizer.bos_id)
                    pred = pred[bos_idx + 1:]
                if tokenizer.eos_id in pred:
                    eos_idx = pred.index(tokenizer.eos_id)
                    pred = pred[:eos_idx]

                all_inputs.append(tokenizer.decode(inp))
                all_preds.append(tokenizer.decode(pred))
                all_refs.append(tokenizer.decode(tgt))

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tps = sum(tps_list) / len(tps_list)
    peak_memory = torch.cuda.max_memory_allocated()

    return all_inputs, all_preds, all_refs, avg_ttft, avg_tps, peak_memory

def _write_preds(all_inputs, all_preds, all_refs):
    df = pd.DataFrame({
        "source": all_inputs,
        "target": all_refs,
        "prediction": all_preds,
    })
    df.to_csv(config.PREDS_PATH, index=False)

def evaluate():
    all_inputs, all_preds, all_refs, avg_ttft, avg_tps, peak_mem = _generate_preds_causal_lm()
    _write_preds(all_inputs, all_preds, all_refs)

    results = {}

    results.update(compute_bleu(all_preds, all_refs))
    results.update(compute_rouge(all_preds, all_refs))

    print("\n===== QUALITY =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n===== SPEED =====")
    print(f"Avg TTFT (s): {avg_ttft:.4f}")
    print(f"Avg TPS     : {avg_tps:.4f}")

    print("\n===== MEMORY =====")
    print(f"Peak memory: {peak_mem / 1024**3:.4f} GB")
    
if __name__ == "__main__":
    evaluate()