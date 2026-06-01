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

    with torch.inference_mode():

        all_inputs = []
        all_preds = []
        all_refs = []

        ttft_list = []
        tps_list = []

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        for batch in tqdm(test_loader, desc="Test"):
            gen_input_ids = batch["input_ids"].to("cuda")
            target_ids = batch["target_ids"]

            gen_cfg = GenerationConfig(
                attn_gate_thresholds=config.ATTN_GATE_THRESHOLDS,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                max_new_tokens=config.MAX_NEW_TOKENS,
                cache_update_interval=config.CACHE_UPDATE_INTERVAL
            )

            cache = [CausalBlockCache() for _ in range(config.NUM_LAYERS)]

            lengths = (gen_input_ids != gen_cfg.pad_token_id).sum(dim=1)
            state = InferenceState(lengths)

            torch.cuda.synchronize()
            t0 = time.time()

            logits = model.forward(
                gen_input_ids,
                lengths,
                gen_cfg.attn_gate_thresholds,
                cache
            )

            batch_size = gen_input_ids.size(0)
            last_indices = lengths - 1
            logits = logits[torch.arange(batch_size, device=device), last_indices]

            torch.cuda.synchronize()
            t1 = time.time()

            ttft = t1 - t0
            ttft_list.append(ttft)

            seq_ids = torch.full(
                (batch_size, gen_input_ids.size(1) + gen_cfg.max_new_tokens),
                fill_value=gen_cfg.pad_token_id,
                dtype=gen_input_ids.dtype,
                device=device
            )

            seq_ids[:, :gen_input_ids.size(1)] = gen_input_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            torch.cuda.synchronize()
            t2 = time.time()

            step_count = 0

            for step in range(gen_cfg.max_new_tokens):
                next_token = torch.argmax(logits, dim=-1)

                seq_ids[:, gen_input_ids.size(1) + step] = next_token
                finished |= (next_token == gen_cfg.eos_token_id)

                step_count += 1

                if finished.all():
                    break

                logits = model.step(next_token, cache, state, gen_cfg)
                state.update()

                decode_peak_mem = max(
                    decode_peak_mem,
                    torch.cuda.max_memory_allocated()
                )

            torch.cuda.synchronize()
            t3 = time.time()

            decode_time = t3 - t2
            num_tokens = step_count * batch_size

            if decode_time > 0:
                tps_list.append(num_tokens / decode_time)

            input_ids = gen_input_ids.cpu()
            seq_ids = seq_ids.cpu()

            for input, pred, tgt in zip(input_ids, seq_ids, target_ids):
                input = input.tolist()
                pred = pred.tolist()
                start_pred_idx = pred.index(tokenizer.bos_id) if tokenizer.bos_id in pred else -1
                if start_pred_idx != -1:
                    pred = pred[start_pred_idx:]

                input_text = tokenizer.decode(input)
                pred_text = tokenizer.decode(pred)
                tgt_text = tokenizer.decode(tgt)

                all_inputs.append(input_text)
                all_preds.append(pred_text)
                all_refs.append(tgt_text)
    
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