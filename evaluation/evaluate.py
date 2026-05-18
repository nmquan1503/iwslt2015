import torch
from tqdm import tqdm
import pandas as pd

from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from selective_attention.models import CausalLM, CausalLMConfig
from selective_attention.inference import GenerationConfig
import config
from evaluation.metrics import compute_bleu

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
        mlconv_radius=config.MLCONV_RADIUS,
        num_layers=config.NUM_LAYERS
    )).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_inputs = []
    all_preds = []
    all_refs = []

    for batch in tqdm(test_loader, desc="Test"):
        gen_input_ids = batch["input_ids"].to("cuda")
        target_ids = batch["target_ids"]

        seq_ids = model.generate(
            gen_input_ids, GenerationConfig(
                attn_gate_threshold=config.ATTN_GATE_THRESHOLD,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
                max_new_tokens=config.MAX_NEW_TOKENS
            )
        ).cpu()
        input_ids = gen_input_ids.cpu()

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
    
    return all_inputs, all_preds, all_refs

def _write_preds(all_inputs, all_preds, all_refs):
    df = pd.DataFrame({
        "source": all_inputs,
        "target": all_refs,
        "prediction": all_preds,
    })
    df.to_csv(config.PREDS_PATH, index=False)

def evaluate():
    all_inputs, all_preds, all_refs = _generate_preds_causal_lm()
    _write_preds(all_inputs, all_preds, all_refs)

    results = {}

    results.update(compute_bleu(all_preds, all_refs))

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
if __name__ == "__main__":
    evaluate()