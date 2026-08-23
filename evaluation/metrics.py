import sacrebleu
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
from bert_score import score

def compute_bleu(preds, refs):
    return {
        "bleu": sacrebleu.corpus_bleu(preds, [refs]).score
    }


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0

    n = len(preds)

    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)

        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure

    return {
        "rouge1": rouge1 / n,
        "rouge2": rouge2 / n,
        "rougeL": rougeL / n,
    }

def compute_comet(sources, preds, refs, batch_size=32, gpus=1):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [
        {
            "src": src,
            "mt": pred,
            "ref": ref,
        }
        for src, pred, ref in zip(sources, preds, refs)
    ]

    result = model.predict(
        data,
        batch_size=batch_size,
        gpus=gpus,
    )

    return {
        "comet": result.system_score
    }

def compute_bert(preds, refs, batch_size=8):
    _, _, f1 = score(
        preds,
        refs,
        model_type="xlm-roberta-large",
        batch_size=batch_size,
        device="cuda",
    )

    return {
        "bert_score": f1.mean().item()
    }