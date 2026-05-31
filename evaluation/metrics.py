import sacrebleu
from rouge_score import rouge_scorer

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