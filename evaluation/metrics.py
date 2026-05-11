import sacrebleu

def compute_bleu(preds, refs):
    scores = {
        "bleu": sacrebleu.corpus_bleu(preds, [refs]).score
    }

    return scores