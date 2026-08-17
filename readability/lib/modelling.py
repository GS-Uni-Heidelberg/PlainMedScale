from transformers import (
    AutoTokenizer,
    BertForNextSentencePrediction,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)
import torch
from spacy.tokens import Doc
import random
import statistics
import math


# Detect Apple Silicon MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def ansp(doc: Doc, tokenizer, model, device) -> float:
    sents = [s.text.strip() for s in doc.sents]
    if len(sents) < 2:
        return -1

    hits = 0
    for i in range(len(sents) - 1):
        enc = tokenizer(sents[i], sents[i+1], return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.softmax(dim=-1)[0]
        hits += float(logits[0] > logits[1])  # 0=IsNext, 1=NotNext

    return hits / (len(sents) - 1)


def appl(
    doc: Doc, tokenizer, model, device,
    min_length=3
) -> float:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sents = [s.text.strip() for s in doc.sents]
    if not sents:
        return -1

    ppl_vals = []
    with torch.no_grad():
        for s in sents:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=512)


            num_tokens = enc["input_ids"].size(1)
            if num_tokens < min_length:
                continue

            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            ppl_val = torch.exp(out.loss).item()
            ppl_vals.append(ppl_val)

    if not ppl_vals:
        return -1

    return statistics.mean(ppl_vals)


def appl_experiments(
    doc: Doc, tokenizer, model, device,
) -> float:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sents = [s.text.strip() for s in doc.sents]
    if not sents:
        return -1, -1, -1, -1

    ppl_vals = []
    ppl_vals_full = []
    with torch.no_grad():
        for s in sents:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=512)

            num_tokens = enc["input_ids"].size(1)
            if num_tokens < 1:
                continue

            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            ppl_val = torch.exp(out.loss).item()
            
            ppl_vals_full.append(ppl_val)
            if num_tokens >= 3:
                ppl_vals.append(ppl_val)

    if not ppl_vals:
        if ppl_vals_full:
            return -1, -1, statistics.mean(ppl_vals_full), statistics.median(ppl_vals_full)
        return -1, -1, -1, -1
    return statistics.mean(ppl_vals), statistics.median(ppl_vals), statistics.mean(ppl_vals_full), statistics.median(ppl_vals_full)


def appl_fulltext(
    doc: Doc, tokenizer, model, device,
) -> float:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = doc.text.strip()
    if not text.strip():
        return -1.0

    enc_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n = enc_ids.size(1)
    if n <= 1:
        return -1.0

    model.eval()
    nll_sum = 0.0
    tok_count = 0

    with torch.no_grad():
        # non-overlapping windows sized to context
        for start in range(0, n, 512):
            end = min(start + 512, n)
            input_ids = enc_ids[:, start:end]
            if input_ids.size(1) <= 1:
                continue
            out = model(input_ids, labels=input_ids)
            # out.loss is mean CE over (L-1) tokens; weight by token count
            tokens_here = input_ids.size(1) - 1
            nll_sum += float(out.loss) * tokens_here
            tok_count += tokens_here

    if tok_count == 0:
        return -1.0
    ce = nll_sum / tok_count
    ppl = float(torch.exp(torch.tensor(ce)))
    return ppl if math.isfinite(ppl) else -1.0


def lmfm(doc: Doc, tokenizer, model, device, mask_ratio=0.15, seed=0) -> float:

    randomizer = random.Random(seed)
    sents = [s.text.strip() for s in doc.sents]
    scores, V = [], model.get_output_embeddings().out_features

    with torch.no_grad():
        for s in sents:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=512)
            ids = enc["input_ids"].to(device)
            cand = [i for i in range(1, ids.size(1)-1)]
            k = max(1, round(mask_ratio * len(cand))) if cand else 0

            for i in randomizer.sample(cand, k):
                orig = ids[0, i].item()
                masked = ids.clone()
                masked[0, i] = tokenizer.mask_token_id
                logits = model(masked).logits[0, i]
                rank = torch.argsort(logits, descending=True).tolist().index(orig) + 1
                scores.append(rank / V)

    return sum(scores) / len(scores) if scores else -1


text_en = "This is absolutely crazy. Like, it is insane."
text_de = "Katzen jagen Laser. In der Ukraine gibt es Krieg. Kaum zu glauben, was Timo da macht."


if __name__ == "__main__":
    import spacy
    nlp_en = spacy.load("en_core_web_lg")
    nlp_de = spacy.load("de_core_news_lg")

    doc_en = nlp_en(text_en)
    doc_de = nlp_de(text_de)

    print("ANSP EN:", ansp(doc_en, "bert-base-uncased"))
    print("ANSP DE:", ansp(doc_de, "bert-base-german-cased"))

    print("APPL EN:", appl(doc_en, "gpt2"))
    print("APPL DE:", appl(doc_de, "dbmdz/german-gpt2"))

    print("LMFM EN:", lmfm(doc_en, "bert-base-uncased"))
    print("LMFM DE:", lmfm(doc_de, "bert-base-german-cased"))
