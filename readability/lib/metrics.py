from .spacy_nlp import clean_sentence, text_len
from spacy.tokens import Doc
import editdistance
from wordfreq import word_frequency
from collections import defaultdict
from itertools import combinations
import tiktoken
import math
import numpy as np


def avg_sentence_length(doc: Doc) -> float:
    sents = [clean_sentence(sent) for sent in doc.sents]
    sents = [s for s in sents if s]  # filter out empty sentences
    if not sents:
        return -1
    return text_len(doc) / len(sents)


def avg_syllable_length(doc: Doc) -> float:
    words = [
        t for sent in doc.sents for t in clean_sentence(sent)
    ]
    if not words:
        return -1

    total_syllables = 0
    for t in words:
        count = getattr(t._, "syllables_count", None)
        if count is None:
            count = 0
        total_syllables += count

    return total_syllables / len(words)


def fkgl(doc: Doc) -> float:
    asl = avg_sentence_length(doc)
    asyl = avg_syllable_length(doc)
    if asl < 0 or asyl < 0:
        return -1
    return 0.39 * asl + 11.8 * asyl - 15.59


def wstf(doc: Doc) -> float:
    """Wiener Sachtextformel variant 1 (Bamberger & Vanecek, 1984).

    Output ≈ German school grade level (~4–15). The constants are
    calibrated for German; running this on English text produces values
    on the same scale but their calibration is not linguistically
    meaningful — keep that caveat when comparing DE vs EN.
    """
    sents = [clean_sentence(s) for s in doc.sents]
    sents = [s for s in sents if s]
    if not sents:
        return -1

    words = [t for s in sents for t in s]
    n_words = len(words)
    if n_words == 0:
        return -1

    sl = n_words / len(sents)

    n_multisyll = 0
    n_monosyll = 0
    n_long = 0
    for t in words:
        c = getattr(t._, "syllables_count", None) or 0
        if c >= 3:
            n_multisyll += 1
        elif c == 1:
            n_monosyll += 1
        if len(t.text) > 6:
            n_long += 1

    ms = 100.0 * n_multisyll / n_words
    es = 100.0 * n_monosyll / n_words
    iw = 100.0 * n_long / n_words

    return 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875


def noun_ratio(doc: Doc) -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    nouns = [t for t in words if t.pos_ in {"NOUN", "PROPN"}]
    return len(nouns) / len(words)


def adjective_ratio(doc: Doc) -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    adjs = [t for t in words if t.pos_ == "ADJ"]
    return len(adjs) / len(words)


def verb_ratio(doc: Doc) -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    verbs = [t for t in words if t.pos_ in {"VERB", "AUX"}]
    return len(verbs) / len(words)


_FUNCTION_WORD_POS = {
    "de": {"AUX", "DET", "ADP", "VMFIN", "VMINF", "VMPP", "PWS", "PWAT", "PWAV"},
    "en": {"AUX", "DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"},
}


def function_word_ratio(doc: Doc, lang: str = "de") -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    pos_set = _FUNCTION_WORD_POS[lang]
    func_words = [t for t in words if t.pos_ not in pos_set]
    return len(func_words) / len(words)


def numbers_ratio(doc: Doc) -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    nums = [t for t in words if t.pos_ == "NUM"]
    return len(nums) / len(words)


_NEGATION_LEMMAS = {
    "de": {
        "nicht", "kein", "nie", "nein", "keineswegs",
        "keinesfalls", "nichts", "nirgends", "nimmer",
        "niemand", "nirgendwo", "niemals",
    },
    "en": {
        "not", "no", "never", "neither", "nor", "nobody",
        "nothing", "nowhere", "none",
    },
}


def negations_ratio(doc: Doc, lang: str = "de") -> float:
    words = [t for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    lemma_set = _NEGATION_LEMMAS[lang]
    negs = [t for t in words if t.lemma_ in lemma_set]
    return len(negs) / len(words)


def grammar_frequency(doc: Doc) -> float:
    """Ratio of unique sentence structures to all sentences in the doc."""
    unique_sentence_structures = set()
    sent_count = 0
    for sent in doc.sents:
        pos_tuple = tuple(t.pos_ for t in clean_sentence(sent))
        if pos_tuple:
            unique_sentence_structures.add(pos_tuple)
            sent_count += 1
    if sent_count == 0:
        return -1
    return len(unique_sentence_structures) / sent_count


def adjacent_sent_edit_distance(doc: Doc) -> float:
    """Average edit distance between adjacent sentences in the doc,
    where the sentence is represented as a sequence of POS tags.
    """
    sents = [
        [token.pos_ for token in clean_sentence(sent)] for sent in doc.sents
    ]
    sents = [s for s in sents if s]  # filter out empty sentences
    if len(sents) < 2:
        return -1
    total_distance = 0
    count = 0
    for i in range(len(sents) - 1):
        dist = editdistance.eval(sents[i], sents[i + 1])
        try:
            normalized_distance = dist / max(len(sents[i]), len(sents[i + 1]))
        except ZeroDivisionError:
            print(doc.text)
            raise ZeroDivisionError
        total_distance += normalized_distance
        count += 1
    if count == 0:
        return -1
    return total_distance / count


def dep_tree_depth(doc: Doc) -> float:
    """Average dependency tree depth across all sentences in the doc."""

    def _tree_depth(node) -> int:
        # consider only non-punctuation children
        kids = [c for c in node.children if not c.is_punct]
        if not kids:
            return 0
        return 1 + max(_tree_depth(c) for c in kids)

    depths = []
    for sent in doc.sents:
        root = [token for token in sent if token.head == token][0]
        depths.append(_tree_depth(root))
    if not depths:
        return -1
    return sum(depths) / len(depths)


def noun_phrase_complexity(doc: Doc) -> float:
    """Average number of words per noun phrase in the doc."""
    nps = list(doc.noun_chunks)
    if not nps:
        return -1
    total_words = sum(len(clean_sentence(np)) for np in nps)
    return total_words / len(nps)


def word_freq(doc: Doc, lang='de') -> float:
    words = [t.text for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    total_freq = sum(word_frequency(word, lang) for word in words)
    return total_freq / len(words)


def word_freq_filtered(doc: Doc, lang='de') -> float:
    words = [
        t.text for t in doc if not t.is_stop and t.is_alpha
    ]
    if not words:
        return -1
    total_freq = sum(word_frequency(word, lang) for word in words)
    return total_freq / len(words)


def lexical_chain_lens(doc: Doc) -> float:
    nouns = {}
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = token.lemma_
            nouns[lemma] = nouns.get(lemma, 0) + 1
    chained_nouns = {
        lemma: count for lemma, count in nouns.items() if count > 1
    }
    if not chained_nouns:
        return -1

    normalization = 1 / text_len(doc)
    avg_len = sum(chained_nouns.values()) / len(chained_nouns)
    return avg_len * normalization


def crossing_lexical_chains(doc: Doc) -> float:

    # collect positions per noun/proper-noun lemma
    pos_by_lemma = defaultdict(list)
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            pos_by_lemma[tok.lemma_].append(tok.i)

    # keep chains, drop singletons
    chains = {lem: sorted(pos) for lem, pos in pos_by_lemma.items()
              if len(pos) > 1}
    items = list(chains.items())

    crossings = 0
    for (lemA, posA), (lemB, posB) in combinations(items, 2):
        startA, endA = posA[0], posA[-1]
        startB, endB = posB[0], posB[-1]
        if startA <= endB and startB <= endA:
            crossings += 1

    return crossings / text_len(doc)


# +++++ OWN IDEAS FOR METRICS +++++ #

def bpe_sentence_length(doc: Doc, model: str = "cl100k_base") -> float:
    """Average sentence length in BPE tokens."""
    lens = []
    for sent in doc.sents:
        text = sent.text.strip()
        if text:
            enc = tiktoken.get_encoding(model)
            tokens = enc.encode(text)
            lens.append(len(tokens))
    if not lens:
        return -1
    return sum(lens) / len(lens)


def bpe_token_length(doc: Doc, model: str = "cl100k_base") -> float:
    """Average token length in BPE tokens."""
    lens = []
    for sent in doc.sents:
        text = sent.text.strip()
        if text:
            enc = tiktoken.get_encoding(model)
            tokens = enc.encode(text)
            lens.append(len(tokens) / len(sent))
    if not lens:
        return -1
    return sum(lens) / len(lens)


def bpe_char_ratio(doc: Doc, model: str = "cl100k_base") -> float:
    """Average ratio of characters to BPE tokens"""

    raw_text = doc.text.strip()
    if not raw_text:
        return -1
    enc = tiktoken.get_encoding(model)
    total_bpe_tokens = len(enc.encode(raw_text))
    if total_bpe_tokens == 0:
        return -1
    return len(raw_text) / total_bpe_tokens


def bpe_char_ratio_lex(doc: Doc, model: str = "cl100k_base") -> float:
    """Average ratio of characters to BPE tokens,
    but only considering lexical (non-stop) words."""

    words = [t.text for t in doc if not t.is_stop and not t.is_alpha]
    if not words:
        return -1
    raw_text = " ".join(words).strip()
    if not raw_text:
        return -1
    enc = tiktoken.get_encoding(model)
    total_bpe_tokens = len(enc.encode(raw_text))
    if total_bpe_tokens == 0:
        return -1
    return len(raw_text) / total_bpe_tokens


def type_token_ratio(doc: Doc) -> float:
    words = [t.lemma_ for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    types = set(words)
    return len(types) / len(words)


def mattr(doc: Doc, window: int = 100) -> float:
    """
    Moving-Average Type-Token Ratio (Covington & McFall, 2010).
    Mirrors `type_token_ratio`'s token set (lemmas of cleaned sentence tokens).
    If the text is shorter than `window`, returns plain TTR over all tokens.
    """
    words = [t.lemma_ for sent in doc.sents for t in clean_sentence(sent)]
    if not words:
        return -1
    if len(words) <= window:
        return len(set(words)) / len(words)
    n_windows = len(words) - window + 1
    total = 0.0
    for i in range(n_windows):
        total += len(set(words[i:i + window])) / window
    return total / n_windows


def herdans_c(doc: Doc) -> float:
    """
    Returns log-corrected TTR (Herdan's C) on content words only
    """
    words = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    if not words:
        return -1

    types = set(words)
    n_types = len(types)
    n_tokens = len(words)

    # Herdan's C: log(V) / log(N) — normalizes for length
    log_ttr = math.log(n_types) / math.log(n_tokens) if n_tokens > 1 else 0.0

    return log_ttr


def _get_tokens(doc: Doc, content_only: bool = False) -> list[str]:
    if content_only:
        return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
    return [t.lemma_ for t in doc if t.is_alpha]


def _fit_heaps_beta(tokens: list[str], tail_start: int = 400) -> float:
    N_total = len(tokens)
    max_points = 600
    step = max(1, N_total // max_points)

    seen = set()
    Ns, Vs = [], []
    for i in range(step, N_total + 1, step):
        seen.update(tokens[i - step : i])
        Ns.append(i)
        Vs.append(len(seen))

    Ns = np.asarray(Ns, float)
    Vs = np.asarray(Vs, float)

    mask = (Ns >= tail_start) & (Vs > 0)
    if mask.sum() < 2:
        return -1

    x = np.log(Ns[mask])
    y = np.log(Vs[mask])

    weights = Ns[mask] / Ns[mask].sum()
    a, _ = np.polyfit(x, y, 1, w=weights)

    beta = float(a)
    return beta if math.isfinite(beta) else -1


def heaps_law_beta(doc: Doc) -> float:
    tokens = _get_tokens(doc, content_only=False)
    if len(tokens) < 500:
        return -1
    return _fit_heaps_beta(tokens, tail_start=400)


def heaps_law_beta_lex(doc: Doc) -> float:
    tokens = _get_tokens(doc, content_only=True)
    if len(tokens) < 200:
        return -1
    return _fit_heaps_beta(tokens, tail_start=100)


def text_length(doc: Doc) -> int:
    return len(_get_tokens(doc, content_only=False))


def text_length_lex(doc: Doc) -> int:
    return len(_get_tokens(doc, content_only=True))


def text_length_sentences(doc: Doc) -> int:
    return len(list(doc.sents))
