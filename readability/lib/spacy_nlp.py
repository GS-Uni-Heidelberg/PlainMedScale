import spacy
from spacy.tokens import Doc, Span, Token
from spacy_syllables import SpacySyllables
from spacy.language import Language
import re


_ger_nlp_base = spacy.load("de_core_news_lg")
syllables = SpacySyllables(_ger_nlp_base)  # registers the factory automatically
_ger_nlp_base.add_pipe("syllables", after="tagger")

_eng_nlp_base = spacy.load("en_core_web_lg")
syllables = SpacySyllables(_eng_nlp_base)  # registers the factory automatically
_eng_nlp_base.add_pipe("syllables", after="tagger")


@Language.component("newline_sentencizer")
def newline_sentencizer(doc):
    if len(doc):
        doc[0].is_sent_start = True

    for i, tok in enumerate(doc[:-1]):
        # Case 1: newline token itself
        if tok.text in {"\n", "\n\n"}:
            doc[i + 1].is_sent_start = True
        # Case 2: newline appears in trailing whitespace
        elif "\n" in tok.whitespace_:
            doc[i + 1].is_sent_start = True

    return doc


_eng_nlp_base.add_pipe("newline_sentencizer", first=True)
_ger_nlp_base.add_pipe("newline_sentencizer", first=True)


def normalize_newlines(text: str) -> str:
    """
    Normalize newlines for consistent sentence splitting:
    1. Strip leading/trailing whitespace
    2. Replace any sequence of newlines with exactly double newline
    """
    # Strip leading/trailing whitespace
    text = text.strip()
    # Replace any sequence of 1+ newlines with exactly double newline
    text = re.sub(r"\n+", r"\n\n", text)
    return text


class NormalizedNLP:
    """Wrapper that normalizes newlines before processing."""
    def __init__(self, nlp):
        self._nlp = nlp
        self.vocab = nlp.vocab

    def __call__(self, text):
        if isinstance(text, str):
            text = normalize_newlines(text)
        return self._nlp(text)


eng_nlp = NormalizedNLP(_eng_nlp_base)
ger_nlp = NormalizedNLP(_ger_nlp_base)


def edit_md_tables(text: str) -> str:
    text = re.sub(r"(\| *-+ *)+\| *\n", r"\n", text)
    text = re.sub(r" *\| *", r"\n", text)
    return text


def collapse_newlines(text: str) -> str:
    """
    Replace multiple newlines with a single newline.
    """
    return re.sub(r"\n\n+", r"\n\n", text)


def is_symbol(token: Token) -> bool:
    """
    Check if a token contains at least one alphabetic character or is a number.
    """
    if token.pos_ == "NUM":
        return False
    for char in token.text:
        if char.isalpha():
            return False
    return True


def clean_sentence(sentence: Span) -> list[Token]:
    """
    Return tokens from a *sentence span* with punctuation/spaces removed.
    Accepts a spaCy Span (one sentence). If a Doc or generator is passed
    by accident, raise a clear error.
    """
    if isinstance(sentence, Doc):
        raise TypeError(
            "clean_sentence expects a single sentence (Span), not a Doc."
        )
    if isinstance(sentence, Token):
        raise TypeError(
            "clean_sentence expects a sentence (Span), not a single Token."
        )
    if not isinstance(sentence, Span):
        raise TypeError(
            "clean_sentence expects a spaCy Span (a single sentence)."
        )
    return [
        t for t in sentence if not (
            t.is_punct or is_symbol(t)
        )
    ]


def text_len(doc: Doc) -> int:
    """
    Count non-punct, non-space tokens across all sentences.
    """
    # doc.sents is a generator; iterate over it explicitly
    return sum(len(clean_sentence(sent)) for sent in doc.sents)


if __name__ == "__main__":
    # Test how markdown tables are handled
    test_text = """This is a sentence.
    | Header 1 | Header 2 |
    |----------|----------|
    | Cell 1   | Cell 2   |
    """
    doc = eng_nlp(test_text)
    for sent in doc.sents:
        print(f"Sentence: {sent.text.strip()}")
        print(f"Cleaned tokens: {[t.text for t in clean_sentence(sent)]}")
    print(f"Total text length (non-punct tokens): {text_len(doc)}")

    # Test is_symbol()
    tokens = ["hello", "world!", "123a", "@#$%", "ÄÄÄ", "!!!test", "550"]
    for sentence in tokens:
        doc = eng_nlp(sentence)
        for token in doc:
            print(f"Token: {token.text}, is_symbol: {is_symbol(token)}")
