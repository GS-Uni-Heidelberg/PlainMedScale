import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ..lib import metrics as m  # noqa: E402
from ..lib import spacy_nlp as nlp  # noqa: E402


example_text1 = """Dies ist ein einfacher Text.
    Dies ist ein einfacher Text.
    Dies ist ein einfacher Text.
"""

example_text2 = """Dies ist ein komplizierterer Text.
    Dies ist ein schwierigerer Text.
    Dies ist ein Text.
"""

cross_cutting_text1 = """Dies ist ein Text.
    Dies ist ein Satz."
    Dies ist ein Text.
    Und ein weiterer Satz.
"""

cross_cutting_text2 = """Dies ist ein Text.
    Dies ist ein Satz."
    Und ein weiterer Satz.
    Du bist ein Schatz.
    Dies ist ein Text.
    Es ist ein Schatz.
"""


doc1 = nlp.ger_nlp(example_text1)
doc2 = nlp.ger_nlp(example_text2)


def test_word_freq():
    assert m.word_freq(doc1) > 0.0001
    assert m.word_freq(doc2) > 0.0001
    assert m.word_freq(doc1) < m.word_freq(doc2)


def test_lexical_chain_lens():
    assert m.lexical_chain_lens(doc1) == 0.2


def test_cross_cutting_chains():
    cross_doc1 = nlp.ger_nlp(cross_cutting_text1)
    cross_doc2 = nlp.ger_nlp(cross_cutting_text2)
    assert m.crossing_lexical_chains(cross_doc1) == 1 / 16
    assert m.crossing_lexical_chains(cross_doc2) == 2 / 24
