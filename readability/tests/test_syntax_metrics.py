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

example_deep_text = """Dies ist ein komplizierterer Text, der mehrere Ebenen
    in der Abhängigkeitsstruktur aufweist, um die Tiefe des Baums zu testen,
    was für die Analyse der syntaktischen Komplexität von Bedeutung ist.
"""


doc1 = nlp.ger_nlp(example_text1)
doc2 = nlp.ger_nlp(example_text2)


def test_grammar_frequency():
    assert m.grammar_frequency(doc1) == 1 / 3
    assert m.grammar_frequency(doc2) == 2 / 3


def test_adjsent_edit_distance():
    assert m.adjacent_sent_edit_distance(doc1) == 0.0
    assert m.adjacent_sent_edit_distance(doc2) < 0.15


def test_tree_depth():
    assert m.dep_tree_depth(doc1) == 2.0
    deep_doc = nlp.ger_nlp(example_deep_text)
    assert m.dep_tree_depth(deep_doc) == 9.0


def test_noun_phrase_complexity():
    assert m.noun_phrase_complexity(doc1) == 2.0
    assert m.noun_phrase_complexity(doc2) == 11 / 6
