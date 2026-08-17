import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ..lib import metrics as m  # noqa: E402
from ..lib import spacy_nlp as nlp  # noqa: E402


def test_avg_sentence_length():
    text = "Dies ist ein einfacher Test. Dies ist der zweite Satz."
    doc = nlp.ger_nlp(text)
    assert m.avg_sentence_length(doc) == 5.0


def test_avg_syllable_length():
    text = "Dies ist ein einfacher Test."
    doc = nlp.ger_nlp(text)
    assert m.avg_syllable_length(doc) == 1.4


def test_wstf_returns_score():
    text = "Dies ist ein einfacher Test. Dies ist der zweite Satz."
    doc = nlp.ger_nlp(text)
    val = m.wstf(doc)
    assert val != -1
    assert isinstance(val, float)
    # short simple sentences should land well below upper-grade range
    assert val < 5.0


def test_wstf_orders_by_complexity():
    simple = "Hund. Katze. Maus. Vogel. Baum."
    complex_text = (
        "Mehrsilbige Fachausdrücke erhöhen die Komplexität erheblich, "
        "insbesondere wenn lange Hauptsatzkonstruktionen mit zahlreichen "
        "Nominalphrasen aneinandergereiht werden."
    )
    s_doc = nlp.ger_nlp(simple)
    c_doc = nlp.ger_nlp(complex_text)
    assert m.wstf(c_doc) > m.wstf(s_doc)


def test_wstf_empty():
    doc = nlp.ger_nlp(".")
    assert m.wstf(doc) == -1
