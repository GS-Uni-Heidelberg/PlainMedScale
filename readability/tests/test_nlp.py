from spacy.tokens import Doc
from ..lib.spacy_nlp import eng_nlp as nlp
from ..lib.spacy_nlp import edit_md_tables, collapse_newlines


def sent_texts(doc):
    return [s.text.strip() for s in doc.sents]


def test_newline_sentencizer_basic_cases():
    # 1. Simple newline
    doc = nlp("Hello\nworld, e.g.")
    assert sent_texts(doc) == ["Hello", "world, e.g."]

    # 2. Sentence without punctuation before newline
    doc = nlp("This is a sentence without a period\nbut we split anyway")
    assert sent_texts(doc) == ["This is a sentence without a period", "but we split anyway"]

    # 3. Double newline (paragraph break)
    doc = nlp("Alpha\n\nBeta gamma.")
    assert sent_texts(doc) == ["Alpha", "Beta gamma."]

    # 4. Multiple stacked newlines
    doc = nlp("One\n\n\nTwo")
    assert sent_texts(doc) == ["One", "Two"]

    # 5. Leading newline(s)
    doc = nlp("\nIntro line")
    assert sent_texts(doc) == ["Intro line"]
    doc = nlp("\n\nIntro line")
    assert sent_texts(doc) == ["Intro line"]

    # 6. Trailing newline(s)
    doc = nlp("Tail line\n")
    assert sent_texts(doc) == ["Tail line"]
    doc = nlp("Tail line\n\n")
    assert sent_texts(doc) == ["Tail line"]

    # 7. Mixed punctuation around newline
    doc = nlp("First part,\nthen a continuation with comma.")
    assert sent_texts(doc) == ["First part,", "then a continuation with comma."]

    # 8. Period + newline
    doc = nlp("Das ist gut.\nAber nicht perfekt.")
    assert sent_texts(doc) == ["Das ist gut.", "Aber nicht perfekt."]

    # 9. Many internal breaks
    doc = nlp("A\nB\nC\nD")
    assert sent_texts(doc) == ["A", "B", "C", "D"]

    # 10. Lots of newlines but no empties
    doc = nlp("A\n\n\n\nB")
    assert sent_texts(doc) == ["A", "B"]


def test_newline_sentencizer_explicit_tokens():
    # "\n\n" as explicit token
    words = ["Hello", "\n\n", "world", ",", "e.g."]
    spaces = [False, False, False, True, False]  # space after comma to yield 'world, e.g.'
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    doc = nlp(doc)
    assert sent_texts(doc) == ["Hello", "world, e.g."]

    # "\n" as explicit token
    words = ["Hello", "\n", "world"]
    spaces = [False, False, False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    doc = nlp(doc)
    assert sent_texts(doc) == ["Hello", "world"]


def test_first_token_is_start():
    doc = nlp("No leading newline here.")
    assert doc[0].is_sent_start is True


def test_tables_editing():
    md_table = (
        "| Header 1 | Header 2 | h3 |\n"
        "|----------|----------| - |\n"
        "| Value 1  | Value 2  | v3 |\n"
        "| Value 4  | Value 5  | v6 |\n"
    )

    edited = edit_md_tables(md_table)
    expected = (
        "Header 1\n"
        "Header 2\n"
        "h3\n"
        "\n"
        "Value 1\n"
        "Value 2\n"
        "v3\n"
        "\n"
        "Value 4\n"
        "Value 5\n"
        "v6\n"
    )
    assert collapse_newlines(edited.strip()) == collapse_newlines(expected.strip())
