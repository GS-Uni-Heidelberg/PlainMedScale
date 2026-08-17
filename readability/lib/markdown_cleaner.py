from __future__ import annotations
from typing import List
import re
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin


MATH_PATTERNS = [
    re.compile(r"\$\$(?:.|\n)*?\$\$", re.MULTILINE),     # $$ block math $$
    re.compile(r"(?<!\$)\$(?:\\.|[^$\n])+\$(?!\$)"),     # $inline math$
    re.compile(r"\\\((?:.|\n)*?\\\)", re.MULTILINE),     # \( inline math \)
    re.compile(r"\\\[(?:.|\n)*?\\\]", re.MULTILINE),     # \[ display math \]
]


def _strip_math(text: str) -> str:
    for pat in MATH_PATTERNS:
        text = pat.sub("", text)
    return text


def markdown_to_text(md_string: str) -> str:
    """
    Convert Markdown to plain text:
      - links -> keep anchor text only
      - images -> drop entirely
      - math -> drop entirely ($...$, $$...$$, \(..\), \[..])
      - code -> keep contents
      - blockquotes -> flattened as normal text
      - footnotes -> collected and appended at bottom
    """
    md = MarkdownIt().use(footnote_plugin).enable("strikethrough")
    tokens = md.parse(md_string)

    body_parts: List[str] = []
    footnotes: List[str] = []

    # helper to extract text from a token slice
    def extract_text(from_idx: int, to_idx: int, into: List[str]) -> None:
        i = from_idx
        while i < to_idx:
            tok = tokens[i]

            # Flatten paragraphs/lists/headings/quotes with newlines around
            if tok.type in {"paragraph_open", "list_item_open", "heading_open", "blockquote_open"}:
                into.append("\n")
            elif tok.type in {"paragraph_close", "list_item_close", "heading_close", "blockquote_close"}:
                into.append("\n")

            # Inline container -> recurse into children
            elif tok.type == "inline":
                for child in tok.children or []:
                    # plain text
                    if child.type == "text":
                        into.append(child.content)
                    # emphasis/strong/etc. -> children already yield text
                    elif child.type in {"em_open", "em_close", "strong_open", "strong_close",
                                        "s_open", "s_close", "code_inline_close", "code_inline_open"}:
                        continue
                    # code inline -> keep content
                    elif child.type == "code_inline":
                        into.append(child.content)
                    # link -> keep only child text (ignore href)
                    elif child.type in {"link_open", "link_close"}:
                        continue
                    # images -> drop entirely
                    elif child.type == "image":
                        continue
                    # footnote references -> drop inline marker
                    elif child.type in {"footnote_ref", "footnote_reference_open", "footnote_reference_close"}:
                        continue
                    else:
                        # fallback: take text-y content if present
                        if getattr(child, "content", "") and child.type == "text":
                            into.append(child.content)

            # Fenced/indented code blocks -> keep content
            elif tok.type in {"code_block", "fence"}:
                into.append(tok.content)

            # Lists: add lightweight separation without bullets
            elif tok.type in {"bullet_list_open", "ordered_list_open"}:
                into.append("\n")
            elif tok.type in {"bullet_list_close", "ordered_list_close"}:
                into.append("\n")

            # Images at block level (rare) -> drop
            elif tok.type == "image":
                pass

            # Tables -> drop entirely
            elif tok.type.startswith("table") or tok.type in {
                "thead_open", "thead_close",
                "tbody_open", "tbody_close",
                "tr_open", "tr_close",
                "th_open", "th_close",
                "td_open", "td_close",
            }:
                # skip everything inside table until table_close
                while i < to_idx and tokens[i].type != "table_close":
                    i += 1

            # Footnotes: skip collecting into body here; we'll harvest separately
            elif tok.type == "footnote_block_open":
                # Collect all footnotes
                i += 1
                while i < to_idx and tokens[i].type != "footnote_block_close":
                    if tokens[i].type == "footnote_open":
                        # gather everything until footnote_close into a temp buffer
                        j = i + 1
                        temp: List[str] = []
                        while j < to_idx and tokens[j].type != "footnote_close":
                            # reuse the same extraction logic on this slice (but ignoring nested footnote blocks)
                            if tokens[j].type == "inline":
                                for ch in tokens[j].children or []:
                                    if ch.type == "text":
                                        temp.append(ch.content)
                                    elif ch.type == "code_inline":
                                        temp.append(ch.content)
                                    elif ch.type == "image":
                                        pass
                                    elif ch.type in {"link_open", "link_close",
                                                     "footnote_ref", "footnote_reference_open", "footnote_reference_close"}:
                                        pass
                                    else:
                                        if getattr(ch, "content", "") and ch.type == "text":
                                            temp.append(ch.content)
                            elif tokens[j].type in {"paragraph_open", "list_item_open", "heading_open"}:
                                temp.append("\n")
                            elif tokens[j].type in {"paragraph_close", "list_item_close", "heading_close"}:
                                temp.append("\n")
                            elif tokens[j].type in {"code_block", "fence"}:
                                temp.append(tokens[j].content)
                            j += 1
                        footnote_txt = " ".join(temp)
                        # whitespace tidy within this footnote
                        footnote_txt = "\n".join(
                            ln.strip() for ln in footnote_txt.splitlines() if ln.strip()
                        ).strip()
                        if footnote_txt:
                            footnotes.append(footnote_txt)
                        i = j  # position on footnote_close; will be incremented by loop
                    i += 1
                # footnote_block_close handled by loop; do not add to body
            # Everything else: ignore
            i += 1

    # Extract body text excluding the footnote block
    extract_text(0, len(tokens), body_parts)

    # Join and tidy whitespace
    body = " ".join(body_parts)
    # Remove math (inline & block) from body
    body = _strip_math(body)
    # Normalize lines
    body = "\n\n".join(ln.strip() for ln in body.splitlines() if ln.strip())
    body = re.sub(r"[ \t]{2,}", " ", body).strip()

    # Prepare footnotes (math stripped as well)
    cleaned_footnotes = []
    for fn in footnotes:
        txt = _strip_math(fn)
        txt = re.sub(r"[ \t]{2,}", " ", txt).strip()
        if txt:
            cleaned_footnotes.append(txt)

    if cleaned_footnotes:
        footer_lines = [""]
        for _, txt in enumerate(cleaned_footnotes, start=1):  # Decided agains footnote numbering
            footer_lines.append(f"{txt}")
        body = body + "\n\n" + "\n\n".join(footer_lines)

    return body


example = """
# Sample Document Title

Some **bold** text, some *italic* text, and some ~~struck~~ text.
Inline code like `print("hello")` should remain as text.

A regular link with anchor: [example anchor](https://example.com)
A bare autolink that should be dropped by URL-stripping rules: [link](https://barelink.example)

An image that should be removed entirely:
![diagram of something](https://images.example.com/diagram.png)

A horizontal rule follows:

---

## Lists

* First bullet item with a [link text only](https://link.to/drop).
* Second bullet item with an image that should vanish: ![alt text](https://img.invalid/x.png)

  * Nested bullet item with *emphasis*.

1. First ordered item
2. Second ordered item with inline code `x = 42`

> This is a blockquote. Treat it as normal text and keep only readable content (like this [quoted link](https://quoted.example)).

## Code Blocks

```
def add(a, b):
    return a + b

print(add(2, 3))
```

And another indented block that mimics shell:

```
$ echo "this is shell-ish text"
```

## Tables

| Term     | Definition                     |
| -------- | ------------------------------ |
| Polyp    | Abnormal tissue growth [^poly] |
| Anamnese | Medical history-taking process |

## Math (should be removed)

Inline math like $x^2 + y^2 = r^2$ should disappear.
Display math should also vanish:

$$
\int_{0}^{1} x^2 , dx = \frac{1}{3}
$$

And LaTeX-style forms should go too: ( a^2 + b^2 ) and [ E = mc^2 ].

## Footnotes

Here is a claim that needs a note.[^1] Another claim with a named note.[^refnote]

Final paragraph with a link whose URL should be dropped but anchor retained: [anchor to keep](https://drop.this.url).

[^1]: This is the first footnote. It contains a URL [keep_anchor](https://footnote.example/keep-anchor-drop-url) and some math $a+b$ that should be removed.

[^refnote]: Second footnote with *emphasis*, an image ![x](https://x.invalid), and a code span `code`.

[^poly]: Footnote used in the table row. Contains a reference link: [see details](https://example.org/polyp).

---

You can now copy everything above as a single block of text; it will remain valid Markdown, no nested code-fence collapse. Perfect for your test corpus.
"""


if __name__ == "__main__":
    print(markdown_to_text(example))
