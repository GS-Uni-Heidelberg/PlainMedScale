"""Load the PlainMedScale corpus in its released (Zenodo) shape.

The release ships one flat JSON per (source, language), with MSD additionally
split by subtree. Every record is a dict with at least:

    source, language, instance_id, article_id, title, subtree (MSD only),
    paragraphs [{title, text}, ...], plain_text, external_ids, alignments,
    metrics

A `Split` is one such file: a (source, subtree, language) view of the corpus,
which is also the unit the metric pipelines write one TSV for.

Usage:
    from lib.corpus import SPLITS, load, iter_texts

    for split in SPLITS:
        records = load(release_dir, split)
        for text, article_id in iter_texts(records):
            ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Split:
    """One (source, subtree, language) view of the corpus."""

    source: str
    subtree: str
    language: str
    filename: str
    stem: str  # output TSV stem, e.g. "de_msd_expert"

    def path(self, release_dir: str | Path) -> Path:
        return Path(release_dir) / self.filename


# The full corpus, in the order the metric TSVs were originally written.
# `stem` keeps the historical TSV names (msd professional -> "expert",
# msd amateur -> "lay") so existing outputs stay addressable.
SPLITS: list[Split] = [
    Split("msd", "professional", "en", "msd_professional_en.json", "en_msd_expert"),
    Split("msd", "professional", "de", "msd_professional_de.json", "de_msd_expert"),
    Split("msd", "amateur", "en", "msd_amateur_en.json", "en_msd_lay"),
    Split("msd", "amateur", "de", "msd_amateur_de.json", "de_msd_lay"),
    Split("msd", "short", "en", "msd_short_en.json", "en_msd_short"),
    Split("msd", "short", "de", "msd_short_de.json", "de_msd_short"),
    Split("gesund", "amateur", "de", "gesund_bund_de.json", "de_gesund"),
    Split("gesund", "amateur", "en", "gesund_bund_en.json", "en_gesund"),
    Split("apoum", "amateur", "de", "apoum_de.json", "de_apoum"),
    Split("nhs", "amateur", "en", "nhs_en.json", "en_nhs"),
]


def splits(source: str | None = None, subtree: str | None = None,
           language: str | None = None) -> list[Split]:
    """SPLITS filtered by any combination of source / subtree / language."""
    return [s for s in SPLITS
            if (source is None or s.source == source)
            and (subtree is None or s.subtree == subtree)
            and (language is None or s.language == language)]


def split_for(source: str, subtree: str, language: str) -> Split:
    matches = splits(source, subtree, language)
    if not matches:
        raise KeyError(f"no release file for {source}/{subtree}/{language}")
    return matches[0]


def load(release_dir: str | Path, split: Split) -> list[dict]:
    """Read one release file. Records keep the release schema verbatim."""
    with split.path(release_dir).open(encoding="utf-8") as f:
        return json.load(f)


def iter_texts(records: list[dict]):
    """Yield (plain_text, article_id) for records that have text."""
    for rec in records:
        text = rec.get("plain_text")
        if not text:
            continue
        yield text, rec["article_id"]


def iter_records(release_dir: str | Path, source: str | None = None,
                 subtree: str | None = None, language: str | None = None):
    """Yield (split, record) across every matching release file that exists."""
    for split in splits(source, subtree, language):
        if not split.path(release_dir).exists():
            continue
        for rec in load(release_dir, split):
            yield split, rec


def has_text(record: dict) -> bool:
    return bool((record.get("plain_text") or "").strip())


def to_markdown(record: dict) -> str:
    """Rebuild the Markdown form: title as H1, then the paragraph blocks.

    `paragraphs[*].text` carries the original Markdown (`##` headings, `*`
    bullets, emphasis); `plain_text` is the flattened prose-only view of the
    same content. Use this where structure matters (LLM input), `plain_text`
    where it hurts (readability metrics).
    """
    parts = [f"# {record.get('title', '')}".strip()]
    for para in record.get("paragraphs") or []:
        text = (para.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def titles(release_dir: str | Path, split: Split) -> dict[str, str]:
    """{instance_id: title} for one split."""
    return {str(rec["instance_id"]): rec.get("title")
            for rec in load(release_dir, split)}


def missing_files(release_dir: str | Path) -> list[str]:
    """Release filenames that are absent from `release_dir`."""
    return [s.filename for s in SPLITS if not s.path(release_dir).exists()]
