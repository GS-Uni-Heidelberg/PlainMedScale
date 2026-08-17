"""Single source of truth for readability/jargon metric metadata.

Reads readability/metrics_config.toml and exposes a typed list of Metric
records plus convenience lookups. Consumers should import from here rather
than re-declaring (key, title, expected_sign, ...) tuples.
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "metrics_config.toml"

VALID_DIR_KEYS = {"std", "llm", "jarg"}
VALID_SIGNS = {-1, 0, 1}


@dataclass(frozen=True)
class Metric:
    key: str
    title_en: str
    title_de: str
    dir_key: str
    expected_sign: int
    decimals: int = 2
    # Multiplier applied to the raw metric value (and its std) before it is
    # formatted into a table cell. Use e.g. 1_000_000 to render a probability
    # in PMW (per-million-words). Statistical tests use the raw value, so the
    # sign and significance markers are unaffected by `scale`.
    scale: float = 1.0


def load_metrics(path: Path = CONFIG_PATH) -> list[Metric]:
    """Load the metric config and return an ordered list of Metric records.

    Order in the returned list matches order in the TOML file, so consumers
    that want a different display order should re-sort or filter explicitly.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    out: list[Metric] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw.get("metrics", [])):
        key = entry["key"]
        if key in seen:
            raise ValueError(
                f"{path}: duplicate metric key {key!r} at entry #{i}"
            )
        seen.add(key)
        dir_key = entry["dir_key"]
        if dir_key not in VALID_DIR_KEYS:
            raise ValueError(
                f"{path}: metric {key!r}: dir_key {dir_key!r} not in "
                f"{sorted(VALID_DIR_KEYS)}"
            )
        sign = int(entry["expected_sign"])
        if sign not in VALID_SIGNS:
            raise ValueError(
                f"{path}: metric {key!r}: expected_sign {sign} not in "
                f"{sorted(VALID_SIGNS)}"
            )
        out.append(Metric(
            key=key,
            title_en=entry["title_en"],
            title_de=entry["title_de"],
            dir_key=dir_key,
            expected_sign=sign,
            decimals=int(entry.get("decimals", 2)),
            scale=float(entry.get("scale", 1.0)),
        ))
    return out


def plot_title(metric: Metric) -> str:
    """`title_en` rendered for matplotlib rather than LaTeX.

    Titles in the TOML are written LaTeX-native, because the paper tables
    emit them verbatim. Matplotlib does not read LaTeX, so escapes have to
    be undone for figures: ``--`` is a real en-dash, ``\\%`` a real percent
    sign. Every plot script should route `title_en` through here.
    """
    return metric.title_en.replace("--", "–").replace("\\%", "%")


def metrics_by_key(metrics: list[Metric] | None = None) -> dict[str, Metric]:
    """Return a key -> Metric lookup (loading if not passed in)."""
    if metrics is None:
        metrics = load_metrics()
    return {m.key: m for m in metrics}


def titles_de(metrics: list[Metric] | None = None) -> dict[str, str]:
    """key -> German title lookup, for graphs.ipynb's `transl` dict."""
    return {m.key: m.title_de for m in (metrics or load_metrics())}


def titles_en(metrics: list[Metric] | None = None) -> dict[str, str]:
    """key -> English title lookup, for graphs.ipynb's `transl` dict."""
    return {m.key: m.title_en for m in (metrics or load_metrics())}


# Spelled-out, human-readable source names for figures (boxplot tick labels).
# Compact table variants live in make_metrics_table.py / stats_*.py; these are
# the fully-spelled-out labels meant for plot axes and legends.
SOURCE_LABELS = {
    "apoum": "Apotheken\nUmschau",
    "nhs": "NHS",
    "gesund": "gesund.\nbund",
    "msd_short": "MSD Manual\n(Quick Facts)",
    "msd_lay": "MSD\nConsumer",
    "msd_expert": "MSD Manual\n(Professional)",
}


def source_labels(keys: list[str] | None = None) -> list[str] | dict[str, str]:
    """Human-readable source names for figures.

    Called with a list of source keys, returns the labels in the same order
    (ready for boxplot ``tick_labels``). Called with no argument, returns the
    full key -> label lookup. Unknown keys fall back to the key itself.
    """
    if keys is None:
        return dict(SOURCE_LABELS)
    return [SOURCE_LABELS.get(k, k) for k in keys]
