"""Append a per-invocation run-stamp to <out_dir>/INFO.txt.

Each pipeline call appends one section so the file accumulates the actual
history of how a data dir got to its current state — useful when auditing
TSVs on the cluster without git/Slack context.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_info(repo_root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except Exception:
        sha, dirty = "unknown", False
    return sha, dirty


def _file_stamp(p: Path) -> str:
    try:
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        return f"{p} (mtime={mtime}, size={st.st_size})"
    except FileNotFoundError:
        return f"{p} (missing)"


def _file_sha(p: Path) -> str | None:
    """Last git commit that touched a tracked file. Returns None on miss."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(p)],
            stderr=subprocess.DEVNULL, text=True, cwd=p.parent,
        ).strip() or None
    except Exception:
        return None


def write_info(
    out_dir: Path | str,
    *,
    pipeline: str,
    inputs: list[Path] | None = None,
    extra: dict | None = None,
) -> None:
    """Append a run-stamp section to <out_dir>/INFO.txt.

    Call once per pipeline invocation, after the output dir exists.
    `inputs` are stamped with mtime+size; `extra` is rendered as key/value
    lines (or bullet lists for sequence values).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    sha, dirty = _git_info(repo_root)

    lines: list[str] = ["=" * 60]
    lines.append(f"pipeline:    {pipeline}")
    lines.append(f"timestamp:   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"hostname:    {socket.gethostname()}")
    if jid := os.environ.get("SLURM_JOB_ID"):
        lines.append(f"slurm_job:   {jid}")
    lines.append(f"git_sha:     {sha}{' (dirty)' if dirty else ''}")
    lines.append(f"cwd:         {Path.cwd()}")
    lines.append(f"argv:        {' '.join(sys.argv)}")

    if inputs:
        lines.append("inputs:")
        for p in inputs:
            lines.append(f"  - {_file_stamp(Path(p))}")

    if extra:
        for k, v in extra.items():
            if isinstance(v, (list, tuple)):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")

    lines.append("")
    (out_dir / "INFO.txt").open("a", encoding="utf-8").write("\n".join(lines) + "\n")
