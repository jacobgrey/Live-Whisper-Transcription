"""Per-config metrics + summary.md generator for the diarization harness."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigMetrics:
    name: str
    distinct_speakers: int = 0
    total_turns: int = 0
    top2_word_share: float = 0.0
    mid_sentence_swaps: int = 0
    runtime_sec: float = 0.0
    error: str | None = None


_END_OF_SENTENCE = re.compile(r"[.?!\"')\]]\s*$")


def compute_metrics(jsonl_path: Path, name: str, runtime_sec: float) -> ConfigMetrics:
    """Read a {start, end, speaker, text} JSONL and compute the comparison
    metrics defined in the plan. Robust to a missing/empty file."""
    m = ConfigMetrics(name=name, runtime_sec=runtime_sec)
    if not jsonl_path.exists():
        m.error = "no jsonl produced"
        return m

    turns: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not turns:
        m.error = "no turns"
        return m

    # distinct_speakers, total_turns
    speakers = [t["speaker"] for t in turns]
    m.distinct_speakers = len(set(speakers))
    m.total_turns = len(turns)

    # top2_word_share
    word_count_by_spk: dict[str, int] = {}
    total_words = 0
    for t in turns:
        n = len(str(t.get("text", "")).split())
        word_count_by_spk[t["speaker"]] = word_count_by_spk.get(t["speaker"], 0) + n
        total_words += n
    if total_words > 0:
        top2 = sorted(word_count_by_spk.values(), reverse=True)[:2]
        m.top2_word_share = sum(top2) / total_words

    # mid_sentence_swaps: previous turn doesn't end in .?! and gap < 0.4s
    swaps = 0
    for prev, cur in zip(turns, turns[1:]):
        if prev["speaker"] == cur["speaker"]:
            continue
        prev_text = str(prev.get("text", ""))
        if _END_OF_SENTENCE.search(prev_text):
            continue
        gap = float(cur.get("start", 0)) - float(prev.get("end", 0))
        if gap < 0.4:
            swaps += 1
    m.mid_sentence_swaps = swaps

    return m


def write_summary(out_path: Path, metrics: list[ConfigMetrics], num_speakers: int, source: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Diarization experiment summary")
    lines.append("")
    lines.append(f"Source: `{source}`")
    lines.append(f"Target speaker count: **{num_speakers}**")
    lines.append("")
    lines.append("Lower `mid_sentence_swaps` is better. `top2_word_share` should approach")
    lines.append(f"~{1 / max(num_speakers, 1) * 2:.2f} for {num_speakers} balanced speakers; values near")
    lines.append("1.0 indicate cluster collapse onto two dominant bins.")
    lines.append("")
    lines.append("| config | distinct_speakers | total_turns | top2_word_share | mid_sentence_swaps | runtime_sec | error |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in metrics:
        err = m.error or ""
        lines.append(
            f"| `{m.name}` | {m.distinct_speakers} | {m.total_turns} "
            f"| {m.top2_word_share:.2f} | {m.mid_sentence_swaps} | {m.runtime_sec:.1f} | {err} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
