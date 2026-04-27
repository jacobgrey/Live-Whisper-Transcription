"""Diarization A/B harness — single-shot, no interactive prompts.

Usage:
    python -m experiments.run_diar_experiments <audio> --num-speakers N [opts]

Imports the daemon's pipeline functions directly so faster-whisper and
wav2vec2 load once and are reused across configs. Pyannote stays in its
existing subprocess (DLL conflict prevention). For each named config it:
  1. Runs transcribe_file_diarized (with `return_jsonl=True`) end-to-end
  2. Writes a self-describing .txt with a config header prepended
  3. Writes a .jsonl of {start, end, speaker, text} for metrics
After all configs run, writes summary.md with the comparison table.

This script never prompts. Missing required input fails fast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make src/ importable when running as a module from repo root.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from whisper_daemon import (  # noqa: E402
    DiarPipelineConfig,
    load_model,
    transcribe_file_diarized,
    get_duration,
    log,
    CHUNK_SECONDS,
)
from experiments.configs import CONFIGS  # noqa: E402
from experiments.report import compute_metrics, write_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multiple diarization configs against one audio file "
                    "and produce a side-by-side comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio_path", type=Path, help="Audio/video file to transcribe")
    p.add_argument(
        "--num-speakers", type=int, required=True,
        help="Exact number of speakers in the recording. Required.",
    )
    p.add_argument(
        "--configs", default=",".join(CONFIGS.keys()),
        help="Comma-separated config names from experiments.configs.CONFIGS",
    )
    p.add_argument(
        "--full-file", action="store_true",
        help="Run all chunks. Default: first chunk only (15 min) for fast iteration.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Default: experiments/<source-stem>/",
    )
    return p.parse_args()


def _config_header(name: str, cfg: DiarPipelineConfig, source: Path,
                   num_speakers: int, full_file: bool) -> list[str]:
    """Return the comment-prefixed header lines prepended to every .txt."""
    d = asdict(cfg)
    chunk_note = "all chunks" if full_file else "first chunk only (15 min)"
    return [
        f"# CONFIG: {name}",
        "# " + "  ".join(f"{k}={v}" for k, v in d.items() if k in (
            "audio_cleanup_for_diar", "align_words", "snap_boundaries",
            "min_turn_duration", "force_num_speakers",
        )),
        "# " + "  ".join(f"{k}={v}" for k, v in d.items() if k in (
            "speaker_match_threshold", "pyannote_model",
            "clustering_threshold", "segmentation_min_duration_off",
        )),
        f"# SOURCE: {source.name}   {chunk_note}   num_speakers={num_speakers}",
    ]


def _truncate_to_first_chunk(audio_path: Path, out_dir: Path) -> Path:
    """Slice audio to roughly CHUNK_SECONDS so iteration is fast. Re-encode to
    16kHz mono WAV so downstream code doesn't have to redo the same work for
    every config."""
    import subprocess
    out = out_dir / f"{audio_path.stem}_first_chunk.wav"
    if out.exists():
        return out
    log(f"Pre-cutting first {CHUNK_SECONDS}s for fast iteration -> {out.name}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_path),
            "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-t", str(CHUNK_SECONDS),
            "-y",
            str(out),
        ],
        check=True,
    )
    return out


def main() -> int:
    args = parse_args()
    if not args.audio_path.exists():
        print(f"audio file not found: {args.audio_path}", file=sys.stderr)
        return 2

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIGS]
    if unknown:
        print(f"unknown config(s): {unknown}. Known: {sorted(CONFIGS)}", file=sys.stderr)
        return 2

    out_dir = args.output_dir or (_HERE.parent / "experiments" / args.audio_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading models (one-time) ...")
    load_model()

    # Sanity check: warn if alignment was requested by any config but the
    # alignment model didn't load. The harness still runs; configs degrade
    # gracefully to raw Whisper timings.
    import whisper_daemon as wd
    wants_align = any(CONFIGS[c].align_words for c in config_names)
    if wants_align and wd._align_model is None:
        log("WARNING: configs request wav2vec2 alignment but the alignment model "
            "did not load. Affected configs will use raw Whisper word timings.")

    # If iterating fast, pre-cut to one chunk so every config sees identical
    # input. Saves ~13 min × N_configs of redundant ffmpeg/Whisper work.
    if args.full_file:
        source = args.audio_path
    else:
        source = _truncate_to_first_chunk(args.audio_path, out_dir)

    metrics_all = []
    for name in config_names:
        cfg = CONFIGS[name]
        log(f"=== Running config: {name} ===")
        txt_out = out_dir / f"{name}.txt"
        jsonl_out = out_dir / f"{name}.jsonl"
        # Stage to a temp .txt so the daemon writes the formatted body, then
        # we prepend the header + metrics line afterward.
        staged_txt = out_dir / f".{name}.body.txt"
        t0 = time.time()
        error: str | None = None
        try:
            turn_records = transcribe_file_diarized(
                source,
                staged_txt,
                speaker_hint={"max_speakers": args.num_speakers},
                audio_cleanup=True,
                silence_aware=False,
                silero_vad=False,
                config=cfg,
                return_jsonl=True,
            ) or []
        except Exception as e:
            error = repr(e)
            log(f"  config '{name}' FAILED: {error}")
            turn_records = []
        runtime = time.time() - t0

        # Write JSONL for metrics regardless of pass/fail.
        with jsonl_out.open("w", encoding="utf-8") as f:
            for r in turn_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Compute metrics now so the .txt header can include them.
        m = compute_metrics(jsonl_out, name, runtime)
        if error and not m.error:
            m.error = error
        metrics_all.append(m)

        # Compose final .txt: config header + metrics line + body.
        header = _config_header(name, cfg, args.audio_path, args.num_speakers, args.full_file)
        metric_line = (
            f"# METRICS: distinct_speakers={m.distinct_speakers}  "
            f"total_turns={m.total_turns}  top2_word_share={m.top2_word_share:.2f}  "
            f"mid_sentence_swaps={m.mid_sentence_swaps}  runtime={m.runtime_sec:.1f}s"
        )
        body = staged_txt.read_text(encoding="utf-8") if staged_txt.exists() else "(no output)\n"
        txt_out.write_text(
            "\n".join(header + [metric_line, "#", ""]) + body,
            encoding="utf-8",
        )
        if staged_txt.exists():
            staged_txt.unlink()

        log(f"  done in {runtime:.1f}s -> {txt_out.name}  ({m.distinct_speakers} speakers, "
            f"{m.mid_sentence_swaps} mid-sentence swaps)")

    summary_path = out_dir / "summary.md"
    write_summary(summary_path, metrics_all, args.num_speakers, args.audio_path)
    log(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
