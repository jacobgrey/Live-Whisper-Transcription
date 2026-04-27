"""Named diarization pipeline configurations for the A/B harness.

Each config is a (DiarPipelineConfig, audio_cleanup_for_whisper) pair so the
harness can also vary whether Whisper sees cleaned or raw audio. Currently
every config sends cleaned audio to Whisper (transcription quality benefits
from cleanup) and varies only what pyannote sees and how its output is
post-processed.
"""

from __future__ import annotations

# Allow running as a script: add ../src to sys.path so the daemon module imports.
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from whisper_daemon import DiarPipelineConfig  # noqa: E402


CONFIGS: dict[str, DiarPipelineConfig] = {
    # Reproduces the production behavior at the time of writing: cleaned audio
    # to both Whisper and pyannote, wav2vec2 alignment + boundary snap on,
    # default pyannote model and tracker threshold, hint passed as max_speakers.
    "baseline": DiarPipelineConfig(),

    # Keep everything else identical to baseline; only flip the hint mode so
    # max_speakers becomes num_speakers (exact). Cheapest possible test.
    "force_num": DiarPipelineConfig(force_num_speakers=True),

    # Feed pyannote raw audio (no highpass/compressor/loudnorm). Embedding
    # quality should be higher because speaker-distinguishing features aren't
    # flattened by loudnorm.
    "raw_audio": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        force_num_speakers=True,
    ),

    # Disable wav2vec2 alignment + boundary snap. Tests whether the recent
    # "optimization" is making misassignments more confident than they should be.
    "no_snap": DiarPipelineConfig(
        align_words=False,
        snap_boundaries=False,
        force_num_speakers=True,
    ),

    # Combine the two big rollbacks: raw audio for pyannote AND no snap.
    "raw_no_snap": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        align_words=False,
        snap_boundaries=False,
        force_num_speakers=True,
    ),

    # Add post-assignment smoothing: any speaker run shorter than 0.4s is
    # absorbed into the longer neighbor. Targets the "single misassigned word
    # becomes its own turn" pattern.
    "raw_no_snap_smooth": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        align_words=False,
        snap_boundaries=False,
        min_turn_duration=0.4,
        force_num_speakers=True,
    ),

    # Tune pyannote's clustering + segmentation hyperparameters. Higher
    # clustering threshold = less aggressive cluster merging = more speakers
    # kept distinct. Higher segmentation min_duration_off = require longer
    # silence to call a turn boundary.
    "tuned": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        align_words=False,
        snap_boundaries=False,
        min_turn_duration=0.4,
        force_num_speakers=True,
        clustering_threshold=0.85,
        segmentation_min_duration_off=0.5,
    ),

    # Try the older but more widely tested pyannote 3.1 model.
    "model_3_1": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        align_words=False,
        snap_boundaries=False,
        min_turn_duration=0.4,
        force_num_speakers=True,
        pyannote_model="pyannote/speaker-diarization-3.1",
    ),

    # Tighten the cross-chunk speaker matching threshold so similar-but-distinct
    # speakers are less likely to be merged at chunk boundaries.
    "tracker_07": DiarPipelineConfig(
        audio_cleanup_for_diar=False,
        align_words=False,
        snap_boundaries=False,
        min_turn_duration=0.4,
        force_num_speakers=True,
        speaker_match_threshold=0.7,
    ),
}
