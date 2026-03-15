"""
diarize_worker.py  --  Isolated pyannote diarization subprocess.

Intentionally imports NO ctranslate2 / faster-whisper code so that
PyTorch's cuDNN context owns the GPU without DLL conflicts on Windows.

Usage (called by whisper_daemon.py):
    python diarize_worker.py <wav_path> [--cpu]

Exits 0 on success and prints a JSON array to stdout:
    [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"}, ...]

Exits 1 on error and prints {"error": "<message>"} to stdout
so the parent process can surface a clean error string.
"""

import sys
import json
import os
from pathlib import Path

def load_hf_token() -> str | None:
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    try:
        token_file = Path(__file__).resolve().parent.parent / "config" / "hf_token.txt"
        if token_file.exists():
            return token_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def main():
    use_cpu = "--cpu" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print(json.dumps({"error": "no wav path supplied"}))
        sys.exit(1)

    wav_path = args[0]

    token = load_hf_token()
    if not token:
        print(json.dumps({"error": "missing Hugging Face token"}))
        sys.exit(1)

    model_name = os.environ.get(
        "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
    )

    try:
        import torch
        from pyannote.audio import Pipeline  # type: ignore

        # PyTorch 2.6+ requires weights_only=False for pyannote checkpoints.
        _orig = torch.load

        def _load_full(*a, **kw):
            kw["weights_only"] = False
            return _orig(*a, **kw)

        torch.load = _load_full  # type: ignore
        try:
            try:
                pipeline = Pipeline.from_pretrained(model_name, token=token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        finally:
            torch.load = _orig  # type: ignore

    except Exception as e:
        print(json.dumps({"error": f"pipeline load failed: {e}"}))
        sys.exit(1)

    # Move to GPU unless caller requested CPU fallback.
    if not use_cpu and torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
        except Exception as e:
            # Non-fatal: fall back to CPU silently.
            sys.stderr.write(f"[diarize_worker] GPU move failed ({e}), using CPU\n")
    else:
        # Cap threads so the worker doesn't saturate an older CPU.
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(max(2, cpu_count // 2))

    try:
        diar = pipeline(wav_path)
    except Exception as e:
        print(json.dumps({"error": f"diarization failed: {e}"}))
        sys.exit(1)

    segments = [
        {"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
        for t, _, spk in diar.itertracks(yield_label=True)
    ]
    print(json.dumps(segments))
    sys.exit(0)

if __name__ == "__main__":
    main()
