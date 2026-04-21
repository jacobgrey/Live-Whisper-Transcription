"""
diarize_worker.py  --  Isolated pyannote diarization subprocess.

Intentionally imports NO ctranslate2 / faster-whisper code so that
PyTorch's cuDNN context owns the GPU without DLL conflicts on Windows.

Usage (called by whisper_daemon.py):
    python diarize_worker.py <wav_path> [--cpu]
                             [--num-speakers N | --max-speakers N [--min-speakers M]]

Exits 0 on success and prints a JSON object to stdout:
    {
      "segments": [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"}, ...],
      "embeddings": {"SPEAKER_00": [float, ...], "SPEAKER_01": [...]}  // may be empty
    }

Exits 1 on error and prints {"error": "<message>"} to stdout
so the parent process can surface a clean error string.
"""

import sys
import json
import os
import argparse
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


def _parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("wav_path")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    return p.parse_args(argv)


def main():
    try:
        args = _parse_args(sys.argv[1:])
    except SystemExit:
        print(json.dumps({"error": "bad arguments"}))
        sys.exit(1)

    wav_path = args.wav_path
    use_cpu = args.cpu

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

    # Build pipeline kwargs from speaker-count hints.
    pipe_kwargs: dict = {}
    if args.num_speakers is not None:
        pipe_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            pipe_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            pipe_kwargs["max_speakers"] = args.max_speakers

    # Request embeddings so the parent process can stitch speakers across chunks.
    # Not all pipeline versions support this; fall back if it raises TypeError.
    embeddings_out: dict[str, list[float]] = {}
    diar = None
    try:
        result = pipeline(wav_path, return_embeddings=True, **pipe_kwargs)
        # pyannote 3.x returns (Annotation, np.ndarray) when return_embeddings=True.
        if isinstance(result, tuple) and len(result) == 2:
            diar, emb_array = result
            if emb_array is not None:
                try:
                    labels = diar.labels()  # local speaker ids, order matches rows of emb_array
                    for i, lab in enumerate(labels):
                        if i >= len(emb_array):
                            break
                        row = emb_array[i]
                        # Skip NaN/None rows (can happen if a label has no assignable embedding).
                        try:
                            vec = [float(x) for x in row]
                        except Exception:
                            continue
                        if any(v != v for v in vec):  # NaN check
                            continue
                        embeddings_out[str(lab)] = vec
                except Exception as emb_e:
                    sys.stderr.write(f"[diarize_worker] embedding extraction failed: {emb_e}\n")
        else:
            diar = result
    except TypeError:
        # Older pipeline: no return_embeddings support. Re-run without it.
        try:
            diar = pipeline(wav_path, **pipe_kwargs)
        except Exception as e:
            print(json.dumps({"error": f"diarization failed: {e}"}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"diarization failed: {e}"}))
        sys.exit(1)

    segments = [
        {"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
        for t, _, spk in diar.itertracks(yield_label=True)
    ]

    print(json.dumps({"segments": segments, "embeddings": embeddings_out}))
    sys.exit(0)

if __name__ == "__main__":
    main()
