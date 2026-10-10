"""Optional, local DeepFilterNet inference without importing Torch into the CLI.

The parent process uses only the standard library. The worker runs inside the
project's private Python 3.11 environment and never downloads a model itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


class NeuralUnavailable(RuntimeError):
    """Optional neural dependency or model is unavailable; baseline stays usable."""


def enhance(input_wav: Path, output_wav: Path, runner, project_root: Path,
            attenuation_limit_db: float = 12) -> dict:
    """Enhance a 48 kHz WAV, preserving channels and exactly its sample count.

    Raises NeuralUnavailable on setup failure. Other failures are propagated so
    the caller can log and continue with the conservative FFmpeg candidate.
    Output is published atomically by the worker after successful validation.
    """
    project_root = Path(project_root).resolve()
    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()
    if input_wav == output_wav:
        raise ValueError("Neural input and output must be different paths")
    if not math.isfinite(attenuation_limit_db) or not 0 < attenuation_limit_db <= 30:
        raise ValueError("attenuation_limit_db must be greater than 0 and at most 30")
    python = project_root / ".venv-neural" / "bin" / "python"
    state_path = project_root / "reports" / "neural_setup.json"
    model_dir = project_root / ".models" / "deepfilternet" / "DeepFilterNet3"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError) as exc:
        raise NeuralUnavailable("Run python3 scripts/setup_neural.py first") from exc
    if not state.get("ready") or not python.is_file() or not model_dir.is_dir():
        raise NeuralUnavailable(state.get("error", "Private neural environment is not ready"))
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".neural-report-", dir=output_wav.parent) as work:
        report = Path(work) / "metrics.json"
        args = [str(python), str(Path(__file__).resolve()), "--worker",
                "--input", str(input_wav), "--output", str(output_wav),
                "--model", str(model_dir), "--metrics", str(report),
                "--attenuation", str(attenuation_limit_db)]
        runner.run(args, label="deepfilternet_cpu")
        if not report.is_file() or not output_wav.is_file():
            raise RuntimeError("Neural worker did not produce validated output")
        metrics = json.loads(report.read_text())
    metrics["setup_state"] = str(state_path)
    return metrics


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker(args) -> dict:
    # DeepFilterNet's configuration reads DEVICE directly from the environment.
    os.environ["DEVICE"] = "cpu"
    os.environ["MASK_PF"] = "False"
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    import torch
    from df.enhance import enhance as df_enhance, init_df
    from df.model import ModelParams
    from libdf import DF

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model).resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"Neural output already exists: {output_path}")
    model_manifest = model_dir.parent / "manifest.json"
    manifest = json.loads(model_manifest.read_text())
    for name, expected in manifest["file_sha256"].items():
        if _sha256(model_dir / name) != expected:
            raise RuntimeError(f"Model integrity mismatch: {name}; rerun setup")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    loaded = init_df(str(model_dir), post_filter=False, log_level="WARNING", log_file=None)
    model, initial_state = loaded[:2]
    if initial_state.sr() != 48000:
        raise ValueError("Only the 48 kHz model is supported")
    model = model.to("cpu").eval()
    fft_size = initial_state.fft_size()
    hop_size = initial_state.hop_size()
    min_nb_erb_freqs = ModelParams().min_nb_freqs
    chunk_frames = int(args.chunk_seconds * 48000)
    context_frames = 48000
    fade_frames = 24000
    max_peak = 0.0
    out_frames = 0
    chunks = 0
    # Each call starts fresh; one second of context warms normalization and
    # recurrence. The preceding prediction is blended across the first half
    # second of each new core, avoiding an abrupt state-reset seam.
    with sf.SoundFile(input_path) as source:
        if source.samplerate != 48000:
            raise ValueError("Neural input must be a 48 kHz WAV")
        if source.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("Neural input must use WAV/RF64")
        channels, total_frames = source.channels, len(source)
        if channels < 1 or total_frames < 1:
            raise ValueError("Input WAV must contain audio")
        if channels > 8:
            raise ValueError("More than eight audio channels is not supported")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-", suffix=".wav", dir=output_path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        previous_tail = None
        try:
            with sf.SoundFile(temporary_path, mode="w", samplerate=48000,
                              channels=channels, format="RF64", subtype="FLOAT") as target:
                for core_start in range(0, total_frames, chunk_frames):
                    core_end = min(total_frames, core_start + chunk_frames)
                    read_start = max(0, core_start - context_frames)
                    read_end = min(total_frames, core_end + context_frames)
                    source.seek(read_start)
                    data = source.read(read_end - read_start, dtype="float32", always_2d=True)
                    if not np.isfinite(data).all():
                        raise ValueError("Input contains non-finite samples")
                    state = DF(sr=48000, fft_size=fft_size, hop_size=hop_size,
                               nb_bands=len(initial_state.erb_widths()),
                               min_nb_erb_freqs=min_nb_erb_freqs)
                    tensor = torch.from_numpy(np.ascontiguousarray(data.T))
                    result = df_enhance(model, state, tensor, pad=True,
                                        atten_lim_db=args.attenuation)
                    if result.shape != tensor.shape:
                        raise RuntimeError("DeepFilterNet changed the chunk sample count")
                    enhanced = result.cpu().numpy().T.copy()
                    local_start = core_start - read_start
                    core = enhanced[local_start:local_start + core_end - core_start].copy()
                    if previous_tail is not None:
                        count = min(len(previous_tail), len(core), fade_frames)
                        weights = ((1 - np.cos(np.linspace(0, np.pi, count))) / 2).astype("float32")
                        core[:count] = previous_tail[:count] * (1 - weights[:, None]) + core[:count] * weights[:, None]
                    tail_start = core_end - read_start
                    previous_tail = enhanced[tail_start:tail_start + fade_frames].copy()
                    if not np.isfinite(core).all():
                        raise RuntimeError("DeepFilterNet produced non-finite samples")
                    max_peak = max(max_peak, float(np.max(np.abs(core))))
                    target.write(core)
                    out_frames += len(core)
                    chunks += 1
                    print(json.dumps({"chunk": chunks, "completed_frames": out_frames,
                                      "total_frames": total_frames}), file=sys.stderr, flush=True)
            info = sf.info(temporary_path)
            if out_frames != total_frames or info.frames != total_frames or info.channels != channels:
                raise RuntimeError("Neural sample-count or channel validation failed")
            # Float WAV retains full precision and reports any peaks above unity
            # for the subsequent measured loudness/peak limiting step.
            elapsed = time.monotonic() - started
            metrics = {"backend": "DeepFilterNet3", "device": "cpu", "post_filter": False,
                       "attenuation_limit_db": args.attenuation,
                       "sample_rate": 48000, "channels": channels,
                       "input_frames": total_frames, "output_frames": out_frames,
                       "duration_seconds": total_frames / 48000,
                       "exact_sample_count": True, "delay_compensated": True,
                       "stft_delay_samples_removed_per_chunk": fft_size - hop_size,
                       "chunks": chunks, "chunk_seconds": args.chunk_seconds,
                       "context_seconds": 1.0, "crossfade_seconds": 0.5,
                       "sample_peak": max_peak,
                       "sample_peak_dbfs": 20 * math.log10(max_peak) if max_peak else None,
                       "elapsed_seconds": elapsed,
                       "processing_realtime_factor": elapsed / (total_frames / 48000),
                       "threads": args.threads, "model_manifest": manifest,
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ("torch", "torchaudio", "numpy", "deepfilternet", "deepfilterlib", "soundfile")},
                       "quality_note": "Metrics verify format and timing, not word preservation or intelligibility."}
            Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n")
            # Hard-link publishes without overwriting a concurrently-created file.
            os.link(temporary_path, output_path)
            temporary_path.unlink()
            return metrics
        finally:
            temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Private DeepFilterNet CPU worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--attenuation", type=float, default=12)
    parser.add_argument("--chunk-seconds", type=float, default=30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not math.isfinite(args.chunk_seconds) or not 2 <= args.chunk_seconds <= 60:
        parser.error("chunk seconds must be between 2 and 60")
    if not math.isfinite(args.attenuation) or not 0 < args.attenuation <= 30:
        parser.error("attenuation must be greater than 0 and at most 30 dB")
    if not 1 <= args.threads <= 8:
        parser.error("threads must be between 1 and 8")
    _worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
