from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def json_read(path: Path):
    return json.loads(path.read_text())


class Runner:
    """Every process argv, version and elapsed time is recorded; no shell execution."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.records = []
        self.logger = logging.getLogger(str(log_dir))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / "commands.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.logger.addHandler(handler)

    def run(self, args, label, timeout=None):
        args = [str(a) for a in args]
        self.sequence += 1
        label = re.sub(r"[^\w.-]", "_", label)
        self.logger.info("%s %s", label, shlex.join(args))
        start = time.monotonic()
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            self.logger.error("%s interrupted or timed out", label)
            raise
        elapsed = time.monotonic() - start
        log = self.log_dir / f"{self.sequence:04d}_{label}.log"
        log.write_text(result.stdout + result.stderr)
        self.records.append({"argv": args, "label": label, "elapsed_seconds": elapsed,
                             "returncode": result.returncode, "log": str(log)})
        json_write(self.log_dir / "commands.json", self.records)
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")
        return result


def probe(path: Path, runner: Runner):
    return json.loads(runner.run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                                  "-of", "json", path], "probe").stdout)


def stream(meta, kind):
    return next((s for s in meta["streams"] if s["codec_type"] == kind), None)


def require_tools(runner):
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise RuntimeError(f"Missing {name}; install FFmpeg with Homebrew (brew install ffmpeg)")
    filters = runner.run(["ffmpeg", "-hide_banner", "-filters"], "filters").stdout
    encoders = runner.run(["ffmpeg", "-hide_banner", "-encoders"], "encoders").stdout
    for f in ("hqdn3d", "bwdif", "afftdn", "loudnorm", "unsharp", "scale", "idet", "hstack"):
        if not re.search(r"\s" + f + r"\s", filters):
            raise RuntimeError(f"FFmpeg lacks required filter {f}")
    for e in ("libx264", "aac", "pcm_s24le"):
        if not re.search(r"\s" + e + r"\s", encoders):
            raise RuntimeError(f"FFmpeg lacks required encoder {e}")
    return filters, encoders


def ffmpeg(*args):
    return ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-filter_threads", "2", *map(str, args)]


def audio_stats(path: Path, runner):
    result = runner.run(ffmpeg("-i", path, "-map", "0:a:0", "-vn", "-af",
                               "loudnorm=I=-18:TP=-2:LRA=50:print_format=json", "-f", "null", "-"),
                        "loudness")
    matches = re.findall(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not matches:
        raise RuntimeError("FFmpeg returned no loudness statistics")
    raw = json.loads(matches[-1])
    stats = {k: float(raw[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh")}
    if not all(math.isfinite(x) for x in stats.values()):
        raise RuntimeError("Audio has insufficient nonsilent content for reliable loudness matching")
    return stats


def common_gain(stats, target=-18.0, ceiling=-2.0):
    """One LUFS target attainable by every candidate without peak limiting."""
    attainable = [s["input_i"] + ceiling - s["input_tp"] for s in stats.values()]
    actual_target = min(target, *attainable)
    return actual_target, {name: actual_target - s["input_i"] for name, s in stats.items()}


def interlace_evidence(source, start, duration, runner):
    result = runner.run(ffmpeg("-ss", start, "-i", source, "-t", min(duration, 10),
                               "-map", "0:v:0", "-an", "-vf", "idet", "-f", "null", "-"), "idet")
    m = re.findall(r"Multi frame detection: TFF:\s*(\d+) BFF:\s*(\d+) Progressive:\s*(\d+) Undetermined:\s*(\d+)", result.stderr)
    if not m:
        raise RuntimeError("No idet results")
    tff, bff, progressive, unknown = map(int, m[-1])
    determined = tff + bff + progressive
    return {"tff": tff, "bff": bff, "progressive": progressive, "undetermined": unknown,
            "interlaced_fraction": (tff + bff) / determined if determined else None}


def deinterlace_decision(meta, evidence, mode):
    if mode not in ("auto", "on", "off"):
        raise ValueError("deinterlace must be auto/on/off")
    if mode != "auto":
        return {"enabled": mode == "on", "reason": "explicit preset override"}
    order = stream(meta, "video").get("field_order", "unknown")
    fraction = evidence.get("interlaced_fraction")
    # Auto requires strong local evidence; screen text can fool idet.
    enabled = (order in ("tt", "bb", "tb", "bt") and fraction is not None and fraction >= 0.8)
    ambiguous = (fraction is not None and fraction > 0.1) or (order not in ("progressive", "unknown") and not enabled)
    return {"enabled": enabled, "ambiguous": ambiguous,
            "reason": "metadata and >=80% idet evidence" if enabled else "no confident interlace evidence; preserve frames"}


def video_filter(preset, decision):
    filters = []
    if decision["enabled"]:
        filters.append("bwdif=mode=send_frame:parity=auto:deint=all")
    if preset.get("deblock"):
        filters.append("deblock=filter=weak:block=8:alpha=0.04:beta=0.02:gamma=0.02:delta=0.02:planes=1")
    if preset.get("hqdn3d"):
        filters.append("hqdn3d=" + preset["hqdn3d"])
    scale = preset["scale"]
    if scale not in (1, 2):
        raise ValueError("Only native or exactly 2x scale supported")
    if scale == 2:
        filters.append("scale=iw*2:ih*2:flags=lanczos")
    if preset.get("unsharp"):
        filters.append("unsharp=" + preset["unsharp"])
    filters.append("format=yuv420p")
    return ",".join(filters)


def disk_guard(directory, needed_bytes):
    free = shutil.disk_usage(directory).free
    if free < needed_bytes + 1024**3:
        raise RuntimeError(f"Insufficient disk: need estimated {needed_bytes/1024**3:.1f} GiB plus 1 GiB reserve, have {free/1024**3:.1f} GiB")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()
