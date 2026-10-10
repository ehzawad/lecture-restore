from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

from .core import (Runner, audio_stats, common_gain, deinterlace_decision, digest,
                   disk_guard, ffmpeg, interlace_evidence, json_read, json_write,
                   now, probe, require_tools, stream, video_filter)
from .validation import validate

CODE_ROOT = Path(__file__).resolve().parents[1]
# Runtime paths are selected by main(); code and shipped presets stay immutable.
ROOT = CODE_ROOT
REVIEW_ITEMS = {"speech", "symbols", "faces", "handwriting", "sync", "motion"}


def presets(path):
    values = json_read(path)["presets"]
    for name, p in values.items():
        if p["scale"] not in (1, 2) or not 0 <= p["crf"] <= 51:
            raise ValueError(f"Invalid preset {name}")
        if not 1 <= p["threads"] <= 32:
            raise ValueError(f"Invalid thread count in {name}")
    return values


def inspect(args, runner):
    require_tools(runner)
    source = args.input.resolve()
    meta = probe(source, runner)
    if not stream(meta, "video") or not stream(meta, "audio"):
        raise RuntimeError("This lecture workflow requires both a video stream and an audio stream")
    result = {"created_at": now(), "source": str(source), "sha256": digest(source), "metadata": meta,
              "architecture": platform.machine(), "python": sys.version,
              "ffmpeg": runner.run(["ffmpeg", "-version"], "version").stdout,
              "ffprobe": runner.run(["ffprobe", "-version"], "probe_version").stdout,
              "disk_free_bytes": shutil.disk_usage(ROOT).free,
              "cuda_used": False, "baseline_compute": "CPU libx264; 4 encoder threads, 2 filter threads",
              "topaz": {"installed": False, "cli_supported": False}}
    if platform.system() == "Darwin":
        for key, cmd in {"macos": ["sw_vers"], "memory_total_bytes": ["sysctl", "-n", "hw.memsize"],
                         "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"],
                         "memory_vm_stat": ["vm_stat"], "memory_pressure": ["memory_pressure"],
                         "hwaccels": ["ffmpeg", "-hide_banner", "-hwaccels"]}.items():
            result[key] = runner.run(cmd, key).stdout
        result["topaz"]["apps"] = [str(p) for base in (Path("/Applications"), Path.home()/"Applications")
                                    if base.exists() for p in base.glob("*Topaz*Video*.app")]
        result["topaz"]["installed"] = bool(result["topaz"]["apps"])
        result["topaz"]["note"] = "No enhancement invoked. Any installed version needs a verified vendor CLI adapter and separately reviewed previews."
    reference = ROOT / "output/reference/original_audio.m4a"
    reference.parent.mkdir(parents=True, exist_ok=True)
    ref_identity = reference.with_suffix(".source.json")
    if reference.exists():
        old = json_read(ref_identity) if ref_identity.exists() else json_read(ROOT / "reports/inspection.json")
        if old.get("sha256") != result["sha256"]:
            raise RuntimeError("Existing original audio reference belongs to a different input; choose a separate project folder")
        expected_audio_sha = old.get("audio_sha256") or old.get("original_audio_reference", {}).get("sha256")
        if expected_audio_sha != digest(reference):
            raise RuntimeError("Original audio reference changed; inspect its identity before regenerating")
    if not reference.exists():
        partial = reference.with_name("original_audio.partial.m4a")
        try:
            runner.run(ffmpeg("-i", source, "-map", "0:a:0", "-vn", "-c:a", "copy", partial), "original_audio_copy")
            os.replace(partial, reference)
        finally:
            partial.unlink(missing_ok=True)
    result["original_audio_reference"] = {"path": str(reference), "sha256": digest(reference),
                                           "note": "AAC stream copied without re-encoding"}
    json_write(ref_identity, {"sha256": result["sha256"], "audio_sha256": digest(reference)})
    json_write(ROOT / "reports/inspection.json", result)
    print(json.dumps({k: result[k] for k in ("architecture", "source", "sha256", "topaz")}, indent=2))


def encode(source, audio, out, start, duration, preset, vf, runner, audio_offset=0):
    cmd = ffmpeg("-ss", start, "-i", source, "-itsoffset", audio_offset, "-i", audio,
                 "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0")
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", preset["encoder_preset"],
            "-crf", str(preset["crf"]), "-threads", str(preset["threads"]),
            "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(out)]
    runner.run(cmd, "encode_" + out.stem)


def extract_audio(source, target, start, duration, runner):
    cmd = ffmpeg("-ss", start, "-i", source)
    if duration is not None:
        cmd += ["-t", str(duration)]
    runner.run(cmd + ["-map", "0:a:0", "-vn", "-ar", "48000", "-c:a", "pcm_s24le", str(target)], "extract_audio")


def matched_audio(raw, out, gain, runner):
    runner.run(ffmpeg("-i", raw, "-af", f"volume={gain:.9f}dB", "-c:a", "pcm_s24le", out), "constant_gain")
    stats = audio_stats(out, runner)
    return stats


def compare_frames(source, restored, start, duration, folder, runner):
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    # Match by nearest frame at each local timestamp. No new temporal interpolation.
    for index, local in enumerate((2.0, duration / 2, duration - 2.0)):
        out = folder / f"comparison_{index}.png"
        graph = "[0:v]setsar=1[a];[1:v]scale=1280:720:flags=lanczos,setsar=1[b];[a][b]hstack=inputs=2"
        meta = probe(source, runner)
        v = stream(meta, "video")
        # Display both at source raster size while preserving SAR in the actual videos.
        graph = f"[0:v]setsar=1[a];[1:v]scale={v['width']}:{v['height']}:flags=lanczos,setsar=1[b];[a][b]hstack=inputs=2"
        runner.run(ffmpeg("-ss", start + local, "-i", source, "-ss", local, "-i", restored,
                           "-filter_complex_threads", "2", "-filter_complex", graph,
                           "-frames:v", "1", out), "comparison_frame")
        frames.append({"path": str(out), "source_seconds": start + local,
                       "left": "source", "right": "restored, displayed at source raster dimensions"})
    return frames


def checked_files(item):
    return all(Path(path).is_file() and digest(Path(path)) == sha for path, sha in item["files"].items())


def write_gallery(run, manifest):
    parts = ["<!doctype html><meta charset='utf-8'><title>Lecture preview comparison</title>",
             "<style>body{font:16px system-ui;background:#171b21;color:#eee;margin:28px}video{width:min(100%,960px)}img{width:100%}section{margin:32px 0;border-top:1px solid #888;padding-top:16px}a{color:#89c8ff}audio{display:block}code{color:#badaff}</style>",
             "<h1>Lecture restoration previews</h1><p>Compare matched audio at the same playback volume. Images: source left, restored right. 2× is interpolation; no recovered detail is claimed.</p>",
             "<p>Listen for consonants, quiet words, lisping or musical noise. Check equations, subscripts, minus signs, handwriting if present, faces, motion and lip sync. A successful validation report verifies media structure, not content quality.</p>"]
    for s in manifest["completed"]:
        parts.append(f"<section><h2>{html.escape(s['segment']['name'])} — {s['segment']['start']} seconds</h2><p>{html.escape(s['segment'].get('content',''))}</p>")
        for variant in s["variants"]:
            def rel(path): return html.escape(os.path.relpath(path, run))
            parts.append(f"<h3>{html.escape(variant['name'])}</h3><video controls preload='none' src='{rel(variant['video'])}'></video>")
            parts.append(f"<p>Validation: {variant['validation']['status']}</p>")
            for frame in variant["frames"]:
                parts.append(f"<a href='{rel(frame['path'])}'><img loading='lazy' src='{rel(frame['path'])}'></a>")
        for label, path in s["matched_audio"].items():
            parts.append(f"<p>{html.escape(label)} (matched)</p><audio controls preload='none' src='{rel(path)}'></audio>")
        parts.append(f"<p><a href='{rel(s['original_audio'])}'>Unfiltered original PCM reference</a></p></section>")
    parts.append("<p>Accept only after comparing all representative sections. No full-length restoration has been generated by this preview command.</p>")
    (run / "index.html").write_text("\n".join(parts))


def _build_segment(args, folder, segment, selected, meta, runner):
    from .neural import enhance
    name = segment["name"]
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Segment names must contain only letters, digits, '-' or '_'")
    start, duration = float(segment["start"]), float(segment["duration"])
    if start < 0 or not 30 <= duration <= 60 or start + duration > float(meta["format"]["duration"]):
        raise ValueError("Preview sections must be valid 30–60 second intervals")
    folder.mkdir(parents=True, exist_ok=True)
    evidence = interlace_evidence(args.input, start, duration, runner)
    raw_original = folder / "original_unfiltered.wav"
    extract_audio(args.input, raw_original, start, duration, runner)
    raw = {"original": raw_original}
    # Different audio presets are compared separately; identical filters share a waveform.
    filter_names = {}
    for preset_name, p in selected.items():
        audio_filter = p["audio_filter"]
        if audio_filter in filter_names:
            continue
        filter_names[audio_filter] = preset_name
        path = folder / f"{preset_name}_raw.wav"
        runner.run(ffmpeg("-i", raw_original, "-af", audio_filter, "-c:a", "pcm_s24le", path), "conservative_audio")
        raw[preset_name] = path
    neural = {"requested": args.neural, "available": False}
    if args.neural:
        path = folder / "neural_raw.wav"
        try:
            neural = enhance(raw_original, path, runner, ROOT)
            neural["available"] = True
            raw["neural"] = path
        except (RuntimeError, OSError, ValueError) as exc:
            neural = {"requested": True, "available": False, "error": str(exc), "fallback": "FFmpeg baseline"}
            print(f"{name}: neural unavailable; preserving baseline: {exc}", file=sys.stderr)
    stats = {label: audio_stats(path, runner) for label, path in raw.items()}
    ceiling = min(p["true_peak_ceiling_db"] for p in selected.values())
    target, gains = common_gain(stats, min(p["target_lufs"] for p in selected.values()), ceiling)
    matched, measured = {}, {}
    for label, path in raw.items():
        matched[label] = str(folder / f"{label}_matched.wav")
        measured[label] = matched_audio(path, Path(matched[label]), gains[label], runner)
    match_error = max(v["input_i"] for v in measured.values()) - min(v["input_i"] for v in measured.values())
    if match_error > 0.2 or any(v["input_tp"] > ceiling + 0.1 for v in measured.values()):
        raise RuntimeError("Loudness matching or peak check failed")
    # A matched, unfiltered control makes motion and slide transitions reviewable.
    # Its video is recompressed with the same encoder; comparison PNGs use source pixels.
    control_preset = {**next(iter(selected.values())), "scale": 1}
    control = folder / "reference.mp4"
    encode(args.input, matched["original"], control, start, duration, control_preset, "format=yuv420p", runner)
    control_validation = validate(args.input, control, runner, expected_duration=duration,
                                  expected_scale=1, segment_start=start)
    json_write(folder / "reference_validation.json", control_validation)
    if not control_validation["passed"]:
        raise RuntimeError(f"Reference preview validation failed for {name}")
    variants = [{"name": "reference", "preset": "reference", "audio": "original", "video": str(control),
                 "video_filter": "format=yuv420p", "deinterlace": {"enabled": False},
                 "validation": control_validation, "frames": []}]
    for preset_name, p in selected.items():
        decision = deinterlace_decision(meta, evidence, p["deinterlace"])
        vf = video_filter(p, decision)
        audio_label = filter_names[p["audio_filter"]]
        out = folder / f"{preset_name}.mp4"
        partial = folder / f".{preset_name}.partial.mp4"
        try:
            encode(args.input, matched[audio_label], partial, start, duration, p, vf, runner)
            validation = validate(args.input, partial, runner, expected_duration=duration,
                                  expected_scale=p["scale"], segment_start=start)
            if not validation["passed"]:
                json_write(folder / f"{preset_name}_failed_validation.json", validation)
                raise RuntimeError(f"Validation failed for {name}/{preset_name}")
            os.replace(partial, out)
        finally:
            partial.unlink(missing_ok=True)
        validation["candidate_path"] = str(out)
        json_write(folder / f"{preset_name}_validation.json", validation)
        frames = compare_frames(args.input, out, start, duration, folder / preset_name, runner)
        variants.append({"name": preset_name, "preset": preset_name, "audio": "baseline", "video": str(out),
                         "video_filter": vf, "deinterlace": decision, "validation": validation, "frames": frames})
        if neural.get("available"):
            nout = folder / f"{preset_name}_neural.mp4"
            runner.run(ffmpeg("-i", out, "-i", matched["neural"], "-map", "0:v:0", "-map", "1:a:0",
                               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", nout), "neural_mux")
            nvalidation = validate(args.input, nout, runner, expected_duration=duration,
                                   expected_scale=p["scale"], segment_start=start)
            json_write(folder / f"{preset_name}_neural_validation.json", nvalidation)
            if not nvalidation["passed"]:
                raise RuntimeError(f"Neural preview validation failed for {name}")
            variants.append({"name": preset_name + "_neural", "preset": preset_name, "audio": "neural", "video": str(nout),
                             "video_filter": vf, "deinterlace": decision, "validation": nvalidation, "frames": frames})
    files = {str(p): digest(p) for p in folder.rglob("*") if p.is_file()}
    return {"segment": segment, "idet": evidence, "original_audio": str(raw_original), "neural": neural,
            "matched_audio": matched, "raw_audio_stats": stats, "matched_audio_stats": measured,
            "match_lufs_spread": match_error, "common_target_lufs": target, "gains_db": gains,
            "variants": variants, "files": files}


def _relocate(value, old, new):
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old + os.sep) else value
    if isinstance(value, list):
        return [_relocate(v, old, new) for v in value]
    if isinstance(value, dict):
        return {_relocate(k, old, new): _relocate(v, old, new) for k, v in value.items()}
    return value


def preview_segment(args, run, segment, selected, meta, runner):
    name = segment["name"]
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid segment name")
    destination = run / name
    with tempfile.TemporaryDirectory(prefix=f".work_{name}_", dir=run) as temporary:
        stage = Path(temporary) / name
        item = _build_segment(args, stage, segment, selected, meta, runner)
        item = _relocate(item, str(stage), str(destination))
        for report in stage.rglob("*.json"):
            json_write(report, _relocate(json_read(report), str(stage), str(destination)))
        if destination.exists():
            manifest = json_read(run / "manifest.json")
            if not any(i["segment"]["name"] == name for i in manifest["completed"]):
                raise RuntimeError(f"Refusing to replace an untracked preview directory: {destination}")
            if destination.is_symlink():
                raise RuntimeError("Refusing to replace a symlinked preview directory")
            shutil.rmtree(destination)
        os.replace(stage, destination)
        item["files"] = {str(p): digest(p) for p in destination.rglob("*") if p.is_file()}
        return item


def neural_signature(runner):
    state_path = ROOT / "reports/neural_setup.json"
    if not state_path.exists() or not json_read(state_path).get("ready"):
        return None
    model = ROOT / ".models/deepfilternet/DeepFilterNet3"
    python = ROOT / ".venv-neural/bin/python"
    code = "import importlib.metadata as m,json,sys,platform;print(json.dumps({'packages':{p:m.version(p) for p in ('torch','torchaudio','numpy','deepfilternet','deepfilterlib','soundfile')},'python':sys.version,'machine':platform.machine()}))"
    try:
        signature = json.loads(runner.run([python, "-c", code], "neural_signature").stdout)
        signature["model_files"] = {str(p.relative_to(model)): digest(p) for p in sorted(model.rglob("*")) if p.is_file()}
        return signature
    except (RuntimeError, OSError, ValueError):
        return None


def previews(args, runner):
    require_tools(runner)
    args.input = args.input.resolve()
    meta = probe(args.input, runner)
    all_presets = presets(args.config)
    selected = {name: all_presets[name] for name in args.presets}
    segments = json_read(args.segments)["segments"]
    disk_guard(ROOT, 2 * 1024**3)
    run = (args.run or ROOT / "output/previews").resolve()
    if run == args.input or args.input.is_relative_to(run):
        raise RuntimeError("Preview directory must not contain the source")
    run.mkdir(parents=True, exist_ok=True)
    identity = {"source": str(args.input), "source_sha256": digest(args.input), "preset_values": selected,
                "segments": segments, "neural_requested": args.neural,
                "implementation_sha256": code_hash(),
                "neural_signature": neural_signature(runner) if args.neural else None,
                "ffmpeg_version": runner.run(["ffmpeg", "-version"], "version").stdout.splitlines()[0]}
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        manifest = json_read(manifest_path)
        if not args.resume or manifest["identity"] != identity:
            raise RuntimeError("Run already exists. Use --resume with identical settings, or --run a new directory")
    else:
        manifest = {"schema_version": 1, "created_at": now(), "identity": identity,
                    "status": "in_progress", "completed": []}
        json_write(manifest_path, manifest)
    for segment in segments:
        existing = next((s for s in manifest["completed"] if s["segment"] == segment and checked_files(s)), None)
        if existing:
            print(f"Resume: verified {segment['name']}", flush=True)
            continue
        print(f"Generating {segment['name']} at {segment['start']}s", flush=True)
        item = preview_segment(args, run, segment, selected, meta, runner)
        manifest["completed"] = [s for s in manifest["completed"] if s["segment"] != segment] + [item]
        json_write(manifest_path, manifest)
        write_gallery(run, manifest)
    if digest(args.input) != identity["source_sha256"]:
        raise RuntimeError("Input changed during processing")
    manifest["status"] = "ready_for_review"
    manifest["finished_at"] = now()
    # Keep the rendering provenance when a no-op resume adds a new invocation.
    manifest.setdefault("commands_log", str(runner.log_dir))
    history = manifest.setdefault("commands_logs", [manifest["commands_log"]])
    if str(runner.log_dir) not in history:
        history.append(str(runner.log_dir))
    json_write(manifest_path, manifest)
    write_gallery(run, manifest)
    print(f"Ready for human comparison: {run / 'index.html'}")


def code_hash():
    import hashlib
    h = hashlib.sha256()
    for path in sorted((CODE_ROOT / "lecture_restore").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def reviewable_selection(manifest, preset, audio):
    if preset not in manifest["identity"]["preset_values"]:
        raise RuntimeError("Select a restoration preset represented in this preview run")
    if manifest["status"] != "ready_for_review" or len(manifest["completed"]) != len(manifest["identity"]["segments"]):
        raise RuntimeError("Previews are incomplete")
    if sorted(i["segment"]["name"] for i in manifest["completed"]) != sorted(i["name"] for i in manifest["identity"]["segments"]):
        raise RuntimeError("Representative preview coverage is incomplete")
    for item in manifest["completed"]:
        if not checked_files(item):
            raise RuntimeError("Preview files changed; regenerate before acceptance")
        v = next((v for v in item["variants"] if v["preset"] == preset and v["audio"] == audio), None)
        if not v or not v["validation"]["passed"] or v["deinterlace"].get("ambiguous"):
            raise RuntimeError("Selected candidate lacks validated previews or has unresolved interlace evidence")
        if audio == "neural":
            signature = manifest["identity"].get("neural_signature")
            if not signature or item["neural"].get("packages") != signature["packages"]:
                raise RuntimeError("Neural preview environment does not match the recorded signature")


def accept(args, runner):
    manifest_path = args.manifest.resolve()
    m = json_read(manifest_path)
    reviewed = set(args.reviewed.split(","))
    if reviewed != REVIEW_ITEMS:
        raise RuntimeError("Attest review of speech,symbols,faces,handwriting,sync,motion (mark nonpresent content after inspection)")
    if not args.note.strip():
        raise RuntimeError("Record a concrete preview comparison note")
    reviewable_selection(m, args.preset, args.audio)
    identity = m["identity"]
    if identity["implementation_sha256"] != code_hash() or digest(Path(identity["source"])) != identity["source_sha256"]:
        raise RuntimeError("Implementation or source changed; regenerate previews")
    if args.audio == "neural" and neural_signature(runner) != identity["neural_signature"]:
        raise RuntimeError("Neural environment or model changed; regenerate previews")
    acceptance = {"schema_version": 1, "accepted_at": now(), "manifest": str(manifest_path),
                  "manifest_sha256": digest(manifest_path), "identity": identity, "preset": args.preset,
                  "audio": args.audio, "reviewed": sorted(reviewed), "comparison_note": args.note,
                  "human_review_attestation": True}
    json_write(args.output.resolve(), acceptance)
    print(f"Recorded review decision: {args.output.resolve()}")


def restore(args, runner):
    require_tools(runner)
    approval = json_read(args.acceptance)
    if not approval.get("human_review_attestation") or set(approval.get("reviewed", [])) != REVIEW_ITEMS:
        raise RuntimeError("A complete preview review is required")
    mpath = Path(approval["manifest"])
    m = json_read(mpath)
    if digest(mpath) != approval["manifest_sha256"] or m["identity"] != approval["identity"]:
        raise RuntimeError("Preview manifest changed since acceptance")
    reviewable_selection(m, approval["preset"], approval["audio"])
    source = Path(approval["identity"]["source"])
    if digest(source) != approval["identity"]["source_sha256"] or code_hash() != approval["identity"]["implementation_sha256"]:
        raise RuntimeError("Source or implementation changed since previews")
    current_version = runner.run(["ffmpeg", "-version"], "version").stdout.splitlines()[0]
    if current_version != approval["identity"]["ffmpeg_version"]:
        raise RuntimeError("FFmpeg version changed since previews")
    if approval["audio"] == "neural" and neural_signature(runner) != approval["identity"].get("neural_signature"):
        raise RuntimeError("Neural environment or model changed since previews")
    p = approval["identity"]["preset_values"][approval["preset"]]
    out = args.output.resolve()
    if out == source or out.exists():
        raise RuntimeError("Output must be a new path and must differ from source")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(source, runner)
    duration = float(meta["format"]["duration"])
    # Extrapolate actual accepted preview bitrate; add 3 stereo PCM intermediates.
    sizes = [Path(v["video"]).stat().st_size / i["segment"]["duration"] for i in m["completed"]
             for v in i["variants"] if v["preset"] == approval["preset"] and v["audio"] == approval["audio"]]
    estimated = int(max(sizes) * duration * 1.5 + duration * 48000 * stream(meta, "audio")["channels"] * 3 * 3)
    disk_guard(out.parent, estimated)
    partial = out.with_name("." + out.stem + ".partial.mp4")
    if partial.exists():
        raise RuntimeError(f"Stale partial output exists: {partial}; inspect and remove it before retrying")
    try:
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=out.parent) as temp:
            work = Path(temp)
            original = work / "original.wav"
            restored = work / "restored.wav"
            matched = work / "matched.wav"
            extract_audio(source, original, 0, None, runner)
            if approval["audio"] == "neural":
                from .neural import enhance
                # Full render never silently substitutes a different approved audio path.
                neural_info = enhance(original, restored, runner, ROOT)
            else:
                runner.run(ffmpeg("-i", original, "-af", p["audio_filter"], "-c:a", "pcm_s24le", restored), "full_audio")
                neural_info = None
            stats = audio_stats(restored, runner)
            target, gains = common_gain({"restored": stats}, p["target_lufs"], p["true_peak_ceiling_db"])
            matched_stats = matched_audio(restored, matched, gains["restored"], runner)
            # Use sampled evidence consensus; refuse mixed or ambiguous scan decisions.
            decisions = [v["deinterlace"] for i in m["completed"] for v in i["variants"]
                         if v["preset"] == approval["preset"] and v["audio"] == approval["audio"]]
            if len({d["enabled"] for d in decisions}) != 1 or any(d.get("ambiguous") for d in decisions):
                raise RuntimeError("Interlace decisions differ across previews; use an explicit reviewed preset")
            vf = video_filter(p, decisions[0])
            audio_offset = float(stream(meta, "audio").get("start_time", 0)) - float(meta["format"].get("start_time", 0))
            encode(source, matched, partial, 0, None, p, vf, runner, audio_offset=audio_offset)
            report = validate(source, partial, runner, expected_duration=duration, expected_scale=p["scale"])
            report.update({"acceptance": str(args.acceptance.resolve()), "source_sha256": digest(source),
                           "output_sha256": digest(partial), "estimated_work_bytes": estimated,
                           "audio_gain_db": gains["restored"], "target_lufs": target,
                           "matched_audio_stats": matched_stats, "neural": neural_info, "video_filter": vf,
                           "output_path": str(out), "commands_log": str(runner.log_dir),
                           "publication_status": "not_published"})
            json_write(out.with_suffix(".validation.json"), report)
            if not report["passed"] or digest(source) != approval["identity"]["source_sha256"]:
                raise RuntimeError("Full output failed validation; final MP4 was not published")
            # Publish exclusively: another process creating the requested output
            # during a long encode must never have its file overwritten.
            try:
                os.link(partial, out)
            except OSError as exc:
                report["publication_error"] = str(exc)
                json_write(out.with_suffix(".validation.json"), report)
                raise
            partial.unlink()
            report["publication_status"] = "published"
            report["validated_temporary_path"] = report["candidate_path"]
            report["candidate_path"] = str(out)
            json_write(out.with_suffix(".validation.json"), report)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Restored: {out}")


def main(argv=None):
    global ROOT
    parser = argparse.ArgumentParser(description="Local review-gated lecture restoration")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="Local media, logs, reports and output directory (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--input", type=Path)
    p = sub.add_parser("previews")
    p.add_argument("--input", type=Path)
    p.add_argument("--config", type=Path, default=CODE_ROOT / "lecture_restore/presets.json")
    p.add_argument("--segments", type=Path, required=True,
                   help="JSON with representative sections selected for this recording")
    p.add_argument("--presets", nargs="+", default=["conservative", "conservative_2x"])
    p.add_argument("--run", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--neural", action="store_true")
    p = sub.add_parser("accept")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--preset", required=True)
    p.add_argument("--audio", choices=["baseline", "neural"], default="baseline")
    p.add_argument("--reviewed", required=True)
    p.add_argument("--note", required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("restore")
    p.add_argument("--acceptance", type=Path)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("validate")
    p.add_argument("candidate", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--scale", type=int, choices=[1, 2], default=1)
    p.add_argument("--duration", type=float)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    previous_root = ROOT
    ROOT = args.workspace.resolve()
    defaults = {"input": ROOT / "source/lecture.mp4",
                "manifest": ROOT / "output/previews/manifest.json",
                "acceptance": ROOT / "config/acceptance.json",
                "output": ROOT / ("config/acceptance.json" if args.command == "accept" else "output/lecture_restored.mp4"),
                "report": ROOT / "reports/validation.json"}
    for name, value in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}_{args.command}"
    runner = None
    try:
        runner = Runner(ROOT / "logs" / run_id)
        if args.command == "validate":
            report = validate(args.input.resolve(), args.candidate.resolve(), runner,
                              expected_duration=args.duration, expected_scale=args.scale, segment_start=args.start)
            json_write(args.report.resolve(), report)
            print(json.dumps({"passed": report["passed"], "report": str(args.report.resolve())}))
            return 0 if report["passed"] else 2
        globals()[args.command](args, runner)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; temporary outputs cleaned. Completed preview checkpoints can be resumed.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        if runner is not None:
            json_write(runner.log_dir / "failure.json", {"status": "failed", "error": str(exc), "at": now()})
        print(f"Error: {exc}" + (f"\nLogs: {runner.log_dir}" if runner is not None else ""), file=sys.stderr)
        return 1
    finally:
        ROOT = previous_root


if __name__ == "__main__":
    raise SystemExit(main())
