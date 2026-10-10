#!/usr/bin/env python3
"""Replay an accepted, local slide-restoration recipe without copying its media.

This deliberately accepts only the reviewed 1280x720, 30000/1001 profile.
New courses need their own reviewed mapping/assets; this is not slide detection.
Verification compares complete original encoder segments, not arbitrary crops.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lecture_restore.core import Runner, digest, json_write, probe, stream
from lecture_restore.validation import validate
from scripts import restore_slides as engine


def unused(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing existing output: {path}")


def checked_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA-256 for {path}")
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Input hash mismatch: {path}")


def local_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Recipe paths must be relative to the recipe file")
    return (base / value).resolve()


def load_recipe(path: Path, reference: Path | None = None) -> dict:
    """Validate structure and every asset before any encoding or publication."""
    path = path.resolve()
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported recipe schema")
    profile = raw["profile"]
    if any(profile.get(k) != v for k, v in {
        "width": 1280, "height": 720, "frame_rate": "30000/1001", "time_base": "1/30000"
    }.items()):
        raise ValueError("Recipe requires the reviewed 1280x720, 30000/1001 profile")
    total = profile["total_frames"]
    if type(total) is not int or total <= 0:
        raise ValueError("Recipe requires a positive frame count")
    settings = raw["settings"]
    if (type(settings.get("crf")) is not int or not 10 <= settings["crf"] <= 23
            or settings.get("camera_preset") not in engine.CAMERA_PRESETS
            or not math.isfinite(settings.get("source_sharpen", math.nan))
            or not 0 <= settings["source_sharpen"] <= .5
            or not math.isfinite(settings.get("audio_gain_db", math.nan))
            or settings["audio_gain_db"] > 0):
        raise ValueError("Invalid reviewed filter or encoder settings")
    frozen = {str(path): digest(path)}
    def asset(value, expected):
        resolved = local_path(path.parent, value)
        if str(resolved) in frozen and frozen[str(resolved)] != expected:
            raise ValueError("Conflicting hashes for the same asset")
        checked_hash(resolved, expected)
        frozen[str(resolved)] = expected
        return resolved
    source = asset(raw["source"]["path"], raw["source"]["sha256"])
    asset(raw["pdf"]["path"], raw["pdf"]["sha256"])
    accepted = raw["accepted_final"]
    if not isinstance(accepted.get("approval"), str) or not accepted["approval"].strip():
        raise ValueError("Exact replay requires a recorded user acceptance of the final video")
    reference = reference.resolve() if reference else local_path(path.parent, accepted["path"])
    checked_hash(reference, accepted["sha256"])
    frozen[str(reference)] = accepted["sha256"]
    assets = {str(asset(p, h)): h for p, h in raw["assets"].items()}
    intervals = deepcopy(raw["intervals"])
    end = 0
    masks = {}
    for interval in intervals:
        a, b = interval["start_frame"], interval["end_frame"]
        if type(a) is not int or type(b) is not int or a != end or b <= a or b > total:
            raise ValueError("Intervals must cover the source contiguously and exactly once")
        end = b
        if interval["kind"] not in ("original", "pdf"):
            raise ValueError("Unknown interval kind")
        for name in ("camera_preset", "source_sharpen"):
            if interval.get(name, settings[name]) != settings[name]:
                raise ValueError("Interval settings differ from reviewed settings")
            interval[name] = settings[name]
        cameras, pointers = interval.get("camera_rectangles", []), interval.get("pointer_layers", [])
        for rectangle in interval.get("rectangles", []) + cameras + [p["rect"] for p in pointers]:
            engine.validate_rectangle(rectangle)
        expected_rectangles = cameras + [p["rect"] for p in pointers]
        if interval.get("rectangles", []) != expected_rectangles:
            raise ValueError("Layer rectangles do not match camera and pointer layers")
        if interval.get("render_path"):
            render = local_path(path.parent, interval["render_path"])
            if str(render) not in assets:
                raise ValueError("Slide has no verified asset hash")
            interval["render_path"] = str(render)
        elif interval["kind"] == "pdf":
            raise ValueError("PDF interval has no rendered slide")
        for pointer in pointers:
            first, last = pointer["mask_start_frame"], pointer["mask_end_frame"]
            if type(first) is not int or type(last) is not int or not 0 <= first <= a < b <= last <= total:
                raise ValueError("Pointer mask does not cover its interval")
            mask = local_path(path.parent, pointer["mask_path"])
            if assets.get(str(mask)) != pointer["mask_sha256"]:
                raise ValueError("Pointer mask has no matching verified asset hash")
            pointer["mask_path"] = str(mask)
            descriptor = (pointer["rect"][2:], last-first)
            if str(mask) in masks and masks[str(mask)] != descriptor:
                raise ValueError("Conflicting pointer mask geometry or frame coverage")
            masks[str(mask)] = descriptor
    if end != total or not intervals:
        raise ValueError("Intervals do not reach the source end")
    if set(raw["segment_sha256"]) != {str(i) for i in range(len(intervals))}:
        raise ValueError("Every original encoder segment needs its recorded hash")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in raw["segment_sha256"].values()):
        raise ValueError("Invalid recorded segment hash")
    return {"raw": raw, "source": source, "reference": reference, "intervals": intervals,
            "frozen": frozen, "masks": masks, "settings": settings, "total": total}


def media_preflight(recipe: dict, runner: Runner) -> dict:
    version = runner.run(["ffmpeg", "-version"], "ffmpeg-version").stdout
    if version.splitlines()[0] != recipe["raw"]["ffmpeg_version"]:
        raise ValueError("FFmpeg version differs from the accepted run; exact replay cannot be asserted")
    metadata = probe(recipe["source"], runner)
    video = stream(metadata, "video") or {}
    if (video.get("width") != 1280 or video.get("height") != 720
            or video.get("time_base") != "1/30000" or video.get("avg_frame_rate") != "30000/1001"
            or int(video.get("nb_frames", -1)) != recipe["total"]
            or video.get("sample_aspect_ratio", "1:1") != "1:1"
            or video.get("field_order") not in ("progressive", "unknown")
            or not stream(metadata, "audio")):
        raise ValueError("Source metadata differs from the reviewed profile")
    packet_scan = runner.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
                              "-show_entries", "packet=pts", "-of", "json", str(recipe["source"])], "source-cadence")
    pts = sorted(int(p["pts"]) for p in json.loads(packet_scan.stdout)["packets"])
    if pts != [i*1001 for i in range(recipe["total"])]:
        raise ValueError("Source does not have the complete reviewed frame grid")
    for path, (dimensions, count) in recipe["masks"].items():
        data = runner.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path], "mask-probe")
        mask = stream(json.loads(data.stdout), "video") or {}
        if ([mask.get("width"), mask.get("height")] != dimensions or mask.get("codec_name") != "ffv1"
                or mask.get("pix_fmt") != "gray" or int(mask.get("nb_read_frames", -1)) != count):
            raise ValueError(f"Mask geometry, codec or frame count differs: {path}")
    return {"source_metadata": metadata, "ffmpeg_version_full": version,
            "source_frame_grid_verified": True, "mask_files_verified": len(recipe["masks"])}


def decoded_hashes(path: Path, runner: Runner, first: int, last: int, label: str) -> list:
    seek = max(0, first*1001/30000-.5)
    filters = (f"settb=1/30000,trim=start_pts={first*1001}:end_pts={last*1001},"
               "showinfo@repro,setpts=N*1001")
    result = runner.run(["ffmpeg", "-hide_banner", "-nostdin", "-copyts", "-ss", f"{seek:.9f}",
                         "-threads", "2", "-i", str(path), "-map", "0:v:0", "-an", "-vf", filters,
                         "-frames:v", str(last-first), "-fps_mode", "passthrough", "-c:v", "rawvideo",
                         "-pix_fmt", "yuv420p", "-threads", "1", "-enc_time_base", "1/30000",
                         "-f", "framehash", "-hash", "sha256", "pipe:1"], label)
    pts = [int(p) for p in re.findall(r"showinfo@repro[^\n]*\bn:\s*\d+\s+pts:\s*(-?\d+)", result.stderr)]
    if pts != [i*1001 for i in range(first, last)]:
        raise ValueError(f"Decoded frame identities differ: {label}")
    rows = [[field.strip() for field in line.split(",")] for line in result.stdout.splitlines()
            if line and not line.startswith("#")]
    if len(rows) != last-first or any(len(row) != 6 for row in rows):
        raise ValueError(f"Decoded frame count differs: {label}")
    return rows


def audio_hash(recipe: dict, runner: Runner) -> dict:
    prefix = ["ffmpeg", "-hide_banner", "-nostdin"]
    suffix = ["-f", "hash", "-hash", "sha256", "pipe:1"]
    gain = recipe["settings"]["audio_gain_db"]
    regenerated = runner.run(prefix + ["-i", str(recipe["source"]), "-map", "0:a:0", "-vn", "-af",
                                      f"volume={gain}dB", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] + suffix, "audio-regenerated")
    reference = runner.run(prefix + ["-i", str(recipe["reference"]), "-map", "0:a:0", "-vn", "-c:a", "copy"] + suffix, "audio-reference")
    pattern = r"SHA256=([0-9a-f]{64})"
    a, b = re.search(pattern, regenerated.stdout), re.search(pattern, reference.stdout)
    if not a or not b:
        raise ValueError("FFmpeg did not produce the expected audio packet hash")
    return {"passed": a[1] == b[1], "regenerated_sha256": a[1], "reference_sha256": b[1],
            "scope": "All AAC packet payloads, full recording; timestamps validated separately"}


def verify_segments(recipe: dict, indices: list[int], runner: Runner, logs: Path) -> list:
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(recipe["intervals"]) for i in indices):
        raise ValueError("Invalid or duplicate segment index")
    results = []
    with tempfile.TemporaryDirectory(prefix="replay-", dir=logs.parent) as directory:
        for index in indices:
            interval = recipe["intervals"][index]
            item = engine.encode_segment(index, interval, recipe["source"], Path(directory), logs,
                                         recipe["settings"]["crf"])
            a, b = interval["start_frame"], interval["end_frame"]
            recreated = decoded_hashes(Path(item["path"]), runner, 0, b-a, f"segment-{index}-recreated")
            original = decoded_hashes(recipe["reference"], runner, a, b, f"segment-{index}-reference")
            matches = sum(left == right for left, right in zip(recreated, original))
            encoded_match = item["sha256"] == recipe["raw"]["segment_sha256"][str(index)]
            results.append({"index": index, "start_frame": a, "end_frame": b, "frame_count": b-a,
                            "encoded_sha256": item["sha256"], "historical_encoded_sha256": recipe["raw"]["segment_sha256"][str(index)],
                            "encoded_bytes_match": encoded_match, "decoded_matching_frames": matches,
                            "decoded_frames_match": matches == b-a, "passed": encoded_match and matches == b-a,
                            "source_frame_identity_verified": item["source_frame_identity_verified"]})
            print(json.dumps({"stage": "segment_verified", **results[-1]}), flush=True)
    return results


def verify_frozen(frozen: dict) -> None:
    for path, expected in frozen.items():
        checked_hash(Path(path), expected)


def render_full(recipe: dict, output: Path, work: Path, resume: bool, runner: Runner, logs: Path,
                metadata: dict, implementation: dict) -> dict:
    unused(output)
    if output.resolve() in map(Path, recipe["frozen"]):
        raise ValueError("Output cannot be a verified input")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 4*1024**3:
        raise ValueError("Full replay requires at least 4 GiB free")
    signature = hashlib.sha256(json.dumps({"inputs": recipe["frozen"], "implementation": implementation,
                                          "output": str(output), "ffmpeg": recipe["raw"]["ffmpeg_version"]}, sort_keys=True).encode()).hexdigest()
    checkpoint_path = work / "checkpoint.json"
    if work.exists():
        if not resume or not checkpoint_path.is_file() or work.is_symlink():
            raise ValueError("Existing work requires --resume and a valid checkpoint")
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("signature") != signature:
            raise ValueError("Checkpoint belongs to different inputs, implementation or output")
    else:
        work.mkdir(parents=True)
        checkpoint = {"schema_version": 1, "signature": signature, "completed": {}}
        json_write(checkpoint_path, checkpoint)
    if work.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("Work and output must share a filesystem for exclusive publication")
    intervals = recipe["intervals"]
    for key, item in checkpoint["completed"].items():
        index = int(key)
        if (not 0 <= index < len(intervals) or item.get("index") != index
                or Path(item["path"]) != work / f"segment-{index:04d}.mp4"
                or item.get("interval") != intervals[index]
                or item.get("sha256") != recipe["raw"]["segment_sha256"][key]):
            raise ValueError("Invalid checkpoint segment")
        checked_hash(Path(item["path"]), item["sha256"])
    for index, interval in enumerate(intervals):
        if str(index) in checkpoint["completed"]:
            continue
        # Only this fixed generated basename may be removed after verifying the checkpoint.
        (work / f"segment-{index:04d}.mp4").unlink(missing_ok=True)
        item = engine.encode_segment(index, interval, recipe["source"], work, logs, recipe["settings"]["crf"])
        if item["sha256"] != recipe["raw"]["segment_sha256"][str(index)]:
            raise ValueError(f"Segment {index} differs from the accepted encoding")
        checkpoint["completed"][str(index)] = item
        json_write(checkpoint_path, checkpoint)
        print(json.dumps({"stage": "full_replay", "completed_segments": len(checkpoint["completed"]),
                          "total_segments": len(intervals)}), flush=True)
    concat = work / "segments.ffconcat"
    concat.write_text("ffconcat version 1.0\n" + "".join(
        f"file 'segment-{i:04d}.mp4'\nduration {(part['end_frame']-part['start_frame'])*1001/30000:.12f}\n"
        for i, part in enumerate(intervals)))
    partial = work / "final.partial.mp4"
    partial.unlink(missing_ok=True)
    try:
        runner.run(engine.final_mux_command(concat, recipe["source"], partial, recipe["settings"]["audio_gain_db"]), "full-mux")
        result = validate(recipe["source"], partial, runner, expected_duration=float(metadata["format"]["duration"]),
                          expected_scale=2, peak_limit_db=-2.0)
        candidate_video = stream(result["candidate_metadata"], "video")
        candidate_timing = result["packet_timing"]["candidate"][str(candidate_video["index"])]
        cadence = result["checks"]["frame_cadence"]
        grid_ok = (candidate_timing["packet_count"] == recipe["total"]
                   and candidate_timing["raw_pts_start_seconds"] == 0
                   and abs(candidate_timing["raw_pts_end_seconds"]-recipe["total"]*1001/30000) < 1e-6
                   and cadence["maximum_relative_pts_error_seconds"] < 1e-6)
        result["checks"]["exact_full_source_frame_grid"] = {"status": "pass" if grid_ok else "fail"}
        if not grid_ok:
            result["failed_checks"].append("exact_full_source_frame_grid")
            result["passed"] = result["ok"] = False
            result["status"] = "failed"
        verify_frozen({**recipe["frozen"], **implementation})
        if not result["passed"]:
            raise ValueError(f"Full output failed validation: {result['failed_checks']}")
        output_hash = digest(partial)
        os.link(partial, output)
        checkpoint["published"] = {"path": str(output), "sha256": output_hash}
        json_write(checkpoint_path, checkpoint)
        result.update({"output_path": str(output), "output_sha256": output_hash,
                       "validated_temporary_path": str(partial), "candidate_path": str(output), "publication_status": "published",
                       "all_segment_bytes_match": True, "metadata_note": "Container title is now generic; whole-file hash may differ"})
        for item in checkpoint["completed"].values():
            Path(item["path"]).unlink()
        concat.unlink()
        return result
    finally:
        partial.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional relocated accepted final; its hash must match")
    parser.add_argument("--segments", type=int, nargs="+", help="Complete original segment indices; required unless --full")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {"schema_version": 1, "passed": False, "mode": "full" if args.full else "verification"}
    report_path = args.report.absolute()
    try:
        unused(report_path)
        if args.full != bool(args.output) or (args.resume or args.work) and not args.full:
            raise ValueError("Use --full with --output; --work and --resume require --full")
        if not args.full and not args.segments:
            raise ValueError("Verification requires explicit --segments from your local recipe")
        if args.full and args.segments:
            raise ValueError("Use --segments for verification or --full for all segments")
        if args.output:
            unused(args.output.absolute())
            if args.output.absolute() == report_path:
                raise ValueError("Output and report paths must differ")
        if Path(engine.__file__).resolve() != ROOT / "scripts/restore_slides.py":
            raise RuntimeError("Renderer import did not come from this repository")
        recipe = load_recipe(args.recipe, args.reference)
        if str(report_path.resolve()) in recipe["frozen"]:
            raise ValueError("Report cannot overwrite an input")
        logs = report_path.parent / (report_path.stem + "-logs-" + uuid.uuid4().hex[:8])
        runner = Runner(logs)
        implementation = {str((ROOT/p).resolve()): digest(ROOT/p) for p in (
            "scripts/replay_recipe.py", "scripts/restore_slides.py", "lecture_restore/core.py", "lecture_restore/validation.py")}
        report.update({"recipe_path": str(args.recipe.resolve()), "input_hashes": recipe["frozen"],
                       "implementation_hashes": implementation, "executing_repository": str(ROOT), "commands_directory": str(logs),
                       "settings": recipe["settings"], "accepted_final_approval": recipe["raw"]["accepted_final"]["approval"],
                       "historical_implementation_sha256": recipe["raw"].get("historical_renderer_sha256", recipe["raw"].get("historical_implementation_sha256")),
                       "limits": "Exact replay of reviewed inputs on the same FFmpeg build; does not certify unseen courses or perceptual quality"})
        preflight = media_preflight(recipe, runner)
        report["preflight"] = preflight
        if args.full:
            output = args.output.absolute()
            work = args.work.absolute() if args.work else output.parent / ("." + output.stem + "-replay")
            report["full_validation"] = render_full(recipe, output, work, args.resume, runner, logs,
                                                     preflight["source_metadata"], implementation)
            report["passed"] = report["full_validation"]["passed"]
        else:
            report["segments"] = verify_segments(recipe, args.segments, runner, logs)
            report["verified_frames"] = sum(item["frame_count"] for item in report["segments"])
            report["verified_seconds"] = report["verified_frames"]*1001/30000
            report["passed"] = all(item["passed"] for item in report["segments"])
        if args.verify_audio:
            report["audio"] = audio_hash(recipe, runner)
            report["passed"] &= report["audio"]["passed"]
        verify_frozen({**recipe["frozen"], **implementation})
        report["inputs_unchanged"] = True
        report["elapsed_seconds"] = time.monotonic()-started
        json_write(report_path, report)
        print(json.dumps({"passed": report["passed"], "report": str(report_path)}), flush=True)
        return 0 if report["passed"] else 1
    except (Exception, KeyboardInterrupt) as exc:
        report.update({"passed": False, "error": str(exc), "error_type": type(exc).__name__, "elapsed_seconds": time.monotonic()-started})
        # Never replace an existing report or another local artifact on failure.
        if not report_path.exists() and not report_path.is_symlink():
            json_write(report_path, report)
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
