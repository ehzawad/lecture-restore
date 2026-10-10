#!/usr/bin/env python3
"""Restore a complete lecture using matched original slides and source overlays.

Uncertain sections retain the original video. Original speech receives constant
gain only. Checkpointed video chunks can resume; final publication is exclusive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lecture_restore.core import Runner, digest, json_write, probe, stream
from lecture_restore.validation import validate


CAMERA_PRESETS = {
    "none": ("", ""),
    "mild_sharp": ("", "unsharp=5:5:0.30:3:3:0,limiter=min=16:max=235:planes=1"),
}


def validate_rectangle(rect):
    if len(rect) != 4 or any(type(n) is not int for n in rect):
        raise ValueError("Source rectangles require integer [x,y,width,height]")
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1280 or y+h > 720 or any(n % 2 for n in rect):
        raise ValueError("Source rectangles must be even and inside 1280×720")


def validate_plan(plan: dict) -> None:
    if plan.get("frame_rate") != "30000/1001" or plan.get("time_base") != "1/30000":
        raise ValueError("This renderer requires the reviewed 30000/1001 source cadence")
    if type(plan.get("total_frames")) is not int or plan["total_frames"] <= 0:
        raise ValueError("Missing positive source frame count")
    end = 0
    for interval in plan["intervals"]:
        a, b = interval["start_frame"], interval["end_frame"]
        if type(a) is not int or type(b) is not int or a != end or b <= a:
            raise ValueError("Intervals must cover every source frame exactly once")
        if interval["kind"] not in ("pdf", "original"):
            raise ValueError("Unknown interval kind")
        if interval["kind"] == "pdf" and not interval.get("render_path"):
            raise ValueError("Matched intervals need verified slide renders")
        end = b
    if end != plan["total_frames"]:
        raise ValueError("Plan does not cover the complete source video")


def merge_intervals(plan: dict, overlays: dict) -> list[dict]:
    """Split slide intervals at every observed source-overlay state change."""
    states = overlays["intervals"]
    end = 0
    for state in states:
        a, b = state["start_frame"], state["end_frame"]
        if type(a) is not int or type(b) is not int or a != end or b <= a:
            raise ValueError("Overlay states must cover every source frame exactly once")
        for rect in state.get("camera_rectangles", state.get("rectangles", [])):
            validate_rectangle(rect)
        for layer in state.get("pointer_layers", []):
            validate_rectangle(layer["rect"])
            ma, mb = layer["mask_start_frame"], layer["mask_end_frame"]
            if (type(ma) is not int or type(mb) is not int or ma > a or mb < b
                    or not layer.get("mask_path") or len(layer.get("mask_sha256", "")) != 64):
                raise ValueError("Pointer mask must identify and cover its complete source interval")
        end = b
    if end != plan["total_frames"]:
        raise ValueError("Overlay states do not cover the whole source")
    merged = []
    j = 0
    for interval in plan["intervals"]:
        start = interval["start_frame"]
        while start < interval["end_frame"]:
            while states[j]["end_frame"] <= start:
                j += 1
            state = states[j]
            stop = min(interval["end_frame"], state["end_frame"])
            item = {**interval, "start_frame": start, "end_frame": stop,
                    "rectangles": state.get("rectangles", []), "overlay_reason": state.get("reason", "")}
            if "camera_rectangles" in state or "pointer_layers" in state:
                item["camera_rectangles"] = state.get("camera_rectangles", [])
                item["pointer_layers"] = state.get("pointer_layers", [])
                # Retain bounding rectangles for existing static fidelity tools;
                # compositing itself uses the sparse masks below.
                item["rectangles"] = item["camera_rectangles"] + [p["rect"] for p in item["pointer_layers"]]
            if state.get("fallback_original"):
                item["kind"] = "original"
                item["fallback_reason"] = state.get("reason", "Uncertain live content")
            merged.append(item)
            start = stop
    return merged


def segment_filter(interval: dict, rectangles: list[list[int]]) -> str:
    a, b = interval["start_frame"], interval["end_frame"]
    source = (f"[0:v]settb=1/30000,trim=start_pts={a*1001}:end_pts={b*1001},"
              "showinfo@source_trim,setpts=N*1001")
    if interval["kind"] == "original":
        strength = interval.get("source_sharpen", 0)
        detail = f",unsharp=5:5:{strength}:3:3:0,limiter=min=16:max=235:planes=1" if strength else ""
        return source + ",scale=2560:1440:flags=lanczos" + detail + ",setsar=1,format=yuv420p[v]"
    pdf_index = 1 if rectangles else 0
    pdf = (f"[{pdf_index}:v]trim=end_frame={b-a},settb=1/30000,setpts=N*1001,"
           "scale=2560:1440:out_color_matrix=bt709:out_range=tv,setsar=1,format=yuv420p")
    if not rectangles:
        return pdf + "[v]"
    pieces = [source + ",setsar=1,format=yuv420p[s]",
              pdf + "[p0]"]
    cameras = interval.get("camera_rectangles", rectangles)
    pointers = interval.get("pointer_layers", [])
    layers = [(rect, None) for rect in cameras] + [(p["rect"], j) for j,p in enumerate(pointers)]
    if len(layers) > 1:
        pieces.append("[s]split=" + str(len(layers)) + "".join(f"[s{i}]" for i in range(len(layers))))
    for i, ((x, y, w, h), pointer_index) in enumerate(layers):
        label = f"s{i}" if len(layers) > 1 else "s"
        if pointer_index is None:
            pre, post = CAMERA_PRESETS[interval.get("camera_preset", "none")]
            filters = [f"crop={w}:{h}:{x}:{y}", pre, "scale=iw*2:ih*2:flags=lanczos", post]
            pieces.append(f"[{label}]" + ",".join(f for f in filters if f) + f"[c{i}]")
        else:
            pointer = pointers[pointer_index]
            first = a - pointer["mask_start_frame"]
            # FFV1/Matroska timestamps may be millisecond based. Mask identity
            # is bound to source frame indices, so normalize after index trim.
            pieces.append(f"[{2+pointer_index}:v]trim=start_frame={first}:end_frame={first+b-a},"
                          f"settb=1/30000,setpts=N*1001,showinfo@mask_{pointer_index},"
                          "scale=iw*2:ih*2:flags=bilinear,format=gray[m" + str(i) + "]")
            pieces.append(f"[{label}]crop={w}:{h}:{x}:{y},scale=iw*2:ih*2:flags=lanczos,format=yuva420p[pc{i}]")
            pieces.append(f"[pc{i}][m{i}]alphamerge=shortest=1:repeatlast=0[c{i}]")
        out = "v" if i == len(layers)-1 else f"p{i+1}"
        pieces.append(f"[p{i}][c{i}]overlay={x*2}:{y*2}:shortest=1:eof_action=pass:repeatlast=0[{out}]")
    return ";".join(pieces)


def segment_command(source: Path, interval: dict, candidate: Path, crf: int) -> list[str]:
    rectangles = interval.get("rectangles", [])
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-filter_complex_threads", "2"]
    if interval["kind"] == "original" or rectangles:
        seek = max(0, interval["start_frame"] * 1001 / 30000 - 0.5)
        command += ["-copyts", "-ss", f"{seek:.9f}", "-threads", "2", "-i", str(source)]
    if interval["kind"] == "pdf":
        command += ["-loop", "1", "-framerate", "30000/1001", "-threads", "1", "-i", interval["render_path"]]
        for layer in interval.get("pointer_layers", []):
            command += ["-threads", "1", "-i", layer["mask_path"]]
    vf = segment_filter(interval, rectangles) + ";[v]setparams=field_mode=prog:range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709[encoded]"
    command += ["-filter_complex", vf, "-map", "[encoded]", "-an",
                "-frames:v", str(interval["end_frame"]-interval["start_frame"]),
                "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-threads", "4", "-bf", "0",
                "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "1/30000",
                "-video_track_timescale", "30000", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range", "tv", str(candidate)]
    return command


def encode_segment(index, interval, source, work, logs, crf):
    runner = Runner(logs / f"segment-{index:04d}")
    output = work / f"segment-{index:04d}.mp4"
    partial = work / f"segment-{index:04d}.partial.mp4"
    partial.unlink(missing_ok=True)
    try:
        completed = runner.run(segment_command(source, interval, partial, crf), "encode")
        metadata = probe(partial, runner)
        video = stream(metadata, "video")
        expected = interval["end_frame"] - interval["start_frame"]
        if (video.get("codec_name") != "h264" or int(video.get("nb_frames", -1)) != expected
                or int(video.get("has_b_frames", -1)) != 0 or [video.get("width"), video.get("height")] != [2560, 1440]
                or video.get("time_base") != "1/30000"):
            raise RuntimeError(f"Segment {index} has unexpected codec, frame count, order or dimensions")
        source_pts = [int(n) for n in re.findall(r"showinfo@source_trim[^\n]*\bn:\s*\d+\s+pts:\s*(-?\d+)", completed.stderr)]
        if interval["kind"] == "original" or interval.get("rectangles"):
            if source_pts != [n*1001 for n in range(interval["start_frame"], interval["end_frame"])]:
                raise RuntimeError(f"Segment {index} source frame identity/order failed")
        if interval["kind"] == "pdf":
            for j, _ in enumerate(interval.get("pointer_layers", [])):
                mask_pts = [int(n) for n in re.findall(rf"showinfo@mask_{j}[^\n]*\bn:\s*\d+\s+pts:\s*(-?\d+)", completed.stderr)]
                if mask_pts != [n*1001 for n in range(expected)]:
                    raise RuntimeError(f"Segment {index} pointer mask frame coverage failed")
        if output.exists():
            raise FileExistsError(f"Refusing existing untracked segment: {output}")
        os.link(partial, output)
        return {"index": index, "path": str(output), "sha256": digest(output), "frame_count": expected,
                "source_frame_identity_verified": bool(source_pts), "interval": interval,
                "commands_log": str(runner.log_dir)}
    finally:
        partial.unlink(missing_ok=True)


def final_mux_command(concat, source, candidate, gain):
    return ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-bsf:v", "setts=pts=N*1001:dts=N*1001:duration=1001:time_base=1/30000",
            "-video_track_timescale", "30000", "-af", f"volume={gain}dB", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", "-metadata", "title=Restored lecture",
            "-metadata", "comment=Matched original PDF slides with source lecturer overlays and source fallback; original speech with constant gain only",
            str(candidate)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=ROOT / "reports/slide_alignment.json")
    parser.add_argument("--overlays", type=Path, default=ROOT / "reports/overlay_analysis.json")
    parser.add_argument("--approval", type=Path, default=ROOT / "config/full_slide_acceptance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/lecture_restored.mp4")
    parser.add_argument("--work", type=Path, default=ROOT / "output/.lecture_restore_work")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--audio-gain-db", type=float, default=-2.3)
    parser.add_argument("--camera-preset", choices=tuple(CAMERA_PRESETS), default="none")
    parser.add_argument("--source-sharpen", type=float, default=0, help="Mild luma detail for genuine source-only sections, 0–0.5")
    args = parser.parse_args(argv)
    logs = ROOT / "logs" / ("full-slides-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    runner = Runner(logs)
    published = False
    partial = None
    try:
        if (not 10 <= args.crf <= 23 or not math.isfinite(args.audio_gain_db) or args.audio_gain_db > 0
                or not math.isfinite(args.source_sharpen) or not 0 <= args.source_sharpen <= 0.5):
            raise ValueError("Use CRF 10–23 and finite nonpositive original-audio gain")
        out, work = args.output.absolute(), args.work.absolute()
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"Output already exists: {out}")
        plan = json.loads(args.alignment.read_text())
        validate_plan(plan)
        overlays = json.loads(args.overlays.read_text())
        if overlays.get("alignment_provenance", {}).get("sha256", digest(args.alignment)) != digest(args.alignment):
            raise ValueError("Overlay analysis belongs to a different slide timeline")
        intervals = merge_intervals(plan, overlays)
        for interval in intervals:
            interval["camera_preset"] = args.camera_preset
            interval["source_sharpen"] = args.source_sharpen
        source = (ROOT / plan["source"]["path"]).resolve()
        pdf = (ROOT / plan["pdf"]["path"]).resolve()
        for path, expected in [(source, plan["source"]["sha256"]), (pdf, plan["pdf"]["sha256"])]:
            if digest(path) != expected:
                raise ValueError(f"Verified input changed: {path}")
        approval = json.loads(args.approval.read_text())
        if (approval.get("source_sha256") != plan["source"]["sha256"]
                or approval.get("pdf_sha256") != plan["pdf"]["sha256"]
                or not approval.get("user_full_render_instruction")
                or not approval.get("requires_full_structural_validation")):
            raise ValueError("Full-render preview approval does not match these inputs")
        for name in ("approved_preview", "approved_comparison"):
            asset = approval[name]
            if digest((ROOT / asset["path"]).resolve()) != asset["sha256"]:
                raise ValueError("Approved preview/comparison changed")
        if args.camera_preset != "none" or args.source_sharpen or any(i.get("pointer_layers") for i in intervals):
            review = approval.get("revision_preview_review", {})
            configuration = {"alignment_sha256": digest(args.alignment), "overlays_sha256": digest(args.overlays),
                             "camera_preset": args.camera_preset, "source_sharpen": args.source_sharpen, "crf": args.crf}
            if (not approval.get("user_revision_instruction") or review.get("status") != "passed"
                    or review.get("configuration") != configuration or not review.get("artifacts")):
                raise ValueError("Enhanced full rendering requires a passed preview comparison of these exact settings")
            for asset in review["artifacts"]:
                if digest((ROOT/asset["path"]).resolve()) != asset["sha256"]:
                    raise ValueError("Reviewed revision preview changed")
        slide_hashes = {}
        for item in plan["slides"]:
            path = (ROOT / item["path"]).resolve()
            if digest(path) != item["sha256"]:
                raise ValueError(f"Verified render changed: {path}")
            slide_hashes[str(path)] = item["sha256"]
        mask_hashes = {}
        for interval in intervals:
            if interval["kind"] == "pdf":
                interval["render_path"] = str((ROOT / interval["render_path"]).resolve())
                if interval["render_path"] not in slide_hashes:
                    raise ValueError("Unverified slide path in interval")
                for layer in interval.get("pointer_layers", []):
                    path = (ROOT / layer["mask_path"]).resolve()
                    layer["mask_path"] = str(path)
                    if str(path) not in mask_hashes:
                        if digest(path) != layer["mask_sha256"]:
                            raise ValueError(f"Pointer mask changed: {path}")
                        mask_meta = json.loads(runner.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)], "mask-probe").stdout)
                        video = stream(mask_meta, "video")
                        if ([video.get("width"), video.get("height")] != layer["rect"][2:]
                                or int(video.get("nb_read_frames", -1)) != layer["mask_end_frame"]-layer["mask_start_frame"]
                                or video.get("codec_name") != "ffv1" or video.get("pix_fmt") != "gray"):
                            raise ValueError(f"Pointer mask metadata/coverage mismatch: {path}")
                        mask_hashes[str(path)] = layer["mask_sha256"]
        version = runner.run(["ffmpeg", "-version"], "ffmpeg-version").stdout.splitlines()[0]
        metadata = probe(source, runner)
        source_video = stream(metadata, "video")
        if (int(source_video.get("nb_frames", -1)) != plan["total_frames"]
                or source_video.get("time_base") != "1/30000" or source_video.get("avg_frame_rate") != "30000/1001"
                or [source_video.get("width"), source_video.get("height")] != [1280, 720]):
            raise ValueError("Source media does not match the complete reviewed frame plan")
        if out.resolve() == source or work.resolve() == source.parent:
            raise ValueError("Unsafe output/work path")
        identity = {"source_sha256": digest(source), "pdf_sha256": digest(pdf), "slide_hashes": slide_hashes,
                    "alignment_sha256": digest(args.alignment), "overlays_sha256": digest(args.overlays),
                    "implementation_sha256": digest(Path(__file__)),
                    "imported_implementation_hashes": {n: digest(ROOT / n) for n in ("lecture_restore/core.py", "lecture_restore/validation.py")},
                    "approval_sha256": digest(args.approval), "approval_path": str(args.approval.resolve()), "ffmpeg_version": version,
                    "crf": args.crf, "audio_gain_db": args.audio_gain_db, "intervals": intervals,
                    "pointer_mask_hashes": mask_hashes, "camera_preset": args.camera_preset,
                    "source_sharpen": args.source_sharpen}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        out.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(out.parent).free < 4*1024**3:
            raise ValueError("Free at least 4 GiB before full rendering")
        checkpoint_path = work / "checkpoint.json"
        if work.exists():
            if not args.resume or not checkpoint_path.is_file():
                raise ValueError("Existing work requires --resume and a valid checkpoint")
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint["signature"] != signature:
                raise ValueError("Inputs, mapping, overlays, implementation or options changed since checkpoint")
        else:
            work.mkdir(parents=True)
            checkpoint = {"schema_version": 1, "signature": signature, "identity": identity, "completed": {}, "logs": []}
        checkpoint["logs"].append(str(logs))
        json_write(checkpoint_path, checkpoint)
        for key, item in checkpoint["completed"].items():
            index = int(key)
            if (not 0 <= index < len(intervals) or item.get("index") != index
                    or Path(item["path"]).absolute() != work / f"segment-{index:04d}.mp4"
                    or item.get("interval") != intervals[index]
                    or item.get("frame_count") != intervals[index]["end_frame"] - intervals[index]["start_frame"]):
                raise ValueError(f"Invalid checkpoint segment identity: {key}")
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Checkpoint segment changed: {key}")
            checked_video = stream(probe(Path(item["path"]), runner), "video")
            if (checked_video.get("codec_name") != "h264" or int(checked_video.get("has_b_frames", -1)) != 0
                    or int(checked_video.get("nb_frames", -1)) != item["frame_count"]
                    or checked_video.get("time_base") != "1/30000"):
                raise ValueError(f"Invalid checkpoint media: {key}")
        missing = [(i, interval) for i, interval in enumerate(intervals) if str(i) not in checkpoint["completed"]]
        for i, _ in missing:
            # A crash between chunk publication and checkpointing can leave an
            # unregistered generated chunk. Its identity is untrusted; regenerate
            # only this fixed filename inside the signature-verified work folder.
            (work / f"segment-{i:04d}.mp4").unlink(missing_ok=True)
        print(json.dumps({"stage": "render", "segments": len(intervals), "remaining": len(missing), "work": str(work)}), flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(encode_segment, i, interval, source, work, logs, args.crf): i for i, interval in missing}
            for future in as_completed(futures):
                try:
                    item = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    failures.append(exc)
                    for pending in futures:
                        pending.cancel()
                    continue
                checkpoint["completed"][str(item["index"])] = item
                json_write(checkpoint_path, checkpoint)
                done_frames = sum(s["frame_count"] for s in checkpoint["completed"].values())
                print(json.dumps({"stage": "render", "completed_segments": len(checkpoint["completed"]),
                                  "total_segments": len(intervals), "completed_frames": done_frames,
                                  "total_frames": plan["total_frames"]}), flush=True)
        if failures:
            raise failures[0]
        concat = work / "segments.ffconcat"
        lines = ["ffconcat version 1.0"]
        for i in range(len(intervals)):
            item = checkpoint["completed"][str(i)]
            # Paths are our fixed safe segment basenames inside the work directory.
            lines += [f"file '{Path(item['path']).name}'", f"duration {item['frame_count']*1001/30000:.12f}"]
        concat.write_text("\n".join(lines)+"\n")
        partial = work / "final.partial.mp4"
        partial.unlink(missing_ok=True)
        print(json.dumps({"stage": "mux_original_audio"}), flush=True)
        runner.run(final_mux_command(concat, source, partial, args.audio_gain_db), "final-mux")
        print(json.dumps({"stage": "full_validation"}), flush=True)
        report = validate(source, partial, runner, expected_duration=float(metadata["format"]["duration"]), expected_scale=2, peak_limit_db=-2.0)
        source_timing = report["packet_timing"]["source"][str(source_video["index"])]
        candidate_video = stream(report["candidate_metadata"], "video")
        candidate_timing = report["packet_timing"]["candidate"][str(candidate_video["index"])]
        cadence = report["checks"]["frame_cadence"]
        grid_ok = (candidate_timing["packet_count"] == source_timing["packet_count"] == plan["total_frames"]
                   and candidate_timing["raw_pts_start_seconds"] == 0
                   and abs(candidate_timing["raw_pts_end_seconds"]-plan["total_frames"]*1001/30000) < 1e-6
                   and cadence["maximum_relative_pts_error_seconds"] < 1e-6)
        report["checks"]["exact_full_source_frame_grid"] = {"status": "pass" if grid_ok else "fail", "expected_frames": plan["total_frames"], "actual_frames": candidate_timing["packet_count"]}
        inputs_intact = (digest(source) == identity["source_sha256"] and digest(pdf) == identity["pdf_sha256"]
                         and all(digest(Path(p)) == h for p, h in slide_hashes.items())
                         and all(digest(Path(p)) == h for p, h in mask_hashes.items())
                         and digest(args.alignment) == identity["alignment_sha256"]
                         and digest(args.overlays) == identity["overlays_sha256"]
                         and digest(args.approval) == identity["approval_sha256"]
                         and digest(Path(__file__)) == identity["implementation_sha256"]
                         and all(digest(ROOT/n) == h for n,h in identity["imported_implementation_hashes"].items()))
        report["checks"]["input_assets_unchanged"] = {"status": "pass" if inputs_intact else "fail"}
        report["failed_checks"] = [k for k, v in report["checks"].items() if v["status"] not in ("pass", "skip")]
        report["passed"] = report["ok"] = not report["failed_checks"]
        report["status"] = "passed" if report["passed"] else "failed"
        report.update({"artifact_type": "full_lecture_original_slide_restoration", "publication_status": "not_published",
                       "output_path": str(out), "output_sha256": digest(partial), "identity": identity,
                       "checkpoint": str(checkpoint_path), "commands_logs": checkpoint["logs"],
                       "source_unchanged": inputs_intact, "audio_processing": {"constant_gain_db": args.audio_gain_db, "neural_or_denoising": False, "aac_reencoding": True},
                       "matched_pdf_frames": sum(s["end_frame"]-s["start_frame"] for s in intervals if s["kind"] == "pdf"),
                       "original_fallback_intervals": [s for s in intervals if s["kind"] == "original"],
                       "content_verification_limits": overlays.get("limitations", []) + plan.get("limitations", [])})
        report_path = out.with_suffix(".validation.json")
        json_write(report_path, report)
        if not report["passed"]:
            raise RuntimeError(f"Full output validation failed: {report['failed_checks']}")
        try:
            os.link(partial, out)
        except OSError as exc:
            report["publication_error"] = str(exc)
            json_write(report_path, report)
            raise
        published = True
        partial.unlink()
        report.update({"publication_status": "published", "candidate_path": str(out)})
        json_write(report_path, report)
        checkpoint["published"] = {"path": str(out), "sha256": report["output_sha256"]}
        json_write(checkpoint_path, checkpoint)
        # Keep the small checkpoint and logs; completed media is now in the final file.
        for item in checkpoint["completed"].values():
            Path(item["path"]).unlink()
        print(json.dumps({"stage": "complete", "video": str(out), "validation": str(report_path), "passed": True}), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if partial is not None:
            partial.unlink(missing_ok=True)
        json_write(logs / "failure.json", {"error": str(exc), "type": type(exc).__name__, "published": published})
        print(f"Full slide restoration failed: {exc}; logs: {logs}; completed chunks retained for --resume", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
