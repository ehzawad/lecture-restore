"""Measured media validation using only the standard library and FFmpeg.

Timing checks detect timestamp/duration regressions, not perceptual lip sync.
Packet tables and clipping metadata are written to temporary files and read
incrementally. Only one double per video packet is retained for cadence checks.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any


# Reuse an unchanged source scan when validating several previews in one run.
# Keep this bounded: a typical two-hour 30 fps source retains about 1.7 MB of PTS.
_SOURCE_CACHE: dict[tuple[str, int, int], tuple[dict, dict]] = {}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(value: Any) -> float | None:
    try:
        result = float(Fraction(str(value).replace(":", "/")))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _primary(metadata: dict, kind: str) -> dict | None:
    return next((stream for stream in metadata.get("streams", [])
                 if stream.get("codec_type") == kind
                 and not stream.get("disposition", {}).get("attached_pic")), None)


def _fps(stream: dict | None) -> float | None:
    if stream is None:
        return None
    return _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))


def _display_aspect(stream: dict | None) -> float | None:
    if stream is None:
        return None
    width, height = _number(stream.get("width")), _number(stream.get("height"))
    if not width or not height:
        return None
    return width / height * (_ratio(stream.get("sample_aspect_ratio")) or 1.0)


def _run(runner, args: list[str], label: str, timeout: float | None = None):
    completed = runner.run(args, label=label, timeout=timeout)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise RuntimeError(f"{label} exited {completed.returncode}: {detail}")
    return completed


def _probe(path: Path, runner) -> dict:
    completed = _run(runner, ["ffprobe", "-v", "error", "-show_streams",
                              "-show_format", "-of", "json", str(path)],
                     f"validate-metadata-{path.stem}", 120)
    return json.loads(completed.stdout)


def _new_packet_stats() -> dict:
    return {"packet_count": 0, "missing_pts_count": 0,
            "missing_duration_count": 0, "dts_regression_count": 0,
            "raw_pts_start_seconds": None, "raw_pts_end_seconds": None,
            "effective_start_seconds": None, "effective_end_seconds": None,
            "_last_dts": None, "_video_pts": array("d")}


def _accumulate_packet(stats: dict, packet: dict, stream: dict) -> None:
    stats["packet_count"] += 1
    pts = _number(packet.get("pts_time"))
    dts = _number(packet.get("dts_time"))
    duration = _number(packet.get("duration_time"))
    if dts is not None:
        last_dts = stats["_last_dts"]
        if last_dts is not None and dts < last_dts - 1e-7:
            stats["dts_regression_count"] += 1
        stats["_last_dts"] = dts
    if pts is None:
        stats["missing_pts_count"] += 1
        return
    if duration is None or duration <= 0:
        stats["missing_duration_count"] += 1
        fps = _fps(stream)
        duration = 1 / fps if stream.get("codec_type") == "video" and fps else 0
    raw_end = pts + duration
    rate = _number(stream.get("sample_rate")) or 1
    skip = (_number(packet.get("skip_samples")) or 0) / rate
    # MOV/MP4 demuxing already shortens the final packet duration to its edit
    # boundary; subtracting the simultaneously reported discard_padding again
    # would remove genuine presentation samples.
    discard = (0 if stream.get("_duration_excludes_discard_padding") else
               (_number(packet.get("discard_padding")) or 0) / rate)
    start, end = pts + skip, max(pts + skip, raw_end - discard)
    for key, value, operation in (
        ("raw_pts_start_seconds", pts, min),
        ("raw_pts_end_seconds", raw_end, max),
        ("effective_start_seconds", start, min),
        ("effective_end_seconds", end, max),
    ):
        old = stats[key]
        stats[key] = value if old is None else operation(old, value)
    if stream.get("codec_type") == "video":
        stats["_video_pts"].append(pts)


def _cadence(pts) -> dict:
    ordered = sorted(pts)
    deltas = [b - a for a, b in zip(ordered, ordered[1:])]
    positive = [delta for delta in deltas if delta > 1e-7]
    median = statistics.median(positive) if positive else None
    return {
        "frame_packet_count": len(ordered),
        "duplicate_pts_count": len(deltas) - len(positive),
        "median_interval_seconds": median,
        "minimum_interval_seconds": min(positive) if positive else None,
        "maximum_interval_seconds": max(positive) if positive else None,
        "observed_average_fps": ((len(ordered) - 1) / (ordered[-1] - ordered[0])
                                 if len(ordered) > 1 and ordered[-1] > ordered[0] else None),
        "variable_intervals": bool(median and any(abs(d - median) > max(0.0001, median * 0.01)
                                                   for d in positive)),
    }


def _parse_packet_line(line: str) -> dict:
    # FFprobe 8 nests compact side-data keys as
    # side_datum/skip_samples:skip_samples; older versions use skip_samples.
    return {key.rsplit(":", 1)[-1]: value
            for key, value in (field.split("=", 1) for field in line.strip().split("|")
                               if "=" in field)}


def _scan_packets(path: Path, metadata: dict, runner) -> dict:
    mov_family = "mov" in metadata.get("format", {}).get("format_name", "").split(",")
    streams = {int(stream["index"]): {**stream, "_duration_excludes_discard_padding": mov_family}
               for stream in metadata.get("streams", [])
               if stream.get("codec_type") in ("video", "audio")}
    result = {index: _new_packet_stats() for index in streams}
    with tempfile.TemporaryDirectory(prefix="lecture-packets-") as directory:
        table = Path(directory) / "packets.txt"
        _run(runner, ["ffprobe", "-v", "error", "-show_packets", "-show_entries",
                      "packet=stream_index,pts_time,dts_time,duration_time:"
                      "packet_side_data=side_data_type,skip_samples,discard_padding",
                      "-of", "compact=p=0:nk=0", "-o", str(table), str(path)],
             f"validate-packets-{path.stem}")
        with table.open(encoding="utf-8") as handle:
            for line in handle:
                packet = _parse_packet_line(line)
                try:
                    index = int(packet.get("stream_index", "-1"))
                except ValueError:
                    continue
                if index in streams:
                    _accumulate_packet(result[index], packet, streams[index])
    for index, stats in result.items():
        stats.pop("_last_dts", None)
        stats["_video_pts"] = array("d", sorted(stats["_video_pts"]))
        if streams[index].get("codec_type") == "video":
            stats["cadence"] = _cadence(stats["_video_pts"])
        # MP4 stream duration excludes AAC trailing codec padding. Record the
        # raw demuxed endpoint above, and use the declared presentation boundary
        # only when it falls inside the final packet.
        stream_start = _number(streams[index].get("start_time"))
        stream_duration = _number(streams[index].get("duration"))
        if stream_start is not None and stream_duration is not None:
            declared_end = stream_start + stream_duration
            stats["declared_end_seconds"] = declared_end
            if stats["effective_end_seconds"] is not None:
                packet_allowance = (1024 / (_number(streams[index].get("sample_rate")) or 48000)
                                    if streams[index].get("codec_type") == "audio"
                                    else 1 / (_fps(streams[index]) or 25))
                if 0 <= stats["effective_end_seconds"] - declared_end <= packet_allowance + 1e-5:
                    stats["effective_end_seconds"] = declared_end
    return result


def _public_packets(packets: dict) -> dict:
    return {str(index): {key: value for key, value in stats.items() if not key.startswith("_")}
            for index, stats in packets.items()}


def _cadence_comparison(source_pts, candidate_pts, start: float, duration: float,
                        timebase: float = 0.00001) -> dict:
    """Compare every video packet PTS; allow one trimmed boundary frame.

    Packet PTS are sorted because B frames legitimately arrive out of order.
    Comparison removes a constant origin shift, but detects changing cadence.
    """
    lower = bisect_left(source_pts, start - 1e-7)
    upper = bisect_left(source_pts, start + duration - 1e-7)
    expected = source_pts[lower:upper]
    actual = sorted(candidate_pts)
    delta_count = len(actual) - len(expected)
    if len(expected) < 2 or len(actual) < 2:
        return {"status": "unavailable", "expected_count": len(expected),
                "actual_count": len(actual), "reason": "Fewer than two video timestamps."}
    # Seek/trim rounding can include or omit one boundary frame. Find the best
    # alignment, with at most one frame excluded at either beginning.
    best_error = math.inf
    best_alignment = None
    for source_skip in (0, 1):
        for candidate_skip in (0, 1):
            count = min(len(expected) - source_skip, len(actual) - candidate_skip)
            if count < 2:
                continue
            origin = actual[candidate_skip] - expected[source_skip]
            error = max(abs((actual[candidate_skip + i] - expected[source_skip + i]) - origin)
                        for i in range(count))
            if error < best_error:
                best_error, best_alignment = error, [source_skip, candidate_skip]
    tolerance = max(0.0001, timebase * 2)
    matched = abs(delta_count) <= 1 and best_error <= tolerance
    return {"status": "pass" if matched else "fail", "expected_count": len(expected),
            "actual_count": len(actual), "count_difference": delta_count,
            "maximum_relative_pts_error_seconds": best_error,
            "tolerance_seconds": tolerance, "boundary_alignment_skips": best_alignment,
            "method": "All packet PTS, sorted; constant origin shift removed; at most one boundary frame."}


def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "input_tp" in value:
            return value
    raise ValueError("FFmpeg did not emit loudnorm measurement JSON.")


def _clip_counts(lines) -> dict:
    """Count decoded samples >= full scale using short-frame binary means.

    FFmpeg emits DC_offset to six decimal places; rounding each frame to an
    integer is exact for normal decoded frames (< 1 million samples/frame).
    """
    per_channel: dict[str, int] = {}
    sample_frames = 0
    largest_frame = 0
    current: dict[str, str] = {}

    def flush():
        nonlocal sample_frames, largest_frame
        count = _number(current.get("lavfi.astats.Overall.Number_of_samples"))
        if count is None:
            return
        sample_frames += int(round(count))
        largest_frame = max(largest_frame, int(round(count)))
        for key, value in current.items():
            match = re.fullmatch(r"lavfi\.astats\.(\d+)\.DC_offset", key)
            mean = _number(value)
            if match and mean is not None:
                channel = match.group(1)
                per_channel[channel] = per_channel.get(channel, 0) + int(round(mean * count))

    for line in lines:
        if line.startswith("frame:"):
            flush()
            current = {}
        elif "=" in line:
            key, value = line.strip().split("=", 1)
            current[key] = value
    flush()
    if not sample_frames or not per_channel:
        raise ValueError("No decoded audio samples were measured.")
    return {"samples_at_or_above_full_scale": sum(per_channel.values()),
            "per_channel_samples_at_or_above_full_scale": per_channel,
            "decoded_sample_frames": sample_frames,
            "maximum_analysis_frame_samples": largest_frame,
            "count_resolution_reliable": largest_frame < 1_000_000,
            "threshold_amplitude": 1.0,
            "limitation": "Detects present full-scale/overload samples; cannot establish historical clipping below full scale."}


def _audio_measurements(path: Path, runner) -> dict:
    loudness = _run(runner, ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
                            "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
                            "-af", "loudnorm=I=-23:TP=-1:LRA=20:print_format=json",
                            "-f", "null", "-"], f"validate-loudness-{path.stem}")
    measured = _extract_json(loudness.stderr or "")
    with tempfile.TemporaryDirectory(prefix="lecture-clipping-") as directory:
        table = Path(directory) / "clipping.txt"
        # TemporaryDirectory paths are generated locally, and escaped as FFmpeg
        # filter values (this is never passed through a shell).
        escaped = str(table).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
        audio_filter = (
            "astats@decoded=measure_perchannel=none:"
            "measure_overall=Peak_level+Min_level+Max_level+Number_of_NaNs+Number_of_Infs+Number_of_samples,"
            "aeval='if(gte(abs(val(ch)),1),1,0)':c=same,"
            "astats@clipping=metadata=1:reset=1:measure_perchannel=DC_offset:"
            "measure_overall=Number_of_samples,"
            f"ametadata=mode=print:file='{escaped}'"
        )
        result = _run(runner, ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
                               "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
                               "-af", audio_filter, "-f", "null", "-"],
                      f"validate-clipping-{path.stem}")
        with table.open(encoding="utf-8") as handle:
            clipping = _clip_counts(handle)
    decoded = {}
    for line in (result.stderr or "").splitlines():
        match = re.match(r"\[astats@decoded[^]]*\]\s+([^:]+):\s+(\S+)", line)
        if match:
            decoded[match.group(1).strip()] = _number(match.group(2))
    # JSON uses null for unbounded/silent loudness, with the original strings
    # retained as evidence so json.dump(..., allow_nan=False) remains valid.
    return {"integrated_loudness_lufs": _number(measured.get("input_i")),
            "true_peak_dbtp": _number(measured.get("input_tp")),
            "loudness_range_lu": _number(measured.get("input_lra")),
            "loudnorm_input_measurements": {key: value for key, value in measured.items()
                                            if key.startswith("input_")},
            "decoded_sample_peak_dbfs": decoded.get("Peak level dB"),
            "decoded_nonfinite_samples": ((decoded.get("Number of NaNs") or 0)
                                          + (decoded.get("Number of Infs") or 0)),
            "decoded_astats_measurements": decoded,
            "clipping": clipping}


def _offsets(packets: dict, video: dict, audio: dict,
             cut_start: float | None = None, cut_end: float | None = None) -> dict:
    video_stats, audio_stats = packets[int(video["index"])], packets[int(audio["index"])]
    result = {}
    for name, key in (("start", "effective_start_seconds"), ("end", "effective_end_seconds")):
        video_time, audio_time = video_stats[key], audio_stats[key]
        if video_time is None or audio_time is None:
            result[f"audio_minus_video_{name}_seconds"] = None
            continue
        if cut_start is not None and name == "start":
            video_time, audio_time = max(video_time, cut_start), max(audio_time, cut_start)
        if cut_end is not None and name == "end":
            video_time, audio_time = min(video_time, cut_end), min(audio_time, cut_end)
        result[f"audio_minus_video_{name}_seconds"] = audio_time - video_time
    return result


def validate(source: Path, candidate: Path, runner,
             expected_duration: float | None = None, expected_scale: int = 1,
             segment_start: float = 0, expected_video_codec: str = "h264",
             peak_limit_db: float = -1.0) -> dict:
    """Return a JSON-safe report; failed tools become failed checks.

    ``runner.run(argv, label=..., timeout=None)`` must return a text
    subprocess.CompletedProcess. ``segment_start`` is relative to the source
    container's presentation start. Preview callers must pass expected_duration.
    ``passed`` confirms technical checks only; preview acceptance still requires
    listening and inspecting symbols, handwriting, faces, movement and lip sync.
    """
    source, candidate = Path(source), Path(candidate)
    if expected_scale not in (1, 2):
        raise ValueError("expected_scale must be 1 or 2")
    if segment_start < 0 or (expected_duration is not None and expected_duration <= 0):
        raise ValueError("segment_start must be nonnegative and expected_duration positive")
    report = {"schema_version": 1, "source_path": str(source.resolve()),
              "candidate_path": str(candidate.resolve()), "checks": {},
              "validation_scope": "First non-attached video stream and first audio stream; complete packet tables inspected.",
              "manual_review_required": [
                  "Listen for changed words, missing consonants, musical noise and unnatural speech.",
                  "Inspect mathematical symbols, handwriting, slide detail, faces and movement.",
                  "Check perceptual lip synchronization at beginning, middle and end; timestamps cannot prove lip sync.",
              ], "quality_improvement_claimed": False}
    checks = report["checks"]

    def check(name, passed, **measurements):
        checks[name] = {"status": "pass" if passed else "fail", **measurements}

    try:
        stat = source.stat()
        cache_key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = _SOURCE_CACHE.get(cache_key)
        if cached is None:
            source_metadata = _probe(source, runner)
            source_packets = _scan_packets(source, source_metadata, runner)
            _SOURCE_CACHE.clear()
            _SOURCE_CACHE[cache_key] = (source_metadata, source_packets)
        else:
            source_metadata, source_packets = cached
        candidate_metadata = _probe(candidate, runner)
        candidate_packets = _scan_packets(candidate, candidate_metadata, runner)
        report["source_metadata"] = source_metadata
        report["candidate_metadata"] = candidate_metadata
        report["packet_timing"] = {"source": _public_packets(source_packets),
                                   "candidate": _public_packets(candidate_packets)}
    except Exception as exc:
        check("media_probe", False, error=str(exc))
        return _finish(report)
    check("media_probe", True)
    source_video, source_audio = _primary(source_metadata, "video"), _primary(source_metadata, "audio")
    candidate_video, candidate_audio = _primary(candidate_metadata, "video"), _primary(candidate_metadata, "audio")
    check("video_present", source_video is not None and candidate_video is not None,
          source_has_video=source_video is not None, candidate_has_video=candidate_video is not None)
    check("audio_presence_preserved", (source_audio is not None) == (candidate_audio is not None),
          source_has_audio=source_audio is not None, candidate_has_audio=candidate_audio is not None)
    source_duration = _number(source_metadata.get("format", {}).get("duration"))
    duration = expected_duration if expected_duration is not None else source_duration
    source_origin = _number(source_metadata.get("format", {}).get("start_time")) or 0
    start = source_origin + segment_start
    source_fps = _fps(source_video)
    tolerance = max(0.05, 1 / source_fps if source_fps else 0.05)
    report["expectations"] = {"duration_seconds": duration, "scale": expected_scale,
                              "segment_start_seconds": segment_start,
                              "duration_and_av_tolerance_seconds": tolerance,
                              "video_codec": expected_video_codec,
                              "peak_limit_dbtp": peak_limit_db}
    candidate_duration = _number(candidate_metadata.get("format", {}).get("duration"))
    check("container_duration", duration is not None and candidate_duration is not None
          and abs(candidate_duration - duration) <= tolerance,
          expected_seconds=duration, actual_seconds=candidate_duration,
          difference_seconds=candidate_duration - duration if duration is not None and candidate_duration is not None else None,
          tolerance_seconds=tolerance)
    if source_video and candidate_video:
        check("video_codec", candidate_video.get("codec_name") == expected_video_codec,
              expected=expected_video_codec, actual=candidate_video.get("codec_name"))
        expected_dimensions = [int(source_video["width"]) * expected_scale,
                               int(source_video["height"]) * expected_scale]
        actual_dimensions = [candidate_video.get("width"), candidate_video.get("height")]
        check("dimensions", actual_dimensions == expected_dimensions,
              expected=expected_dimensions, actual=actual_dimensions)
        source_aspect, candidate_aspect = _display_aspect(source_video), _display_aspect(candidate_video)
        check("display_aspect_ratio", bool(source_aspect and candidate_aspect
                                            and abs(source_aspect / candidate_aspect - 1) <= 0.001),
              source=source_aspect, candidate=candidate_aspect,
              source_sar=source_video.get("sample_aspect_ratio"),
              candidate_sar=candidate_video.get("sample_aspect_ratio"),
              source_dar=source_video.get("display_aspect_ratio"),
              candidate_dar=candidate_video.get("display_aspect_ratio"))
        # Average metadata fps can change slightly for a short VFR excerpt.
        # Packet timestamp cadence below is the stronger preservation test.
        report["frame_rate"] = {
            "source_nominal": source_video.get("r_frame_rate"),
            "source_average": source_video.get("avg_frame_rate"),
            "candidate_nominal": candidate_video.get("r_frame_rate"),
            "candidate_average": candidate_video.get("avg_frame_rate"),
            "method": "Preservation assessed against source segment packet PTS; metadata alone is insufficient for VFR."}
        if duration is not None:
            timebase = max(_ratio(source_video.get("time_base")) or 1e-5,
                           _ratio(candidate_video.get("time_base")) or 1e-5)
            checks["frame_cadence"] = _cadence_comparison(
                source_packets[int(source_video["index"])]["_video_pts"],
                candidate_packets[int(candidate_video["index"])]["_video_pts"], start, duration, timebase)
        else:
            check("frame_cadence", False, error="Expected duration is unavailable.")
    for stream in (candidate_video, candidate_audio):
        if not stream:
            continue
        kind = stream["codec_type"]
        stats = candidate_packets[int(stream["index"])]
        check(f"{kind}_packet_timestamps", stats["packet_count"] > 0
              and stats["missing_pts_count"] == 0 and stats["dts_regression_count"] == 0,
              packet_count=stats["packet_count"], missing_pts_count=stats["missing_pts_count"],
              dts_regression_count=stats["dts_regression_count"],
              missing_duration_count=stats["missing_duration_count"])
    if source_video and source_audio and candidate_video and candidate_audio and duration is not None:
        expected_offsets = _offsets(source_packets, source_video, source_audio, start, start + duration)
        actual_offsets = _offsets(candidate_packets, candidate_video, candidate_audio)
        for boundary in ("start", "end"):
            key = f"audio_minus_video_{boundary}_seconds"
            expected, actual = expected_offsets[key], actual_offsets[key]
            check(f"av_{boundary}_offset", expected is not None and actual is not None
                  and abs(actual - expected) <= tolerance,
                  expected_seconds=expected, actual_seconds=actual,
                  difference_seconds=actual - expected if actual is not None and expected is not None else None,
                  tolerance_seconds=tolerance,
                  limitation="Timestamp consistency does not prove perceptual lip sync.")
    else:
        checks["av_start_offset"] = checks["av_end_offset"] = {
            "status": "skip", "reason": "Audio/video pair or expected duration unavailable."}
    try:
        completed = _run(runner, ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror",
                                  "-i", str(candidate), "-map", "0:v:0?", "-map", "0:a:0?",
                                  "-sn", "-dn", "-f", "null", "-"],
                         f"validate-decode-{candidate.stem}")
        check("complete_decode", not (completed.stderr or "").strip(),
              errors=(completed.stderr or "").strip()[-4000:])
    except Exception as exc:
        check("complete_decode", False, error=str(exc))
    if candidate_audio:
        try:
            measurements = _audio_measurements(candidate, runner)
            report["audio_measurements"] = measurements
            true_peak = measurements["true_peak_dbtp"]
            # A genuinely silent stream has input_tp=-inf and no finite peak.
            silent = measurements["loudnorm_input_measurements"].get("input_tp") == "-inf"
            check("audio_true_peak", silent or (true_peak is not None and true_peak <= peak_limit_db + 0.05),
                  actual_dbtp=true_peak, limit_dbtp=peak_limit_db,
                  measurement_rounding_tolerance_db=0.05, silent=silent)
            clipping = measurements["clipping"]
            check("decoded_audio_clipping", clipping["count_resolution_reliable"]
                  and clipping["samples_at_or_above_full_scale"] == 0,
                  **clipping)
            check("finite_audio_samples", measurements["decoded_nonfinite_samples"] == 0,
                  nonfinite_samples=measurements["decoded_nonfinite_samples"])
        except Exception as exc:
            check("audio_measurement", False, error=str(exc))
    else:
        checks["audio_true_peak"] = {"status": "skip", "reason": "Candidate has no audio stream."}
        checks["decoded_audio_clipping"] = {"status": "skip", "reason": "Candidate has no audio stream."}
    return _finish(report)


def _finish(report: dict) -> dict:
    report["failed_checks"] = [name for name, result in report["checks"].items()
                               if result["status"] not in ("pass", "skip")]
    report["passed"] = not report["failed_checks"]
    report["ok"] = report["passed"]
    report["status"] = "passed" if report["passed"] else "failed"
    return report
