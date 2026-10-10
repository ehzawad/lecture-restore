"""Regression tests for validation calculations; no Python media packages needed."""

from array import array
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from lecture_restore.validation import (
    _accumulate_packet, _cadence, _cadence_comparison, _clip_counts,
    _display_aspect, _extract_json, _finish, _new_packet_stats, _number,
    _offsets, _parse_packet_line, _ratio, validate,
)


class TimingTests(unittest.TestCase):
    def test_b_frame_packet_order_does_not_change_cadence(self):
        source = array("d", [index / 30 for index in range(300)])
        reordered = [source[index] for index in (0, 3, 1, 2)] + list(source[4:])
        result = _cadence_comparison(source, reordered, 0, 10)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["maximum_relative_pts_error_seconds"], 0)

    def test_fractional_rate_and_constant_origin_shift(self):
        source = array("d", [100 + index * 1001 / 30000 for index in range(500)])
        selected = source[100:400]
        candidate = [pts - selected[0] for pts in selected]
        result = _cadence_comparison(source, candidate, selected[0], 300 * 1001 / 30000)
        self.assertEqual(result["status"], "pass")

    def test_variable_cadence_preserved_but_resampling_detected(self):
        source = array("d", [0, 0.033, 0.070, 0.110, 0.143, 0.176])
        self.assertTrue(_cadence(source)["variable_intervals"])
        preserved = _cadence_comparison(source, source, 0, 0.21)
        resampled = _cadence_comparison(source, [index / 30 for index in range(6)], 0, 0.21)
        self.assertEqual(preserved["status"], "pass")
        self.assertEqual(resampled["status"], "fail")

    def test_lost_interior_frame_is_detected(self):
        source = array("d", [index / 25 for index in range(100)])
        candidate = list(source[:50]) + list(source[51:])
        result = _cadence_comparison(source, candidate, 0, 4)
        self.assertEqual(result["count_difference"], -1)
        self.assertEqual(result["status"], "fail")

    def test_one_trimmed_boundary_frame_is_allowed(self):
        source = array("d", [index / 25 for index in range(100)])
        result = _cadence_comparison(source, source[1:], 0, 4)
        self.assertEqual(result["status"], "pass")

    def test_aac_priming_is_excluded_from_effective_start(self):
        stream = {"codec_type": "audio", "sample_rate": "48000"}
        stats = _new_packet_stats()
        _accumulate_packet(stats, {"pts_time": str(-1024 / 48000),
                                  "duration_time": str(1024 / 48000),
                                  "dts_time": str(-1024 / 48000),
                                  "skip_samples": "1024"}, stream)
        _accumulate_packet(stats, {"pts_time": "0", "duration_time": str(1024 / 48000),
                                  "dts_time": "0"}, stream)
        self.assertLess(stats["raw_pts_start_seconds"], 0)
        self.assertEqual(stats["effective_start_seconds"], 0)
        self.assertAlmostEqual(stats["effective_end_seconds"], 1024 / 48000)

    def test_discard_padding_is_excluded_from_effective_end(self):
        stream = {"codec_type": "audio", "sample_rate": "48000"}
        stats = _new_packet_stats()
        _accumulate_packet(stats, {"pts_time": "1", "duration_time": "0.02",
                                  "discard_padding": "480"}, stream)
        self.assertAlmostEqual(stats["effective_end_seconds"], 1.01)

    def test_mp4_presentation_duration_is_not_trimmed_twice(self):
        stream = {"codec_type": "audio", "sample_rate": "48000",
                  "_duration_excludes_discard_padding": True}
        stats = _new_packet_stats()
        _accumulate_packet(stats, {"pts_time": "1", "duration_time": "0.01",
                                  "discard_padding": "480"}, stream)
        self.assertAlmostEqual(stats["effective_end_seconds"], 1.01)

    def test_compact_side_data_works_with_old_and_new_ffprobe(self):
        older = _parse_packet_line("stream_index=1|pts_time=-0.02|skip_samples=960")
        newer = _parse_packet_line("stream_index=1|pts_time=-0.02|side_datum/skip_samples:skip_samples=960")
        self.assertEqual(older, newer)

    def test_dts_regression_is_detected_without_rejecting_reordered_pts(self):
        stats = _new_packet_stats()
        stream = {"codec_type": "video", "avg_frame_rate": "25/1"}
        for pts, dts in ((0, -0.08), (0.12, -0.04), (0.04, 0), (0.08, -0.04)):
            _accumulate_packet(stats, {"pts_time": str(pts), "dts_time": str(dts),
                                      "duration_time": "0.04"}, stream)
        self.assertEqual(stats["dts_regression_count"], 1)
        self.assertEqual(stats["missing_pts_count"], 0)
        self.assertEqual(_cadence(stats["_video_pts"])["duplicate_pts_count"], 0)

    def test_source_offsets_respect_trim_window(self):
        packets = {0: {"effective_start_seconds": 0, "effective_end_seconds": 10},
                   1: {"effective_start_seconds": 0.2, "effective_end_seconds": 9.9}}
        video, audio = {"index": 0}, {"index": 1}
        full = _offsets(packets, video, audio, 0, 10)
        middle = _offsets(packets, video, audio, 3, 8)
        self.assertAlmostEqual(full["audio_minus_video_start_seconds"], 0.2)
        self.assertAlmostEqual(full["audio_minus_video_end_seconds"], -0.1)
        self.assertEqual(middle["audio_minus_video_start_seconds"], 0)
        self.assertEqual(middle["audio_minus_video_end_seconds"], 0)


class AudioAndReportTests(unittest.TestCase):
    def test_fullscale_sample_count_sums_channels_without_using_peak_count(self):
        # One overload in the left channel, two in the right, then two silent
        # channels. Metadata is rounded to six decimal places by FFmpeg.
        lines = """frame:0 pts:0
lavfi.astats.1.DC_offset=0.000977
lavfi.astats.2.DC_offset=0.001953
lavfi.astats.Overall.Number_of_samples=1024.000000
frame:1 pts:1024
lavfi.astats.1.DC_offset=0.000000
lavfi.astats.2.DC_offset=0.000000
lavfi.astats.Overall.Number_of_samples=300.000000
""".splitlines()
        result = _clip_counts(lines)
        self.assertEqual(result["samples_at_or_above_full_scale"], 3)
        self.assertEqual(result["per_channel_samples_at_or_above_full_scale"], {"1": 1, "2": 2})
        self.assertEqual(result["decoded_sample_frames"], 1324)
        self.assertTrue(result["count_resolution_reliable"])

    def test_silent_loudnorm_json_is_json_safe(self):
        parsed = _extract_json('banner\n{ "input_tp": "-inf", "input_i": "-inf" }\n')
        self.assertIsNone(_number(parsed["input_tp"]))
        json.dumps({"peak": _number(parsed["input_tp"]), "raw": parsed}, allow_nan=False)

    def test_sample_aspect_ratio_is_part_of_display_aspect(self):
        anamorphic = {"width": 720, "height": 576, "sample_aspect_ratio": "16:15"}
        square = {"width": 1440, "height": 1152, "sample_aspect_ratio": "1:1"}
        self.assertAlmostEqual(_display_aspect(anamorphic), 4 / 3)
        self.assertNotEqual(_display_aspect(anamorphic), _display_aspect(square))

    def test_invalid_ratios_and_nonfinite_numbers_are_rejected(self):
        for value in ("0/0", "0:1", "N/A", None):
            self.assertIsNone(_ratio(value))
        for value in ("inf", "-inf", "nan", "N/A"):
            self.assertIsNone(_number(value))
        self.assertAlmostEqual(_ratio("30000/1001"), 29.97002997002997)

    def test_unavailable_measurement_does_not_pass(self):
        report = _finish({"checks": {"cadence": {"status": "unavailable"},
                                      "audio": {"status": "skip"}}})
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_checks"], ["cadence"])

    def test_tool_failure_becomes_reviewable_failed_report(self):
        class FailedRunner:
            def run(self, args, label, timeout=None):
                return subprocess.CompletedProcess(args, 1, "", "corrupt input")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"corrupt")
            report = validate(source, Path(directory) / "candidate.mp4", FailedRunner())
        self.assertFalse(report["passed"])
        self.assertIn("corrupt input", report["checks"]["media_probe"]["error"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
