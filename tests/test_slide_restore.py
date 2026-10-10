"""Measured full-slide renderer tests, confined to synthetic temporary assets.

The video encodes five unmistakable color states. Comparing source and output
inset pixels detects lost, reordered or frozen source frames even when packet
timestamps have been regenerated. A synthetic approval cannot approve a lecture.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import logging
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from lecture_restore.core import Runner, digest, ffmpeg, json_read, json_write, probe, stream
from scripts import restore_slides as renderer


class PlanGuardTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"frame_rate": "30000/1001", "time_base": "1/30000", "total_frames": 72,
                     "intervals": [{"start_frame": 0, "end_frame": 24, "kind": "pdf", "render_path": "one.png"},
                                   {"start_frame": 24, "end_frame": 72, "kind": "original"}]}
        self.overlays = {"intervals": [{"start_frame": 0, "end_frame": 72,
                                       "rectangles": [[1000, 80, 200, 140]]}]}

    def test_gaps_overlaps_reversed_intervals_and_wrong_cadence_are_rejected(self):
        renderer.validate_plan(self.plan)
        for field, value in (("start_frame", 25), ("start_frame", 23), ("end_frame", 24)):
            with self.subTest(field=field, value=value):
                bad = deepcopy(self.plan)
                bad["intervals"][1][field] = value
                with self.assertRaises(ValueError):
                    renderer.validate_plan(bad)
        for key, value in (("total_frames", 73), ("frame_rate", "30/1"), ("time_base", "1/1000")):
            with self.subTest(field=key, value=value):
                bad = deepcopy(self.plan)
                bad[key] = value
                with self.assertRaises(ValueError):
                    renderer.validate_plan(bad)

    def test_uncertain_live_interval_overrides_pdf_and_preserves_complete_coverage(self):
        overlays = {"intervals": [{"start_frame": 0, "end_frame": 12, "rectangles": []},
                                   {"start_frame": 12, "end_frame": 30, "rectangles": [],
                                    "fallback_original": True, "reason": "unresolved annotation"},
                                   {"start_frame": 30, "end_frame": 72, "rectangles": []}]}
        merged = renderer.merge_intervals(self.plan, overlays)
        self.assertEqual([(part["start_frame"], part["end_frame"], part["kind"]) for part in merged],
                         [(0, 12, "pdf"), (12, 24, "original"), (24, 30, "original"), (30, 72, "original")])
        self.assertEqual(sum(part["end_frame"] - part["start_frame"] for part in merged), 72)

    def test_missing_overlay_coverage_and_invalid_rectangles_are_rejected(self):
        for rectangle in ([1001, 80, 200, 140], [1000, 80, 300, 140], [0, 0, 0, 20], [0, 0, 20]):
            with self.subTest(rectangle=rectangle):
                bad = deepcopy(self.overlays)
                bad["intervals"][0]["rectangles"] = [rectangle]
                with self.assertRaises(ValueError):
                    renderer.merge_intervals(self.plan, bad)
        bad = deepcopy(self.overlays)
        bad["intervals"][0]["end_frame"] = 71
        with self.assertRaises(ValueError):
            renderer.merge_intervals(self.plan, bad)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not installed")
class MeasuredSlideRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lecture-full-slides-test-")
        cls.root = Path(cls.temporary.name).resolve()
        cls.addClassCleanup(cls._cleanup)
        cls.fixture_runner = Runner(cls.root / "logs/fixture")
        cls.source = cls.root / "source/synthetic.mp4"
        cls.source.parent.mkdir()
        # Preserve source metadata conventions while changing pixel identity
        # several times inside encoded chunks. Source B frames exercise seeking.
        colors = [(0, 12, "red"), (12, 28, "green"), (28, 44, "blue"),
                  (44, 60, "yellow"), (60, 72, "magenta")]
        filters = [f"drawbox=x=0:y=0:w=iw:h=ih:color={color}:t=fill:enable='gte(n,{a})*lt(n,{b})'"
                   for a, b, color in colors]
        cls.fixture_runner.run(ffmpeg(
            "-f", "lavfi", "-i", "color=black:size=1280x720:rate=30000/1001:duration=2.4024",
            "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=44100:duration=2.56",
            "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "fast",
            "-threads", "2", "-bf", "2", "-pix_fmt", "yuv420p", "-enc_time_base", "1/30000",
            "-video_track_timescale", "30000", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv", "-c:a", "aac", "-b:a", "128k", cls.source),
            "synthetic-source")
        cls.source_hash = digest(cls.source)
        cls.source_metadata = probe(cls.source, cls.fixture_runner)
        if int(stream(cls.source_metadata, "video")["nb_frames"]) != 72:
            raise AssertionError("Synthetic source frame count changed")
        cls.pdf = cls.root / "source/synthetic.pdf"
        cls.pdf.write_bytes(b"Synthetic test-only PDF identity; pre-rendered images are generated below.\n")
        slides = []
        for page, color in ((1, "teal"), (2, "orange"), (3, "gray")):
            path = cls.root / f"slide-{page}.png"
            cls.fixture_runner.run(ffmpeg("-f", "lavfi", "-i", f"color={color}:size=2560x1440",
                                          "-frames:v", "1", "-threads", "1", path), "synthetic-slide")
            slides.append({"page": page, "path": str(path), "sha256": digest(path)})
        cls.plan = {
            "schema_version": 1, "source": {"path": str(cls.source), "sha256": cls.source_hash},
            "pdf": {"path": str(cls.pdf), "sha256": digest(cls.pdf)}, "slides": slides,
            "frame_rate": "30000/1001", "time_base": "1/30000", "total_frames": 72,
            "intervals": [
                {"start_frame": 0, "end_frame": 15, "kind": "pdf", "page": 1, "render_path": slides[0]["path"], "confidence": 1},
                {"start_frame": 15, "end_frame": 29, "kind": "pdf", "page": 1, "render_path": slides[0]["path"], "confidence": 1},
                {"start_frame": 29, "end_frame": 42, "kind": "original", "confidence": 0},
                {"start_frame": 42, "end_frame": 57, "kind": "pdf", "page": 2, "render_path": slides[1]["path"], "confidence": 1},
                {"start_frame": 57, "end_frame": 72, "kind": "pdf", "page": 3, "render_path": slides[2]["path"], "confidence": 1},
            ], "limitations": ["Synthetic fixture, not a mapping of the lecture."],
        }
        cls.overlay = {"schema_version": 1, "source_coordinate_space": [1280, 720], "intervals": [
            {"start_frame": 0, "end_frame": 48, "rectangles": [[1000, 80, 200, 140]], "reason": "synthetic moving inset"},
            {"start_frame": 48, "end_frame": 72, "rectangles": [[1000, 80, 200, 140], [0, 700, 1280, 20]],
             "reason": "synthetic inset and live footer"},
        ], "limitations": ["Synthetic live-content rectangles only."]}
        cls.alignment_path, cls.overlays_path = cls.root / "alignment.json", cls.root / "overlays.json"
        json_write(cls.alignment_path, cls.plan)
        json_write(cls.overlays_path, cls.overlay)
        cls.approval_path = cls.root / "synthetic_approval.json"
        asset = {"path": slides[0]["path"], "sha256": slides[0]["sha256"]}
        json_write(cls.approval_path, {
            "schema_version": 1, "source_sha256": cls.source_hash, "pdf_sha256": digest(cls.pdf),
            "user_full_render_instruction": "Synthetic fixture approval for automated tests only.",
            "requires_full_structural_validation": True, "approved_preview": asset, "approved_comparison": asset,
        })
        (cls.root / "lecture_restore").mkdir()
        original_package = Path(__file__).resolve().parents[1] / "lecture_restore"
        for name in ("core.py", "validation.py"):
            shutil.copyfile(original_package / name, cls.root / "lecture_restore" / name)

    @classmethod
    def _cleanup(cls):
        for logger in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(logger, logging.Logger) and logger.name.startswith(str(cls.root)):
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
        cls.temporary.cleanup()

    def _arguments(self, output, work, resume=False):
        args = ["--alignment", str(self.alignment_path), "--overlays", str(self.overlays_path),
                "--approval", str(self.approval_path), "--output", str(output), "--work", str(work),
                "--workers", "1", "--crf", "16", "--audio-gain-db", "-2.3"]
        return args + (["--resume"] if resume else [])

    def _run(self, output, work, resume=False):
        with mock.patch.object(renderer, "ROOT", self.root), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return renderer.main(self._arguments(output, work, resume))

    def _pixels(self, path, x, y):
        result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
                                 "-map", "0:v:0", "-an", "-vf", f"crop=2:2:{x}:{y},format=rgb24",
                                 "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(len(result.stdout) % 12, 0)
        return [tuple(result.stdout[index:index+3]) for index in range(0, len(result.stdout), 12)]

    def _audio_rms(self, path):
        result = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", "0.3",
                                 "-i", str(path), "-t", "1.5", "-map", "0:a:0", "-vn", "-ac", "1",
                                 "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        samples = [number for (number,) in struct.iter_unpack("<f", result.stdout)]
        return math.sqrt(sum(value*value for value in samples) / len(samples))

    def _assert_restored(self, output, work):
        report = json_read(output.with_suffix(".validation.json"))
        self.assertTrue(report["passed"], json.dumps(report.get("checks"), indent=2))
        self.assertEqual(report["publication_status"], "published")
        self.assertEqual(report["output_sha256"], digest(output))
        self.assertEqual(digest(self.source), self.source_hash)
        self.assertEqual(report["checks"]["exact_full_source_frame_grid"]["actual_frames"], 72)
        self.assertEqual(report["candidate_metadata"]["streams"][0]["has_b_frames"], 0)
        self.assertEqual(report["audio_processing"], {"constant_gain_db": -2.3, "neural_or_denoising": False, "aac_reencoding": True})
        self.assertAlmostEqual(report["checks"]["av_end_offset"]["expected_seconds"],
                               report["checks"]["av_end_offset"]["actual_seconds"], delta=0.05)
        self.assertGreater(report["checks"]["av_end_offset"]["expected_seconds"], 0.08)
        source_pixels = self._pixels(self.source, 1100, 140)
        inset_pixels = self._pixels(output, 2200, 280)
        self.assertEqual(len(source_pixels), 72)
        self.assertEqual(len(inset_pixels), 72)
        for index, (before, after) in enumerate(zip(source_pixels, inset_pixels)):
            self.assertLessEqual(max(abs(a-b) for a, b in zip(before, after)), 8, f"Inset frame identity changed at frame {index}: {before} -> {after}")
        outside = self._pixels(output, 800, 600)
        for index in range(29, 42):
            self.assertLessEqual(max(abs(a-b) for a, b in zip(source_pixels[index], outside[index])), 8,
                                 f"Original fallback content changed at {index}")
        for a, b in ((0, 29), (42, 57), (57, 72)):
            for index in range(a, b):
                self.assertLessEqual(max(abs(x-y) for x, y in zip(outside[index], outside[a])), 3,
                                     "Static slide pixels changed within one PDF state")
        self.assertGreater(max(abs(a-b) for a, b in zip(outside[42], outside[57])), 30,
                           "Distinct final PDF pages failed to switch at the expected frame")
        self.assertGreater(max(abs(a-b) for a, b in zip(outside[28], outside[29])), 30,
                           "Original fallback did not start at its exact planned frame")
        footer = self._pixels(output, 40, 1420)
        for index in range(48, 72):
            self.assertLessEqual(max(abs(a-b) for a, b in zip(source_pixels[index], footer[index])), 8,
                                 f"Live footer frame identity changed at {index}")
        self.assertAlmostEqual(20*math.log10(self._audio_rms(output) / self._audio_rms(self.source)), -2.3, delta=0.15)
        checkpoint = json_read(work / "checkpoint.json")
        self.assertTrue(all(item["source_frame_identity_verified"] for item in checkpoint["completed"].values()))
        self.assertFalse(list(work.glob("*.partial.mp4")))
        self.assertFalse(list(work.glob("segment-*.mp4")))
        json.dumps(report, allow_nan=False)

    def test_full_render_preserves_source_frames_fallback_footer_gain_and_packet_grid(self):
        output, work = self.root / "successful/output.mp4", self.root / "successful/work"
        self.assertEqual(self._run(output, work), 0)
        self._assert_restored(output, work)

    def test_publication_race_checkpoint_guard_and_orphan_recovery(self):
        output, work = self.root / "race/output.mp4", self.root / "race/work"
        competing = b"A concurrent writer owns this destination."
        original_link = renderer.os.link

        def competing_publication(source, destination, *args, **kwargs):
            if Path(destination) == output:
                with output.open("xb") as handle:
                    handle.write(competing)
            return original_link(source, destination, *args, **kwargs)

        with mock.patch.object(renderer.os, "link", side_effect=competing_publication):
            self.assertEqual(self._run(output, work), 1)
        self.assertEqual(output.read_bytes(), competing)
        failed_report = json_read(output.with_suffix(".validation.json"))
        self.assertTrue(failed_report["passed"])
        self.assertEqual(failed_report["publication_status"], "not_published")
        self.assertIn("publication_error", failed_report)
        self.assertFalse(list(work.glob("*.partial.mp4")))
        output.unlink()  # This competing file belongs solely to our synthetic test.
        checkpoint_path = work / "checkpoint.json"
        checkpoint = json_read(checkpoint_path)
        intact = deepcopy(checkpoint)
        sentinel = self.root / "do_not_delete.txt"
        sentinel.write_text("Outside-work checkpoint paths must never authorize deletion.")
        checkpoint["completed"]["0"]["path"] = str(sentinel)
        checkpoint["completed"]["0"]["sha256"] = digest(sentinel)
        json_write(checkpoint_path, checkpoint)
        with mock.patch.object(renderer, "encode_segment", side_effect=AssertionError("Untrusted checkpoint reached encoding")):
            self.assertEqual(self._run(output, work, resume=True), 1)
        self.assertTrue(sentinel.is_file())
        self.assertFalse(output.exists())
        # Simulate a crash after a chunk was linked but before checkpointing.
        orphan = Path(intact["completed"]["0"]["path"])
        self.assertTrue(orphan.is_file())
        del intact["completed"]["0"]
        json_write(checkpoint_path, intact)
        original_encode = renderer.encode_segment
        with mock.patch.object(renderer, "encode_segment", wraps=original_encode) as encode:
            self.assertEqual(self._run(output, work, resume=True), 0)
            self.assertEqual(encode.call_count, 1, "Resume should rebuild only the uncheckpointed chunk")
            self.assertEqual(encode.call_args.args[0], 0)
        self.assertTrue(sentinel.is_file())
        self._assert_restored(output, work)


if __name__ == "__main__":
    unittest.main()
