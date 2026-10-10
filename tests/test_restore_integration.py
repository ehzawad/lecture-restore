"""Exercise the real restore path with local, synthetic accepted previews.

All source, preview, manifest, acceptance, log and output files are temporary.
The test-only review fixture does not approve or process the real lecture.
Only project ROOT and the implementation identity are mocked; extraction,
filtering, normalization, encoding, validation and publication run normally.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from lecture_restore import cli
from lecture_restore.core import (
    Runner, audio_stats, common_gain, digest, ffmpeg, json_read, json_write,
    probe, stream, video_filter,
)
from lecture_restore.validation import validate


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not installed")
class RestoreIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lecture-restore-integration-")
        # macOS /var is a symlink to /private/var; use one canonical identity.
        cls.root = Path(cls.temporary.name).resolve()
        cls.runners = []
        cls.addClassCleanup(cls._cleanup_fixture)
        cls.fixture_runner = cls._runner("fixture")
        cls.source = cls.root / "source/synthetic.mp4"
        cls.source.parent.mkdir()
        # A 31 s, anamorphic fractional-rate image. Audio starts later and ends
        # sooner by >50 ms, so preserving only overall duration is insufficient.
        cls.fixture_runner.run(ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30000/1001:duration=31",
            "-itsoffset", "0.2", "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000:duration=30.7",
            "-vf", "setsar=16/15", "-c:v", "libx264", "-preset", "ultrafast",
            "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            cls.source), "synthetic_source")
        cls.source_sha256 = digest(cls.source)
        cls.source_metadata = probe(cls.source, cls.fixture_runner)
        cls.preset = {
            "scale": 2, "deinterlace": "off", "deblock": False,
            "hqdn3d": "0.6:0.4:1.0:0.6", "unsharp": "3:3:0.15:3:3:0",
            "audio_filter": "highpass=f=65,afftdn=nr=6:nf=-50:tn=0:gs=3",
            "encoder_preset": "ultrafast", "crf": 18, "threads": 1,
            "target_lufs": -18.0, "true_peak_ceiling_db": -2.0,
        }
        cls.deinterlace = {"enabled": False, "ambiguous": False, "reason": "synthetic progressive fixture"}
        section = {"name": "synthetic_only", "start": 0.5, "duration": 30}
        preview_directory = cls.root / "preview"
        preview_directory.mkdir()
        original = preview_directory / "original.wav"
        filtered = preview_directory / "filtered.wav"
        matched = preview_directory / "matched.wav"
        preview = preview_directory / "restored.mp4"
        cli.extract_audio(cls.source, original, section["start"], section["duration"], cls.fixture_runner)
        cls.fixture_runner.run(ffmpeg("-i", original, "-af", cls.preset["audio_filter"],
                                      "-c:a", "pcm_s24le", filtered), "fixture_audio_filter")
        _, gains = common_gain({"baseline": audio_stats(filtered, cls.fixture_runner)},
                               cls.preset["target_lufs"], cls.preset["true_peak_ceiling_db"])
        cli.matched_audio(filtered, matched, gains["baseline"], cls.fixture_runner)
        cli.encode(cls.source, matched, preview, section["start"], section["duration"],
                   cls.preset, video_filter(cls.preset, cls.deinterlace), cls.fixture_runner)
        preview_validation = validate(cls.source, preview, cls.fixture_runner,
                                      expected_duration=section["duration"], expected_scale=2,
                                      segment_start=section["start"])
        if not preview_validation["passed"]:
            raise AssertionError(f"Synthetic preview failed: {preview_validation['failed_checks']}")
        report_path = preview_directory / "validation.json"
        json_write(report_path, preview_validation)
        ffmpeg_version = cls.fixture_runner.run(["ffmpeg", "-version"], "version").stdout.splitlines()[0]
        cls.identity = {
            "source": str(cls.source), "source_sha256": cls.source_sha256,
            "implementation_sha256": "synthetic-test-only-implementation",
            "preset_values": {"synthetic_2x": cls.preset}, "segments": [section],
            "neural_requested": False, "ffmpeg_version": ffmpeg_version,
        }
        manifest = {
            "schema_version": 1, "status": "ready_for_review", "identity": cls.identity,
            "completed": [{
                "segment": section,
                "files": {str(path): digest(path) for path in preview_directory.iterdir() if path.is_file()},
                "variants": [{"name": "synthetic_2x", "preset": "synthetic_2x", "audio": "baseline",
                              "video": str(preview), "validation": preview_validation,
                              "deinterlace": cls.deinterlace}],
            }],
        }
        cls.manifest = cls.root / "manifest.json"
        cls.acceptance = cls.root / "acceptance.json"
        json_write(cls.manifest, manifest)
        json_write(cls.acceptance, {
            "schema_version": 1, "manifest": str(cls.manifest), "manifest_sha256": digest(cls.manifest),
            "identity": cls.identity, "preset": "synthetic_2x", "audio": "baseline",
            "human_review_attestation": True, "reviewed": sorted(cli.REVIEW_ITEMS),
            "comparison_note": "Synthetic integration test fixture; never an acceptance of the university lecture.",
        })

    @classmethod
    def _runner(cls, name):
        runner = Runner(cls.root / "logs" / name)
        cls.runners.append(runner)
        return runner

    @classmethod
    def _cleanup_fixture(cls):
        for runner in cls.runners:
            for handler in list(runner.logger.handlers):
                handler.close()
                runner.logger.removeHandler(handler)
        cls.temporary.cleanup()

    def _restore(self, output, runner):
        with mock.patch.object(cli, "ROOT", self.root), \
             mock.patch.object(cli, "code_hash", return_value=self.identity["implementation_sha256"]), \
             redirect_stdout(io.StringIO()):
            cli.restore(SimpleNamespace(acceptance=self.acceptance, output=output), runner)

    def test_full_restore_publishes_valid_mp4_and_preserves_anamorphic_timing(self):
        output = self.root / "successful/lecture_restored.mp4"
        runner = self._runner("successful")
        self._restore(output, runner)
        self.assertTrue(output.is_file())
        report = json_read(output.with_suffix(".validation.json"))
        self.assertTrue(report["passed"], json.dumps(report["checks"], indent=2))
        self.assertEqual(report["source_sha256"], self.source_sha256)
        self.assertEqual(report["output_sha256"], digest(output))
        self.assertEqual(digest(self.source), self.source_sha256)
        result = probe(output, runner)
        before, after = stream(self.source_metadata, "video"), stream(result, "video")
        self.assertEqual((after["width"], after["height"]), (320, 240))
        self.assertEqual(Fraction(after["sample_aspect_ratio"].replace(":", "/")), Fraction(16, 15))
        self.assertEqual(after["display_aspect_ratio"], before["display_aspect_ratio"])
        self.assertEqual(Fraction(after["avg_frame_rate"]), Fraction(30000, 1001))
        self.assertEqual(after["codec_name"], "h264")
        self.assertEqual(stream(result, "audio")["codec_name"], "aac")
        self.assertAlmostEqual(float(result["format"]["duration"]),
                               float(self.source_metadata["format"]["duration"]), delta=0.05)
        for boundary in ("start", "end"):
            timing = report["checks"][f"av_{boundary}_offset"]
            # Ensure this fixture actually exercises nonzero source offsets.
            self.assertGreater(abs(timing["expected_seconds"]), 0.08)
            self.assertEqual(timing["status"], "pass")
            self.assertAlmostEqual(timing["actual_seconds"], timing["expected_seconds"], delta=0.05)
        self.assertLessEqual(report["audio_measurements"]["true_peak_dbtp"], -1.0)
        self.assertEqual(report["audio_measurements"]["clipping"]["samples_at_or_above_full_scale"], 0)
        # Logs establish that the real full audio processing and encode ran.
        labels = {entry["label"] for entry in runner.records}
        self.assertTrue({"extract_audio", "full_audio", "constant_gain"} <= labels)
        self.assertTrue(any(entry["label"].startswith("encode_") for entry in runner.records))
        self.assertFalse(list(output.parent.glob(".restore-*")))
        self.assertFalse(list(output.parent.glob("*.partial.mp4")))
        json.dumps(report, allow_nan=False)

    def test_concurrent_destination_creation_is_never_overwritten(self):
        output = self.root / "publication-race/lecture_restored.mp4"
        runner = self._runner("publication-race")
        competing_content = b"An independently created destination must survive publication."
        original_link = cli.os.link
        attempts = []

        def create_destination_then_link(source, destination, *args, **kwargs):
            attempts.append((Path(source), Path(destination)))
            self.assertFalse(Path(destination).exists())
            with Path(destination).open("xb") as handle:
                handle.write(competing_content)
            # Exercise the actual filesystem exclusive publication operation.
            return original_link(source, destination, *args, **kwargs)

        with mock.patch.object(cli.os, "link", side_effect=create_destination_then_link):
            with self.assertRaises(FileExistsError):
                self._restore(output, runner)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0][1], output)
        self.assertEqual(output.read_bytes(), competing_content)
        self.assertEqual(digest(self.source), self.source_sha256)
        self.assertFalse(list(output.parent.glob(".restore-*")))
        self.assertFalse(list(output.parent.glob("*.partial.mp4")))
        # A report of the successfully decoded candidate remains useful when
        # publication fails, but its digest must not be mistaken for the raced file.
        report = json_read(output.with_suffix(".validation.json"))
        self.assertTrue(report["passed"])
        self.assertNotEqual(report["output_sha256"], digest(output))


if __name__ == "__main__":
    unittest.main()
