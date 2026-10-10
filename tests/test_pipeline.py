"""Safety gates and measured FFmpeg regressions for the restoration workflow.

The approval tests use a runner that refuses processing. The optional media
test encodes a tiny anamorphic, fractional-rate fixture and measures the actual
output; it does not merely compare filter strings.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from lecture_restore import cli
from lecture_restore.core import common_gain, digest, video_filter
from lecture_restore.validation import _cadence_comparison


class GainTests(unittest.TestCase):
    def test_every_candidate_shares_an_attainable_target_without_peak_overload(self):
        # The original has the highest transient/average ratio. Matching to the
        # requested -18 LUFS would overload it, so every candidate must be quieter.
        measurements = {
            "original": {"input_i": -24.0, "input_tp": -0.5},
            "conservative": {"input_i": -27.0, "input_tp": -5.0},
            "neural": {"input_i": -29.0, "input_tp": -9.0},
        }
        untouched = deepcopy(measurements)
        target, gains = common_gain(measurements, target=-18.0, ceiling=-2.0)
        self.assertLess(target, -18)
        for name, measurement in measurements.items():
            self.assertAlmostEqual(measurement["input_i"] + gains[name], target)
            self.assertLessEqual(measurement["input_tp"] + gains[name], -2.0 + 1e-10)
        self.assertEqual(measurements, untouched)

    def test_ample_headroom_keeps_requested_loudness(self):
        target, gains = common_gain({"quiet": {"input_i": -32, "input_tp": -22}},
                                    target=-18, ceiling=-2)
        self.assertEqual(target, -18)
        self.assertEqual(gains["quiet"], 14)


class RefuseProcessingRunner:
    """Versions may be inspected; no processing command may cross the gate."""
    def __init__(self):
        self.commands = []

    def run(self, args, label, timeout=None):
        self.commands.append(list(map(str, args)))
        if list(map(str, args)) == ["ffmpeg", "-version"]:
            return subprocess.CompletedProcess(args, 0, "ffmpeg version test-fixture\n", "")
        raise AssertionError(f"Processing started before approval refusal: {label}")


class ApprovalGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lecture-gates-")
        self.root = Path(self.directory.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"immutable source fixture")
        self.source_digest = digest(self.source)
        self.preview = self.root / "preview.mp4"
        self.preview.write_bytes(b"reviewed preview fixture")
        self.manifest_path = self.root / "manifest.json"
        self.approval_path = self.root / "acceptance.json"
        self.output = self.root / "output.mp4"
        self.preset = {"scale": 1, "audio_filter": "anull", "target_lufs": -18,
                       "true_peak_ceiling_db": -2, "encoder_preset": "medium",
                       "crf": 18, "threads": 1}
        segment = {"name": "representative", "start": 10, "duration": 30}
        self.identity = {"source": str(self.source), "source_sha256": self.source_digest,
                         "implementation_sha256": "reviewed-code", "segments": [segment],
                         "preset_values": {"conservative": self.preset},
                         "ffmpeg_version": "ffmpeg version test-fixture", "neural_requested": False}
        self.manifest = {"status": "ready_for_review", "identity": self.identity,
                         "completed": [{"segment": segment,
                                        "files": {str(self.preview): digest(self.preview)},
                                        "variants": [{"preset": "conservative", "audio": "baseline",
                                                      "video": str(self.preview),
                                                      "validation": {"passed": True},
                                                      "deinterlace": {"enabled": False, "ambiguous": False}}]}]}
        self.approval = {"manifest": str(self.manifest_path), "identity": self.identity,
                         "preset": "conservative", "audio": "baseline",
                         "human_review_attestation": True, "reviewed": sorted(cli.REVIEW_ITEMS),
                         "comparison_note": "All representative content reviewed"}
        self._save()

    def tearDown(self):
        self.directory.cleanup()

    def _save(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.approval["manifest_sha256"] = digest(self.manifest_path)
        self.approval_path.write_text(json.dumps(self.approval))

    def _refuses(self, errors=RuntimeError, implementation="reviewed-code"):
        runner = RefuseProcessingRunner()
        args = SimpleNamespace(acceptance=self.approval_path, output=self.output)
        # Make all non-gate prerequisites available. Any attempt to extract audio
        # or encode is a test failure, even if later validation would reject it.
        metadata = {"format": {"duration": "90", "start_time": "0"},
                    "streams": [{"codec_type": "video", "width": 160, "height": 120},
                                {"codec_type": "audio", "channels": 2, "start_time": "0"}]}
        with mock.patch.object(cli, "require_tools"), \
             mock.patch.object(cli, "code_hash", return_value=implementation), \
             mock.patch.object(cli, "probe", return_value=metadata), \
             mock.patch.object(cli, "disk_guard"), \
             mock.patch.object(cli, "extract_audio", side_effect=AssertionError("Audio extraction crossed approval gate")) as extract, \
             mock.patch.object(cli, "encode", side_effect=AssertionError("Encoding crossed approval gate")) as encode:
            with self.assertRaises(errors):
                cli.restore(args, runner)
            extract.assert_not_called()
            encode.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(digest(self.source), self.source_digest)

    def test_missing_acceptance_never_starts_processing(self):
        self.approval_path.unlink()
        self._refuses(errors=FileNotFoundError)

    def test_unaccepted_preview_never_starts_processing(self):
        self.approval["human_review_attestation"] = False
        self._save()
        self._refuses()

    def test_incomplete_review_attestation_never_starts_processing(self):
        self.approval["reviewed"].remove("symbols")
        self._save()
        self._refuses()

    def test_changed_manifest_never_starts_processing(self):
        self.manifest_path.write_text(self.manifest_path.read_text() + " ")
        self._refuses()

    def test_changed_preview_never_starts_processing(self):
        self.preview.write_bytes(b"different, unreviewed preview")
        self._refuses()

    def test_changed_source_never_starts_processing(self):
        self.source.write_bytes(b"different input")
        self.source_digest = digest(self.source)  # Refusal must leave this new source intact.
        self._refuses()

    def test_changed_implementation_never_starts_processing(self):
        self._refuses(implementation="unreviewed-code")

    def test_selected_failed_preview_is_refused_even_with_self_consistent_acceptance(self):
        self.manifest["completed"][0]["variants"][0]["validation"]["passed"] = False
        self._save()
        self._refuses()

    def test_selected_missing_preview_is_refused_even_with_self_consistent_acceptance(self):
        self.manifest["completed"][0]["variants"][0]["audio"] = "neural"
        self._save()
        self._refuses()

    def test_incomplete_representative_sections_are_refused_before_processing(self):
        self.manifest["identity"]["segments"].append({"name": "missing", "start": 50, "duration": 30})
        self._save()
        self._refuses()

    def test_manifest_not_ready_for_review_is_refused_before_processing(self):
        self.manifest["status"] = "in_progress"
        self._save()
        self._refuses()

    def test_ambiguous_interlace_is_refused_before_processing(self):
        self.manifest["completed"][0]["variants"][0]["deinterlace"]["ambiguous"] = True
        self._save()
        self._refuses()

    def test_neural_package_model_and_python_changes_are_refused_before_processing(self):
        signature = {"packages": {"torch": "2.2.2", "torchaudio": "2.2.2",
                                   "numpy": "1.26.4", "deepfilternet": "0.5.6",
                                   "deepfilterlib": "0.5.6", "soundfile": "0.12.1"},
                     "model_files": {"checkpoints/model.ckpt": "reviewed-model"},
                     "python": "3.11.16", "machine": "arm64"}
        self.manifest["identity"]["neural_signature"] = signature
        self.manifest["identity"]["neural_requested"] = True
        item = self.manifest["completed"][0]
        item["variants"][0]["audio"] = "neural"
        item["neural"] = {"packages": deepcopy(signature["packages"])}
        self.approval["audio"] = "neural"
        self._save()
        for field in ("packages", "model_files", "python", "machine"):
            with self.subTest(changed=field):
                current = deepcopy(signature)
                if field == "packages":
                    current[field]["torch"] = "2.3.0"
                elif field == "model_files":
                    current[field]["checkpoints/model.ckpt"] = "unreviewed-model"
                else:
                    current[field] = "unreviewed-environment"
                with mock.patch.object(cli, "neural_signature", return_value=current):
                    self._refuses()


class PreviewPublicationTests(unittest.TestCase):
    def test_publish_relocates_paths_and_hashes_json_and_removes_staging(self):
        with tempfile.TemporaryDirectory(prefix="lecture-publication-") as directory:
            run = Path(directory)
            segment = {"name": "speech", "start": 60, "duration": 30}
            destination = run / "speech"
            def build(args, stage, section, selected, metadata, runner):
                stage.mkdir(parents=True)
                video, report = stage / "preview.mp4", stage / "validation.json"
                video.write_bytes(b"validated preview")
                report.write_text(json.dumps({"candidate_path": str(video), "passed": True}))
                return {"segment": segment, "original_audio": str(stage / "original.wav"),
                        "variants": [{"video": str(video), "validation": {"candidate_path": str(video)}}],
                        "files": {str(video): digest(video), str(report): digest(report)}}
            with mock.patch.object(cli, "_build_segment", side_effect=build):
                item = cli.preview_segment(None, run, segment, {}, {}, None)
            self.assertEqual(item["variants"][0]["video"], str(destination / "preview.mp4"))
            self.assertEqual(json.loads((destination / "validation.json").read_text())["candidate_path"],
                             str(destination / "preview.mp4"))
            self.assertTrue(cli.checked_files(item))
            self.assertFalse(list(run.glob(".work_*")))
            self.assertNotIn(".work_", json.dumps(item))

    def test_build_failure_preserves_previous_segment_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory(prefix="lecture-recovery-") as directory:
            run = Path(directory)
            segment = {"name": "speech", "start": 60, "duration": 30}
            destination = run / "speech"
            destination.mkdir()
            old = destination / "neural_raw.wav"
            old.write_bytes(b"previous checkpoint")
            old_sha = digest(old)
            def failing_build(args, stage, section, selected, metadata, runner):
                self.assertFalse((stage / "neural_raw.wav").exists())
                stage.mkdir(parents=True)
                (stage / "neural_raw.wav").write_bytes(b"incomplete new processing")
                raise RuntimeError("Simulated interrupted neural inference")
            with mock.patch.object(cli, "_build_segment", side_effect=failing_build):
                with self.assertRaisesRegex(RuntimeError, "Simulated interrupted"):
                    cli.preview_segment(None, run, segment, {}, {}, None)
            self.assertEqual(digest(old), old_sha)
            self.assertFalse(list(run.glob(".work_*")))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not installed")
class MeasuredVideoTests(unittest.TestCase):
    def test_two_times_scale_preserves_anamorphic_aspect_fractional_rate_and_source(self):
        with tempfile.TemporaryDirectory(prefix="lecture-video-test-") as directory:
            root = Path(directory)
            source, candidate = root / "source.mp4", root / "candidate.mp4"
            def run(command):
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                return result
            run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                 "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30000/1001",
                 "-vf", "setsar=16/15", "-frames:v", "15", "-c:v", "libx264",
                 "-threads", "1", "-pix_fmt", "yuv420p", str(source)])
            source_hash = digest(source)
            vf = video_filter({"scale": 2, "hqdn3d": "0.6:0.4:1.0:0.6",
                               "unsharp": "3:3:0.15:3:3:0", "deblock": False},
                              {"enabled": True})
            run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
                 "-vf", vf, "-c:v", "libx264", "-threads", "1", "-fps_mode", "passthrough",
                 str(candidate)])
            def probe(path):
                return json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                       "-show_streams", "-show_packets", "-of", "json", str(path)]).stdout)
            before, after = probe(source), probe(candidate)
            original, restored = before["streams"][0], after["streams"][0]
            self.assertEqual((restored["width"], restored["height"]), (320, 240))
            self.assertEqual(Fraction(restored["sample_aspect_ratio"].replace(":", "/")), Fraction(16, 15))
            self.assertEqual(Fraction(restored["avg_frame_rate"]), Fraction(30000, 1001))
            self.assertEqual(restored["display_aspect_ratio"], original["display_aspect_ratio"])
            cadence = _cadence_comparison(sorted(float(p["pts_time"]) for p in before["packets"]),
                                          [float(p["pts_time"]) for p in after["packets"]],
                                          0, 15 * 1001 / 30000)
            self.assertEqual(cadence["status"], "pass")
            self.assertEqual(digest(source), source_hash)


if __name__ == "__main__":
    unittest.main()
