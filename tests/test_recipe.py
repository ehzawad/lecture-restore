"""Recipe safety and actual complete-segment replay, using tiny synthetic media."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from lecture_restore.core import Runner, digest, json_write
from scripts import replay_recipe as replay
from scripts import restore_slides as engine


def fixture(directory):
    for name in ("source.mp4", "slides.pdf", "accepted.mp4", "slide.png", "pointer.mkv"):
        (directory/name).write_bytes(name.encode())
    return {
        "schema_version": 1,
        "profile": {"width": 1280, "height": 720, "frame_rate": "30000/1001", "time_base": "1/30000", "total_frames": 12},
        "source": {"path": "source.mp4", "sha256": digest(directory/"source.mp4")},
        "pdf": {"path": "slides.pdf", "sha256": digest(directory/"slides.pdf")},
        "accepted_final": {"path": "accepted.mp4", "sha256": digest(directory/"accepted.mp4"), "approval": "Synthetic fixture only"},
        "assets": {"slide.png": digest(directory/"slide.png"), "pointer.mkv": digest(directory/"pointer.mkv")},
        "intervals": [{"kind": "pdf", "start_frame": 0, "end_frame": 6, "render_path": "slide.png",
                       "rectangles": [], "camera_rectangles": [], "pointer_layers": []},
                      {"kind": "original", "start_frame": 6, "end_frame": 12, "rectangles": [],
                       "camera_rectangles": [], "pointer_layers": []}],
        "settings": {"crf": 16, "audio_gain_db": -2.3, "camera_preset": "mild_sharp", "source_sharpen": .25},
        "ffmpeg_version": "placeholder", "segment_sha256": {"0": "0"*64, "1": "1"*64}}


class RecipeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.recipe = fixture(self.directory)
        self.path = self.directory/"recipe.json"

    def load(self, recipe=None):
        json_write(self.path, recipe or self.recipe)
        return replay.load_recipe(self.path)

    def test_relative_assets_resolve_independent_of_repository(self):
        loaded = self.load()
        self.assertEqual(loaded["source"], (self.directory/"source.mp4").resolve())
        self.assertEqual(loaded["intervals"][0]["render_path"], str((self.directory/"slide.png").resolve()))

    def test_changed_source_asset_and_accepted_final_are_rejected(self):
        for filename in ("source.mp4", "slide.png", "accepted.mp4"):
            with self.subTest(filename=filename):
                path = self.directory/filename
                saved = path.read_bytes()
                path.write_bytes(saved+b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.load()
                path.write_bytes(saved)

    def test_gap_overlap_and_uncovered_source_end_are_rejected(self):
        for field, value in (("start_frame", 5), ("start_frame", 7), ("end_frame", 11)):
            bad = deepcopy(self.recipe)
            bad["intervals"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(bad)

    def test_mask_geometry_coverage_hash_and_layer_consistency_are_rejected(self):
        good = deepcopy(self.recipe)
        layer = {"rect": [100, 100, 20, 20], "mask_path": "pointer.mkv",
                 "mask_sha256": good["assets"]["pointer.mkv"], "mask_start_frame": 0, "mask_end_frame": 6}
        good["intervals"][0].update({"rectangles": [layer["rect"]], "pointer_layers": [layer]})
        self.load(good)
        for field, value in (("mask_start_frame", 1), ("mask_end_frame", 5),
                             ("mask_sha256", "f"*64), ("rect", [101, 100, 20, 20])):
            bad = deepcopy(good)
            bad["intervals"][0]["pointer_layers"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(bad)
        bad = deepcopy(good)
        bad["intervals"][0]["rectangles"] = []
        with self.assertRaisesRegex(ValueError, "Layer rectangles"):
            self.load(bad)

    def test_existing_output_and_report_are_never_replaced(self):
        self.load()
        output = self.directory/"keep.mp4"
        output.write_bytes(b"keep")
        report = self.directory/"report.json"
        with mock.patch.object(replay, "load_recipe") as loader, redirect_stderr(io.StringIO()):
            status = replay.main(["--recipe", str(self.path), "--full", "--output", str(output), "--report", str(report)])
        self.assertEqual(status, 1)
        loader.assert_not_called()
        self.assertEqual(output.read_bytes(), b"keep")
        saved_report = report.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(replay.main(["--recipe", str(self.path), "--report", str(report)]), 1)
        self.assertEqual(report.read_bytes(), saved_report)

    def test_unaccepted_final_or_absolute_asset_path_is_rejected(self):
        bad = deepcopy(self.recipe)
        bad["accepted_final"]["approval"] = ""
        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.load(bad)
        bad = deepcopy(self.recipe)
        bad["source"]["path"] = str(self.directory/"source.mp4")
        with self.assertRaisesRegex(ValueError, "relative"):
            self.load(bad)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RecipeRealMediaTests(unittest.TestCase):
    def test_complete_replay_matches_and_full_recovery_skips_completed_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = fixture(root)
            source, accepted, slide = root/"source.mp4", root/"accepted.mp4", root/"slide.png"
            source.unlink(); accepted.unlink(); slide.unlink()
            runner = Runner(root/"setup-logs")
            runner.run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-f", "lavfi", "-i",
                        "testsrc2=size=1280x720:rate=30000/1001", "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=48000",
                        "-t", "0.4004", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-bf", "0",
                        "-video_track_timescale", "30000", "-c:a", "aac", "-b:a", "192k", str(source)], "source")
            runner.run(["ffmpeg", "-hide_banner", "-nostdin", "-n", "-f", "lavfi", "-i",
                        "color=white:size=2560x1440", "-frames:v", "1", "-threads", "1", str(slide)], "slide")
            raw["source"]["sha256"] = digest(source)
            raw["assets"]["slide.png"] = digest(slide)
            raw["intervals"][0].update({"rectangles": [[1000, 0, 280, 180]], "camera_rectangles": [[1000, 0, 280, 180]]})
            raw["ffmpeg_version"] = runner.run(["ffmpeg", "-version"], "version").stdout.splitlines()[0]
            chunks = root/"original-chunks"; chunks.mkdir()
            for i, interval in enumerate(raw["intervals"]):
                interval = deepcopy(interval)
                interval.update({"camera_preset": "mild_sharp", "source_sharpen": .25})
                if interval.get("render_path"):
                    interval["render_path"] = str(slide)
                item = engine.encode_segment(i, interval, source, chunks, root/"original-logs", 16)
                raw["segment_sha256"][str(i)] = item["sha256"]
            concat = chunks/"list.ffconcat"
            concat.write_text("ffconcat version 1.0\nfile 'segment-0000.mp4'\nduration 0.2002\nfile 'segment-0001.mp4'\nduration 0.2002\n")
            runner.run(engine.final_mux_command(concat, source, accepted, -2.3), "accepted-final")
            raw["accepted_final"]["sha256"] = digest(accepted)
            recipe_path, report = root/"recipe.json", root/"reproduction.json"
            json_write(recipe_path, raw)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = replay.main(["--recipe", str(recipe_path), "--segments", "0", "1", "--report", str(report), "--verify-audio"])
            data = json.loads(report.read_text())
            self.assertEqual(result, 0, data)
            self.assertTrue(data["passed"])
            self.assertEqual(data["verified_frames"], 12)
            self.assertTrue(all(p["encoded_bytes_match"] and p["decoded_frames_match"] for p in data["segments"]))
            self.assertTrue(data["audio"]["passed"])
            self.assertEqual(digest(accepted), raw["accepted_final"]["sha256"])
            self.assertEqual(list(root.glob("replay-*")), [])

            output, work = root/"replayed.mp4", root/"checkpoint-work"
            failure_report, resumed_report = root/"failure.json", root/"resumed.json"
            encode = engine.encode_segment
            def fail_second(index, *args, **kwargs):
                if index == 1:
                    raise RuntimeError("Injected failure after the first checkpoint")
                return encode(index, *args, **kwargs)
            common = ["--recipe", str(recipe_path), "--full", "--output", str(output), "--work", str(work)]
            with mock.patch.object(engine, "encode_segment", side_effect=fail_second), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(replay.main(common + ["--report", str(failure_report)]), 1)
            self.assertFalse(output.exists())
            checkpoint = json.loads((work/"checkpoint.json").read_text())
            self.assertEqual(set(checkpoint["completed"]), {"0"})
            self.assertEqual(digest(work/"segment-0000.mp4"), raw["segment_sha256"]["0"])
            self.assertEqual(digest(accepted), raw["accepted_final"]["sha256"])
            with mock.patch.object(engine, "encode_segment", wraps=encode) as resumed_encoder, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = replay.main(common + ["--report", str(resumed_report), "--resume"])
            resumed = json.loads(resumed_report.read_text())
            self.assertEqual(status, 0, resumed)
            self.assertEqual([call.args[0] for call in resumed_encoder.call_args_list], [1])
            self.assertTrue(resumed["passed"])
            validation = resumed["full_validation"]
            self.assertEqual(validation["checks"]["exact_full_source_frame_grid"]["status"], "pass")
            self.assertEqual(validation["candidate_path"], str(output))
            self.assertEqual(validation["publication_status"], "published")
            self.assertTrue(output.is_file())
            self.assertEqual(list(work.glob("*.mp4")), [])
            self.assertEqual(digest(accepted), raw["accepted_final"]["sha256"])


if __name__ == "__main__":
    unittest.main()
