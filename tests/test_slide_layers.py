"""Real FFmpeg tests of sparse pointer masks, confined to temporary fixtures."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

from lecture_restore.core import digest, ffmpeg, json_read, json_write
from scripts import restore_slides as renderer
from tests import test_slide_restore as base_tests


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is not installed")
class SparsePointerLayerTests(unittest.TestCase):
    """A red pointer and its sparse alpha move across a contrasting PDF page."""

    @classmethod
    def setUpClass(cls):
        base_tests.MeasuredSlideRestoreTests.setUpClass.__func__(cls)
        cls.pointer_rect = [200, 240, 120, 80]
        width, height = cls.pointer_rect[2:]
        rgba_path, gray_path = cls.root / "pointer.rgba", cls.root / "pointer.gray"
        # Both streams describe exactly the same source-frame index. Mask
        # begins at source frame 5 and will be clipped at three slide boundaries.
        with rgba_path.open("wb") as color, gray_path.open("wb") as alpha:
            for n in range(72):
                rgba = bytearray(width*height*4)
                mask = bytearray(width*height)
                if 5 <= n < 65:
                    x = 4 + 4*((n-5) % 20)
                    for y in range(22, 34):
                        for px in range(x, x+16):
                            p = y*width+px
                            rgba[p*4:p*4+4] = bytes((255, 32, 32, 255))
                            mask[p] = 255
                    alpha.write(mask)
                color.write(rgba)
        cls.mask = cls.root / "pointer-mask.mkv"
        cls.fixture_runner.run(ffmpeg("-f", "rawvideo", "-pixel_format", "gray", "-video_size", "120x80",
                                      "-framerate", "30000/1001", "-i", gray_path, "-an", "-c:v", "ffv1",
                                      "-level", "3", "-pix_fmt", "gray", cls.mask), "lossless-moving-mask")
        cls.source = cls.root / "source/synthetic-pointer.mp4"
        cls.fixture_runner.run(ffmpeg(
            "-f", "lavfi", "-i", "color=0x6a6a6a:size=1280x720:rate=30000/1001:duration=2.4024",
            "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", "120x80", "-framerate", "30000/1001", "-i", rgba_path,
            "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=44100:duration=2.56",
            "-filter_complex", "[0:v]drawbox=x=190:y=246:w=170:h=4:color=white:t=fill,gblur=sigma=1[b];[b][1:v]overlay=200:240:shortest=1[v]",
            "-map", "[v]", "-map", "2:a:0", "-c:v", "libx264", "-preset", "fast", "-threads", "2", "-bf", "2",
            "-pix_fmt", "yuv420p", "-enc_time_base", "1/30000", "-video_track_timescale", "30000",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "128k", cls.source), "moving-pointer-source")
        cls.source_hash = digest(cls.source)
        cls.plan["source"] = {"path": str(cls.source), "sha256": cls.source_hash}
        slides = cls.plan["slides"]
        cls.plan["intervals"] = [
            {"start_frame": 0, "end_frame": 17, "kind": "pdf", "page": 1, "render_path": slides[0]["path"]},
            {"start_frame": 17, "end_frame": 38, "kind": "pdf", "page": 1, "render_path": slides[0]["path"]},
            {"start_frame": 38, "end_frame": 72, "kind": "pdf", "page": 2, "render_path": slides[1]["path"]},
        ]
        layer = {"rect": cls.pointer_rect, "mask_path": str(cls.mask), "mask_sha256": digest(cls.mask),
                 "mask_start_frame": 5, "mask_end_frame": 65}
        cls.overlay = {"schema_version": 2, "intervals": [
            {"start_frame": 0, "end_frame": 5, "camera_rectangles": [], "pointer_layers": []},
            {"start_frame": 5, "end_frame": 65, "camera_rectangles": [], "pointer_layers": [layer]},
            {"start_frame": 65, "end_frame": 72, "camera_rectangles": [], "pointer_layers": []},
        ]}
        json_write(cls.alignment_path, cls.plan)
        json_write(cls.overlays_path, cls.overlay)
        approval = json_read(cls.approval_path)
        approval["source_sha256"] = cls.source_hash
        approval["user_revision_instruction"] = "Synthetic sparse-mask test fixture only; no real lecture approval."
        approval["revision_preview_review"] = {
            "status": "passed", "configuration": {"alignment_sha256": digest(cls.alignment_path),
            "overlays_sha256": digest(cls.overlays_path), "camera_preset": "none", "source_sharpen": 0.0, "crf": 16},
            "artifacts": [approval["approved_preview"]],
        }
        json_write(cls.approval_path, approval)

    @classmethod
    def _cleanup(cls):
        base_tests.MeasuredSlideRestoreTests._cleanup.__func__(cls)

    _arguments = base_tests.MeasuredSlideRestoreTests._arguments
    _run = base_tests.MeasuredSlideRestoreTests._run
    _pixels = base_tests.MeasuredSlideRestoreTests._pixels

    def _rgb_roi(self, path, rectangle, count):
        x, y, width, height = rectangle
        p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
                            "-map", "0:v:0", "-an", "-vf", f"crop={width}:{height}:{x}:{y},format=rgb24",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"], capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr.decode(errors="replace"))
        self.assertEqual(len(p.stdout), width*height*count*3)
        return p.stdout

    def test_sparse_alpha_preserves_moving_pointer_identity_and_pdf_inside_old_broad_roi(self):
        output, work = self.root / "sparse/output.mp4", self.root / "sparse/work"
        self.assertEqual(self._run(output, work), 0)
        report = json_read(output.with_suffix(".validation.json"))
        self.assertTrue(report["passed"], report["failed_checks"])
        self.assertEqual(report["checks"]["exact_full_source_frame_grid"]["actual_frames"], 72)
        self.assertEqual(report["identity"]["pointer_mask_hashes"], {str(self.mask): digest(self.mask)})
        self.assertEqual(digest(self.source), self.source_hash)
        original = self._rgb_roi(self.source, self.pointer_rect, 72)
        final = self._rgb_roi(output, [n*2 for n in self.pointer_rect], 72)
        outside = self._pixels(output, 1200, 1000)
        # The source strip and blurry gray background occupy the whole old
        # rectangle. Its unmasked pixels must retain the contrasting PDF.
        for n in range(72):
            p = n*240*160*3 + (120*240+220)*3
            self.assertLessEqual(max(abs(a-b) for a,b in zip(final[p:p+3], outside[n])), 3,
                                 f"Broad source background leaked through sparse mask at source frame {n}")
        for n in range(5, 65):
            x = 4 + 4*((n-5) % 20)
            for px in (x+2, x+13):
                p = n*120*80*3 + (27*120+px)*3
                q = n*240*160*3 + (54*240+px*2)*3
                before, after = original[p:p+3], final[q:q+3]
                self.assertGreater(before[0]-before[1], 120, "Synthetic pointer source was not red at the expected source index")
                self.assertLessEqual(max(abs(a-b) for a,b in zip(before,after)), 10,
                                     f"Cursor mask clipped or shifted at source frame {n}, x={px}: {list(before)} -> {list(after)}")
        # Three independent chunks trim different starts from one lossless
        # mask; matching every moving edge detects even a one-frame phase shift.
        checkpoint = json_read(work / "checkpoint.json")
        active = [s for s in checkpoint["completed"].values() if s["interval"].get("pointer_layers")]
        self.assertEqual([(s["interval"]["start_frame"],s["interval"]["end_frame"]) for s in sorted(active,key=lambda s:s["index"])],
                         [(5,17),(17,38),(38,65)])

    def test_insufficient_mask_source_range_is_rejected(self):
        for key,value in (("mask_start_frame",6),("mask_end_frame",64)):
            with self.subTest(key=key):
                bad = deepcopy(self.overlay)
                bad["intervals"][1]["pointer_layers"][0][key] = value
                with self.assertRaises(ValueError):
                    renderer.merge_intervals(self.plan,bad)

    def test_changed_mask_hash_is_rejected_before_encoding(self):
        bad = deepcopy(self.overlay)
        bad["intervals"][1]["pointer_layers"][0]["mask_sha256"] = "0"*64
        path = self.root / "bad-mask.json"
        json_write(path,bad)
        output,work = self.root / "bad-hash/output.mp4",self.root / "bad-hash/work"
        arguments = self._arguments(output,work)
        arguments[arguments.index("--overlays")+1] = str(path)
        approval = json_read(self.approval_path)
        approval["revision_preview_review"]["configuration"]["overlays_sha256"] = digest(path)
        approval_path = self.root / "bad-mask-synthetic-approval.json"
        json_write(approval_path,approval)
        arguments[arguments.index("--approval")+1] = str(approval_path)
        with mock.patch.object(renderer,"ROOT",self.root), mock.patch.object(renderer,"encode_segment",side_effect=AssertionError("Changed mask reached encoding")), redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            self.assertEqual(renderer.main(arguments),1)
        self.assertFalse(output.exists())
        self.assertFalse(work.exists())
        failures = [json_read(p) for p in self.root.glob("logs/full-slides-*/failure.json")]
        self.assertTrue(any("Pointer mask changed" in f["error"] for f in failures), failures)


if __name__ == "__main__":
    unittest.main()
