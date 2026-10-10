"""A separate data workspace must not change code identity or code defaults."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lecture_restore import cli
from scripts import setup_neural


class WorkspaceTests(unittest.TestCase):
    def test_explicit_workspace_keeps_code_and_runtime_paths_separate(self):
        original_root = cli.ROOT
        original_hash = cli.code_hash()
        with tempfile.TemporaryDirectory(prefix="lecture-workspace-") as directory:
            workspace = (Path(directory) / "new-job").resolve()
            segments = Path(directory) / "chosen-sections.json"

            def check(args, runner):
                self.assertEqual(cli.ROOT, workspace.resolve())
                self.assertEqual(args.input, workspace / "source/lecture.mp4")
                self.assertEqual(args.config, cli.CODE_ROOT / "lecture_restore/presets.json")
                self.assertIn("conservative_2x", cli.presets(args.config))
                self.assertEqual(args.segments, segments)
                self.assertTrue(runner.log_dir.is_relative_to(workspace))
                self.assertEqual(cli.code_hash(), original_hash)

            with mock.patch.object(cli, "previews", side_effect=check):
                self.assertEqual(cli.main(["--workspace", str(workspace), "previews",
                                           "--segments", str(segments)]), 0)
            self.assertTrue((workspace / "logs").is_dir())
        self.assertEqual(cli.ROOT, original_root)

    def test_workspace_defaults_to_current_directory(self):
        with tempfile.TemporaryDirectory(prefix="lecture-cwd-") as directory:
            workspace = Path(directory).resolve()

            def check(args, runner):
                self.assertEqual(cli.ROOT, workspace)
                self.assertEqual(args.output, workspace / "output/lecture_restored.mp4")
                self.assertEqual(args.acceptance, workspace / "config/acceptance.json")

            with mock.patch.object(Path, "cwd", return_value=workspace), \
                    mock.patch.object(cli, "restore", side_effect=check):
                self.assertEqual(cli.main(["restore"]), 0)

    def test_preview_sections_cannot_fall_back_to_another_course(self):
        with mock.patch.object(cli, "Runner", side_effect=AssertionError("Processing started")), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as context:
            cli.main(["previews"])
        self.assertEqual(context.exception.code, 2)

    def test_missing_optional_neural_environment_stays_inside_workspace(self):
        with tempfile.TemporaryDirectory(prefix="lecture-neural-check-") as directory:
            workspace = Path(directory) / "new-job"
            with mock.patch.object(setup_neural.urllib.request, "urlopen",
                                   side_effect=AssertionError("Unexpected download")), \
                    redirect_stdout(io.StringIO()):
                result = setup_neural.main(["--project-root", str(workspace), "--check-only"])
            self.assertEqual(result, 1)
            state = cli.json_read(workspace / "reports/neural_setup.json")
            self.assertFalse(state["ready"])
            self.assertFalse(state["media_uploaded"])
            self.assertTrue((workspace / "logs/neural_setup.log").exists())


if __name__ == "__main__":
    unittest.main()
