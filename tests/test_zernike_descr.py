import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from shape_retrieval.zernike_descr import get_inv, run_binary


class Test3DZernike(unittest.TestCase):
    def test_get_inv_runs_binaries_and_preserves_source_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_obj = tmp_path / "source.obj"
            source_obj.write_text("# header\nv 0 0 0\nf 1 1 1\n")

            def fake_run(command, check, capture_output, text):
                if command[0] == "/bin/map2zernike":
                    Path(f"{command[1]}.inv").write_text("0\n1\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch(
                "shape_retrieval.zernike_descr.subprocess.run", side_effect=fake_run
            ) as mock_run:
                inv_file = get_inv(
                    source_obj, "entry", "/bin/map2zernike", "/bin/obj2grid", tmp_path
                )

            self.assertEqual(inv_file, tmp_path / "entry.inv")
            self.assertEqual(inv_file.read_text(), "0\n1\n")
            self.assertEqual(source_obj.read_text(), "# header\nv 0 0 0\nf 1 1 1\n")
            self.assertFalse((tmp_path / "entry.obj").exists())
            self.assertFalse((tmp_path / "entry.obj.grid").exists())
            mock_run.assert_has_calls(
                [
                    call(
                        ["/bin/obj2grid", "-g", "64", str(tmp_path / "entry.obj")],
                        check=True,
                        capture_output=True,
                        text=True,
                    ),
                    call(
                        [
                            "/bin/map2zernike",
                            str(tmp_path / "entry.obj.grid"),
                            "-c",
                            "0.5",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    ),
                ]
            )

    def test_get_inv_uses_temporary_name_when_source_matches_output_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_obj = tmp_path / "entry.obj"
            source_obj.write_text("# header\nv 0 0 0\n")

            def fake_run(command, check, capture_output, text):
                if command[0] == "/bin/map2zernike":
                    Path(f"{command[1]}.inv").write_text("0\n1\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch(
                "shape_retrieval.zernike_descr.subprocess.run", side_effect=fake_run
            ) as mock_run:
                get_inv(
                    source_obj, "entry", "/bin/map2zernike", "/bin/obj2grid", tmp_path
                )

            temporary_obj = tmp_path / "entry_zernike_input.obj"
            mock_run.assert_any_call(
                ["/bin/obj2grid", "-g", "64", str(temporary_obj)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(temporary_obj.exists())
            self.assertEqual(source_obj.read_text(), "# header\nv 0 0 0\n")

    def test_run_binary_includes_stderr_on_failure(self):
        error = subprocess.CalledProcessError(
            returncode=2,
            cmd=["/bin/tool"],
            output="partial stdout",
            stderr="specific failure",
        )

        with patch("shape_retrieval.zernike_descr.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "specific failure"):
                run_binary(["/bin/tool"], "tool")

    def test_run_binary_wraps_os_error(self):
        with patch(
            "shape_retrieval.zernike_descr.subprocess.run",
            side_effect=OSError("missing executable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing executable"):
                run_binary(["/bin/tool"], "tool")
