import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shape_utils.pipeline import PipelineConfig, resolve_executable, run_pipeline, validate_config


class TestPipelineValidation(unittest.TestCase):
    def test_validate_config_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh1 = Path(tmpdir) / "a.obj"
            mesh2 = Path(tmpdir) / "b.off"
            output = Path(tmpdir) / "out"
            mesh1.write_text("v 0 0 0\n")
            mesh2.write_text("OFF\n")

            validate_config(PipelineConfig(mesh1=mesh1, mesh2=mesh2, entry_ids=("a", "b"), output=output))

            self.assertTrue(output.is_dir())

    def test_validate_config_rejects_missing_mesh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh1 = Path(tmpdir) / "missing.obj"
            mesh2 = Path(tmpdir) / "b.obj"
            mesh2.write_text("v 0 0 0\n")

            with self.assertRaises(FileNotFoundError):
                validate_config(PipelineConfig(mesh1=mesh1, mesh2=mesh2, entry_ids=("a", "b"), output=Path(tmpdir)))

    def test_validate_config_rejects_invalid_descriptor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh1 = Path(tmpdir) / "a.obj"
            mesh2 = Path(tmpdir) / "b.obj"
            mesh1.write_text("v 0 0 0\n")
            mesh2.write_text("v 0 0 0\n")

            with self.assertRaises(ValueError):
                validate_config(
                    PipelineConfig(mesh1=mesh1, mesh2=mesh2, entry_ids=("a", "b"), output=Path(tmpdir), descriptor="BAD")
                )

    def test_validate_config_rejects_fix_mesh_dependent_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh1 = Path(tmpdir) / "a.obj"
            mesh2 = Path(tmpdir) / "b.obj"
            mesh1.write_text("v 0 0 0\n")
            mesh2.write_text("v 0 0 0\n")

            with self.assertRaises(ValueError):
                validate_config(
                    PipelineConfig(
                        mesh1=mesh1,
                        mesh2=mesh2,
                        entry_ids=("a", "b"),
                        output=Path(tmpdir),
                        collapse_vertices=True,
                    )
                )


class TestResolveExecutable(unittest.TestCase):
    def test_resolve_executable_accepts_executable_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = Path(tmpdir) / "tool"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)

            self.assertEqual(resolve_executable(str(binary), "tool"), str(binary))

    def test_resolve_executable_uses_path_lookup(self):
        with patch("shape_utils.pipeline.shutil.which", return_value="/usr/local/bin/tool"):
            self.assertEqual(resolve_executable("tool", "tool"), "/usr/local/bin/tool")

    def test_resolve_executable_raises_when_missing(self):
        with patch("shape_utils.pipeline.shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                resolve_executable("missing-tool", "missing-tool")


class TestRunPipelineDispatch(unittest.TestCase):
    def _config(self, tmpdir, *, descriptor="WKS", **overrides):
        tmp_path = Path(tmpdir)
        mesh1 = tmp_path / "a.obj"
        mesh2 = tmp_path / "b.obj"
        mesh1.write_text("v 0 0 0\n")
        mesh2.write_text("v 0 0 0\n")
        values = {
            "mesh1": mesh1,
            "mesh2": mesh2,
            "entry_ids": ("a", "b"),
            "output": tmp_path / "out",
            "descriptor": descriptor,
        }
        values.update(overrides)
        return PipelineConfig(**values)

    def test_run_pipeline_dispatches_spectral_descriptor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, descriptor="WKS")

            with patch("shape_utils.pipeline._run_spectral_descriptors") as mock_spectral:
                run_pipeline(config)

        mock_spectral.assert_called_once_with(config, "WKS", None)

    def test_run_pipeline_dispatches_zernike_descriptor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, descriptor="3DZD")

            with patch("shape_utils.pipeline._run_zernike_descriptors") as mock_zernike:
                run_pipeline(config)

        mock_zernike.assert_called_once_with(config, None)

    def test_run_pipeline_returns_early_when_shape_retrieval_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, no_shape_retrieval=True)

            with patch("shape_utils.pipeline._run_spectral_descriptors") as mock_spectral:
                run_pipeline(config)

        mock_spectral.assert_not_called()

    def test_run_pipeline_calculates_min_distance_before_early_return(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, min_dist_mesh=True, no_shape_retrieval=True)

            with patch("shape_utils.pipeline._calculate_minimum_distance") as mock_min_distance:
                run_pipeline(config)

        mock_min_distance.assert_called_once_with(config, None)

    def test_run_pipeline_passes_fixed_meshes_to_spectral_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, fix_meshes=True)
            fixed_meshes = object()

            with patch("shape_utils.pipeline._fix_meshes", return_value=fixed_meshes) as mock_fix_meshes:
                with patch("shape_utils.pipeline._run_spectral_descriptors") as mock_spectral:
                    run_pipeline(config)

        mock_fix_meshes.assert_called_once_with(config)
        mock_spectral.assert_called_once_with(config, "WKS", fixed_meshes)
