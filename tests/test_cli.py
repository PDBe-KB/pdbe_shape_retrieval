import os
import unittest
from unittest.mock import patch

from shape_retrieval.cli import (
    default_map2zernike_binary,
    default_obj2grid_binary,
    env_flag,
)


class TestCliEnvironmentDefaults(unittest.TestCase):
    def test_map2zernike_binary_env_has_priority(self):
        with patch.dict(
            os.environ, {"MAP2ZERNIKE_BINARY": "/tools/map2zernike"}, clear=True
        ):
            self.assertEqual(default_map2zernike_binary(), "/tools/map2zernike")

    def test_map2zernike_setup_dir_appends_binary_name(self):
        with patch.dict(
            os.environ, {"MAP2ZERNIKE_SETUP_DIR": "/tools/bin"}, clear=True
        ):
            self.assertEqual(default_map2zernike_binary(), "/tools/bin/map2zernike")

    def test_obj2grid_binary_env_has_priority(self):
        with patch.dict(os.environ, {"OBJ2GRID_BINARY": "/tools/obj2grid"}, clear=True):
            self.assertEqual(default_obj2grid_binary(), "/tools/obj2grid")

    def test_obj2grid_path_accepts_direct_binary_path(self):
        with patch.dict(
            os.environ, {"OBJ2GRID_PATH": "/tools/bin/obj2grid"}, clear=True
        ):
            self.assertEqual(default_obj2grid_binary(), "/tools/bin/obj2grid")

    def test_obj2grid_path_uses_last_path_entry(self):
        value = os.pathsep.join(["/usr/bin", "/tools/bin/obj2grid"])
        with patch.dict(os.environ, {"OBJ2GRID_PATH": value}, clear=True):
            self.assertEqual(default_obj2grid_binary(), "/tools/bin/obj2grid")

    def test_defaults_fall_back_to_command_names(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_map2zernike_binary(), "map2zernike")
            self.assertEqual(default_obj2grid_binary(), "obj2grid")

    def test_env_flag_truthy_values(self):
        for value in ("1", "true", "yes", "on", "TRUE"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"SHAPE_RETRIEVAL_PROFILE": value}, clear=True
                ):
                    self.assertTrue(env_flag("SHAPE_RETRIEVAL_PROFILE"))

    def test_env_flag_false_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_flag("SHAPE_RETRIEVAL_PROFILE"))
