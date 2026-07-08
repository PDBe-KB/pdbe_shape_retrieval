# Copyright (C) 2021 Tunde Aderinwale, Daisuke Kihara, and Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from shape_retrieval.meshes import remove_until_vertex

logger = logging.getLogger(__name__)


def get_inv(
    obj_file: str | Path,
    fileid: str,
    map3dz_binary: str | Path,
    obj2grid_binary: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Generate a Zernike moments .inv file from a 3D OBJ mesh file.

    The OBJ is copied into the output directory before header cleanup and binary
    execution, so the original input mesh is not modified.
    """
    source_obj = Path(obj_file).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_obj.is_file():
        raise FileNotFoundError(f"OBJ file not found: {source_obj}")

    obj_output = _temporary_obj_path(source_obj, output_path, fileid)
    grid_file = Path(f"{obj_output}.grid")
    inv_file = Path(f"{grid_file}.inv")
    final_inv = output_path / f"{fileid}.inv"

    shutil.copy2(source_obj, obj_output)
    remove_until_vertex(str(obj_output))

    try:
        run_binary([str(obj2grid_binary), "-g", "64", str(obj_output)], "obj2grid")

        run_binary([str(map3dz_binary), str(grid_file), "-c", "0.5"], "map2zernike")

        inv_file.replace(final_inv)
        return final_inv
    finally:
        _remove_if_exists(obj_output)
        _remove_if_exists(grid_file)


def run_binary(
    command: list[str], binary_name: str
) -> subprocess.CompletedProcess[str]:
    logger.info("Running %s: %s", binary_name, " ".join(command))
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        if stderr:
            logger.error("%s failed with stderr: %s", binary_name, stderr)
        if stdout:
            logger.debug("%s stdout before failure: %s", binary_name, stdout)
        raise RuntimeError(
            f"{binary_name} failed with exit code {exc.returncode}. stderr: {stderr or '<empty>'}"
        ) from exc
    except OSError as exc:
        logger.error("Could not execute %s: %s", binary_name, exc)
        raise RuntimeError(f"Could not execute {binary_name}: {exc}") from exc


def _temporary_obj_path(source_obj: Path, output_dir: Path, fileid: str) -> Path:
    candidate = output_dir / f"{fileid}.obj"
    if candidate.resolve(strict=False) == source_obj.resolve(strict=False):
        return output_dir / f"{fileid}_zernike_input.obj"
    return candidate


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
