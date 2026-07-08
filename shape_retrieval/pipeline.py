from __future__ import annotations

import csv
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pyFM import mesh

from shape_retrieval.config import FunctionalMapConfig
from shape_retrieval.functional_maps import (
    calculate_functional_maps,
    calculate_p2p_map,
    set_FM_model_parameters,
)
from shape_retrieval.meshes import fix_mesh
from shape_retrieval.similarity_scores import calculate_geodesic_norm_score
from shape_retrieval.utils import (
    find_minimum_distance_meshes,
    save_data_to_csv,
    save_list_to_csv,
)
from shape_retrieval.zernike_descr import get_inv

logger = logging.getLogger(__name__)

SUPPORTED_DESCRIPTORS = {"WKS", "HKS", "3DZD"}
SUPPORTED_REFINEMENT = {"icp", "zoomout"}


@dataclass(frozen=True)
class PipelineConfig:
    mesh1: Path
    mesh2: Path
    entry_ids: tuple[str, str]
    output: Path
    descriptor: str = "WKS"
    fix_meshes: bool = False
    collapse_vertices: bool = False
    resolution: float = 0.5
    reconstruct_mesh: bool = False
    map2zernike_binary: str = "map2zernike"
    obj2grid_binary: str = "obj2grid"
    neigvecs: int = 200
    n_ev: int = 50
    n_descr: int = 100
    step: int = 1
    landmarks: str | None = None
    n_cpus: int = 1
    refine: str | None = None
    min_dist_mesh: bool = False
    no_shape_retrieval: bool = False


@dataclass(frozen=True)
class FixedMeshes:
    vertices_1: np.ndarray
    faces_1: np.ndarray
    vertices_2: np.ndarray
    faces_2: np.ndarray


def run_pipeline(config: PipelineConfig) -> None:
    validate_config(config)

    descriptor = config.descriptor.upper()
    fixed_meshes = _fix_meshes(config) if config.fix_meshes else None

    if config.min_dist_mesh:
        _calculate_minimum_distance(config, fixed_meshes)

    if config.no_shape_retrieval:
        logger.info("Shape retrieval calculation disabled.")
        return

    if descriptor in {"WKS", "HKS"}:
        _run_spectral_descriptors(config, descriptor, fixed_meshes)
        return

    if descriptor == "3DZD":
        _run_zernike_descriptors(config, fixed_meshes)
        return

    raise ValueError(f"Descriptor type not implemented: {config.descriptor}")


def validate_config(config: PipelineConfig) -> None:
    validate_input_file(config.mesh1, "mesh1")
    validate_input_file(config.mesh2, "mesh2")

    if len(config.entry_ids) != 2:
        raise ValueError(
            f"--entry-ids must contain exactly two values, got {len(config.entry_ids)}."
        )

    descriptor = config.descriptor.upper()
    if descriptor not in SUPPORTED_DESCRIPTORS:
        raise ValueError(
            f"--descr must be one of {', '.join(sorted(SUPPORTED_DESCRIPTORS))}; got {config.descriptor}."
        )

    if config.reconstruct_mesh and not config.fix_meshes:
        raise ValueError("--reconstruct-mesh must be used together with --fix-meshes.")

    if config.collapse_vertices and not config.fix_meshes:
        raise ValueError("--collapse-vertices must be used together with --fix-meshes.")

    if config.refine is not None and config.refine not in SUPPORTED_REFINEMENT:
        raise ValueError(
            f"--refine must be one of {', '.join(sorted(SUPPORTED_REFINEMENT))}."
        )

    if config.n_cpus < 1:
        raise ValueError("--n-cpus must be greater than or equal to 1.")

    if config.neigvecs < 1 or config.n_ev < 1 or config.n_descr < 1 or config.step < 1:
        raise ValueError(
            "--neigvecs, --n-ev, --n-descr, and --step must all be positive."
        )

    config.output.mkdir(parents=True, exist_ok=True)


def validate_input_file(path: Path, label: str) -> None:
    expanded_path = path.expanduser()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"Input {label} file not found: {expanded_path}")

    if not os.access(expanded_path, os.R_OK):
        raise PermissionError(f"Input {label} file is not readable: {expanded_path}")


def resolve_executable(binary: str, label: str) -> str:
    candidate = Path(binary).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)

    resolved = shutil.which(binary)
    if resolved:
        return resolved

    raise FileNotFoundError(
        f"{label} binary not found. Provide a valid --{label}-binary path or make sure "
        f"{binary!r} is available on PATH."
    )


def _fix_meshes(config: PipelineConfig) -> FixedMeshes:
    logger.info("Repairing meshes: %s and %s", config.mesh1, config.mesh2)
    vertices_1, faces_1 = fix_mesh(
        str(config.mesh1),
        config.resolution,
        config.collapse_vertices,
        config.reconstruct_mesh,
    )
    vertices_2, faces_2 = fix_mesh(
        str(config.mesh2),
        config.resolution,
        config.collapse_vertices,
        config.reconstruct_mesh,
    )
    return FixedMeshes(vertices_1, faces_1, vertices_2, faces_2)


def _calculate_minimum_distance(
    config: PipelineConfig, fixed_meshes: FixedMeshes | None
) -> None:
    if fixed_meshes is None:
        mesh1 = mesh.TriMesh(str(config.mesh1), area_normalize=False, center=False)
        mesh2 = mesh.TriMesh(str(config.mesh2), area_normalize=False, center=False)
    else:
        mesh1 = mesh.TriMesh(fixed_meshes.vertices_1, fixed_meshes.faces_1)
        mesh2 = mesh.TriMesh(fixed_meshes.vertices_2, fixed_meshes.faces_2)

    min_distance = find_minimum_distance_meshes(mesh1, mesh2)
    logger.info("Minimum distance is: %s", min_distance)


def _build_area_normalized_meshes(
    config: PipelineConfig,
    fixed_meshes: FixedMeshes | None,
) -> tuple[Any, Any]:
    if fixed_meshes is None:
        mesh1 = mesh.TriMesh(str(config.mesh1), area_normalize=True, center=False)
        mesh2 = mesh.TriMesh(str(config.mesh2), area_normalize=True, center=False)
        return mesh1, mesh2

    mesh1 = mesh.TriMesh(
        fixed_meshes.vertices_1, fixed_meshes.faces_1, area_normalize=True, center=False
    )
    mesh2 = mesh.TriMesh(
        fixed_meshes.vertices_2, fixed_meshes.faces_2, area_normalize=True, center=False
    )
    return mesh1, mesh2


def _run_spectral_descriptors(
    config: PipelineConfig,
    descriptor: str,
    fixed_meshes: FixedMeshes | None,
) -> None:
    entry_id_1, entry_id_2 = config.entry_ids
    mesh1, mesh2 = _build_area_normalized_meshes(config, fixed_meshes)

    output_file_1 = config.output / f"{descriptor}_descr_{entry_id_1}.csv"
    output_file_2 = config.output / f"{descriptor}_descr_{entry_id_2}.csv"
    output_fm = config.output / f"{entry_id_1}_{entry_id_2}_FM.csv"
    output_p2p21 = config.output / f"{entry_id_1}_{entry_id_2}_p2p21.csv"

    descriptor_files_exist = output_file_1.exists() and output_file_2.exists()
    correspondence_files_exist = output_fm.exists() and output_p2p21.exists()

    if correspondence_files_exist:
        fm = _read_functional_map(output_fm)
        _log_similarity_score(fm)
        return

    logger.info(
        "Calculating %s descriptors for structures %s and %s",
        descriptor,
        entry_id_1,
        entry_id_2,
    )
    model = set_FM_model_parameters(
        mesh1,
        mesh2,
        config.neigvecs,
        config.n_ev,
        config.n_descr,
        config.step,
        config.landmarks,
        descriptor,
    )

    if not descriptor_files_exist:
        save_data_to_csv(np.array(model.descr1), str(output_file_1))
        save_data_to_csv(np.array(model.descr2), str(output_file_2))

    fm_config = FunctionalMapConfig(n_cpus=config.n_cpus, refine=config.refine)
    model_fm, fm = calculate_functional_maps(model, fm_config)
    p2p21 = calculate_p2p_map(model_fm, config.n_cpus)

    save_data_to_csv(fm, str(output_fm))
    save_list_to_csv(p2p21, str(output_p2p21))
    _log_similarity_score(fm)


def _read_functional_map(path: Path) -> np.ndarray:
    with path.open(newline="") as fm_file:
        return np.asarray(list(csv.reader(fm_file)), dtype=float)


def _log_similarity_score(fm: np.ndarray) -> None:
    score = calculate_geodesic_norm_score(fm)
    logger.info("Shape dissimilarity score is: %s", score)


def _run_zernike_descriptors(
    config: PipelineConfig, fixed_meshes: FixedMeshes | None
) -> None:
    map2zernike_binary = resolve_executable(config.map2zernike_binary, "map2zernike")
    obj2grid_binary = resolve_executable(config.obj2grid_binary, "obj2grid")
    entry_id_1, entry_id_2 = config.entry_ids

    if fixed_meshes is not None:
        mesh_1 = trimesh.Trimesh(
            vertices=fixed_meshes.vertices_1, faces=fixed_meshes.faces_1
        )
        mesh_2 = trimesh.Trimesh(
            vertices=fixed_meshes.vertices_2, faces=fixed_meshes.faces_2
        )
        output1_obj = config.output / f"{entry_id_1}_fixed.obj"
        output2_obj = config.output / f"{entry_id_2}_fixed.obj"
        mesh_1.export(output1_obj)
        mesh_2.export(output2_obj)
        get_inv(
            str(output1_obj),
            entry_id_1,
            map2zernike_binary,
            obj2grid_binary,
            str(config.output),
        )
        get_inv(
            str(output2_obj),
            entry_id_2,
            map2zernike_binary,
            obj2grid_binary,
            str(config.output),
        )
        return

    _validate_zernike_mesh(config.mesh1)
    _validate_zernike_mesh(config.mesh2)
    get_inv(
        str(config.mesh1),
        entry_id_1,
        map2zernike_binary,
        obj2grid_binary,
        str(config.output),
    )
    get_inv(
        str(config.mesh2),
        entry_id_2,
        map2zernike_binary,
        obj2grid_binary,
        str(config.output),
    )


def _validate_zernike_mesh(path: Path) -> None:
    if path.suffix.lower() != ".obj":
        raise ValueError(f"Zernike descriptors take .obj files as input: {path}")
