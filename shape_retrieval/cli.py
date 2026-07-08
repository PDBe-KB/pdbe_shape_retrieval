from __future__ import annotations

import argparse
import cProfile
import io
import logging
import multiprocessing
import os
import pstats
from pathlib import Path
from typing import Sequence

from shape_retrieval.pipeline import PipelineConfig, run_pipeline

logger = logging.getLogger(__name__)

PROFILE_ENV_VAR = "SHAPE_RETRIEVAL_PROFILE"
PROFILE_OUTPUT_ENV_VAR = "SHAPE_RETRIEVAL_PROFILE_OUTPUT"
LOG_LEVEL_ENV_VAR = "SHAPE_RETRIEVAL_LOG_LEVEL"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate shape retrieval descriptors for protein surfaces."
    )
    default_cpu_count = multiprocessing.cpu_count()

    parser.add_argument(
        "--input-mesh1",
        "--mesh1",
        dest="mesh1",
        help="Input path to triangulated mesh file 1.",
        required=True,
    )
    parser.add_argument(
        "--input-mesh2",
        "--mesh2",
        dest="mesh2",
        help="Input path to triangulated mesh file 2.",
        required=True,
    )
    parser.add_argument(
        "--entry-ids",
        nargs=2,
        metavar=("ENTRY_ID_1", "ENTRY_ID_2"),
        help="Entry IDs for the two protein structures.",
        required=True,
    )
    parser.add_argument("-o", "--output", help="Path to output files.", required=True)
    parser.add_argument(
        "--descr",
        type=str.upper,
        choices=("WKS", "HKS", "3DZD"),
        default="WKS",
        help="Type of descriptor to calculate.",
    )
    parser.add_argument(
        "--fix-meshes",
        action="store_true",
        help="Preprocess meshes to be well-conditioned before computing descriptors.",
    )
    parser.add_argument(
        "--collapse-vertices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reduce mesh resolution using decimation quadric edge collapse.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Factor used when collapsing vertices, for example 0.5.",
    )
    parser.add_argument(
        "--reconstruct-mesh",
        action="store_true",
        default=False,
        help="Reconstruct the mesh before repair.",
    )
    parser.add_argument(
        "--map2zernike-binary",
        default=default_map2zernike_binary(),
        help="Path or command name for the map2zernike binary.",
    )
    parser.add_argument(
        "--obj2grid-binary",
        default=default_obj2grid_binary(),
        help="Path or command name for the obj2grid binary.",
    )
    parser.add_argument(
        "--neigvecs",
        type=int,
        default=200,
        help="Number of eigenvalues/eigenvectors to process.",
    )
    parser.add_argument(
        "--n-ev",
        type=int,
        default=50,
        help="Number of Laplacian eigenvalues to consider for the functional map.",
    )
    parser.add_argument(
        "--n-descr",
        type=int,
        default=100,
        help="Number of descriptors to process for spectral calculations.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Subsample step to avoid using too many spectral descriptors.",
    )
    parser.add_argument(
        "--landmarks",
        default=None,
        help="Input indices of landmarks for spectral descriptors.",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=default_cpu_count,
        help="Number of threads to use for this calculation.",
    )
    parser.add_argument(
        "--refine",
        choices=("icp", "zoomout"),
        default=None,
        help="Refining method for calculation of functional maps.",
    )
    parser.add_argument(
        "--min-dist-mesh",
        action="store_true",
        help="Calculate minimum distance between the two meshes.",
    )
    parser.add_argument(
        "--no-shape-retrieval",
        action="store_true",
        help="Switch off shape retrieval calculation.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mesh1=Path(args.mesh1).expanduser(),
        mesh2=Path(args.mesh2).expanduser(),
        entry_ids=tuple(args.entry_ids),
        output=Path(args.output).expanduser(),
        descriptor=args.descr,
        fix_meshes=args.fix_meshes,
        collapse_vertices=args.collapse_vertices,
        resolution=args.resolution,
        reconstruct_mesh=args.reconstruct_mesh,
        map2zernike_binary=args.map2zernike_binary,
        obj2grid_binary=args.obj2grid_binary,
        neigvecs=args.neigvecs,
        n_ev=args.n_ev,
        n_descr=args.n_descr,
        step=args.step,
        landmarks=args.landmarks,
        n_cpus=args.n_cpus,
        refine=args.refine,
        min_dist_mesh=args.min_dist_mesh,
        no_shape_retrieval=args.no_shape_retrieval,
    )


def configure_logging() -> None:
    level_name = os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def default_map2zernike_binary() -> str:
    explicit_binary = os.environ.get("MAP2ZERNIKE_BINARY")
    if explicit_binary:
        return explicit_binary

    setup_dir = os.environ.get("MAP2ZERNIKE_SETUP_DIR")
    if setup_dir:
        return binary_from_env_path(setup_dir, "map2zernike")

    return "map2zernike"


def default_obj2grid_binary() -> str:
    explicit_binary = os.environ.get("OBJ2GRID_BINARY")
    if explicit_binary:
        return explicit_binary

    obj2grid_path = os.environ.get("OBJ2GRID_PATH")
    if obj2grid_path:
        return binary_from_env_path(obj2grid_path, "obj2grid")

    return "obj2grid"


def binary_from_env_path(value: str, binary_name: str) -> str:
    for raw_part in reversed(value.split(os.pathsep)):
        part = raw_part.strip()
        if not part:
            continue

        candidate = Path(part).expanduser()
        if candidate.name == binary_name:
            return str(candidate)

        return str(candidate / binary_name)

    return binary_name


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def run_with_profiling(config: PipelineConfig) -> None:
    profile = cProfile.Profile()
    profile.enable()
    try:
        run_pipeline(config)
    finally:
        profile.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("tottime")
    stats.print_stats()

    output_path = Path(
        os.environ.get(PROFILE_OUTPUT_ENV_VAR, "shape_retrieval_profile.txt")
    ).expanduser()
    output_path.write_text(stream.getvalue())
    logger.info("Wrote profile output to %s", output_path)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        if env_flag(PROFILE_ENV_VAR):
            run_with_profiling(config)
        else:
            run_pipeline(config)
    except Exception:
        logger.exception("Shape retrieval failed.")
        return 1

    return 0
