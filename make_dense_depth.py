#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
MODULE_ROOT = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_module.pipeline import run_foundation_stereo_depth


DEFAULT_CKPT = MODULE_ROOT / "pretrained_models" / "11-33-40" / "model_best_bp2.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stereo dense depth maps and a fused point cloud from a COLMAP undistorted scene."
    )
    parser.add_argument("source_folder", help="Source folder containing undistorted/")
    parser.add_argument(
        "--ckpt-dir",
        default=str(DEFAULT_CKPT),
        help=f"FoundationStereo checkpoint path (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in meters for fused point cloud filtering (default: 0.05)",
    )
    parser.add_argument(
        "--fusion-chunk-size",
        type=int,
        default=10,
        help="Number of stereo pairs to merge before chunk-level voxel filtering (default: 10)",
    )
    parser.add_argument(
        "--voxel-backend",
        default="torch-cuda",
        choices=["auto", "torch-cuda", "numpy"],
        help="Voxel filtering backend (default: torch-cuda)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many stereo pairs",
    )
    parser.add_argument(
        "--image-stride",
        type=int,
        default=1,
        help="Process every Nth stereo pair after sorting by basename (default: 1)",
    )
    parser.add_argument(
        "--overwrite-sparse",
        action="store_true",
        help="Regenerate sparse depth even if existing sparse depth is complete",
    )
    parser.add_argument(
        "--overwrite-depth",
        action="store_true",
        help="Regenerate stereo dense depth even if existing outputs are complete",
    )
    parser.add_argument(
        "--depth-short-edge",
        type=int,
        default=720,
        help="Default dense-depth short edge in pixels when --full-res-depth is omitted (default: 720)",
    )
    parser.add_argument(
        "--full-res-depth",
        action="store_true",
        help="Run dense depth at the original registered image resolution",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Additional FoundationStereo input scale in (0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--hiera",
        action="store_true",
        help="Enable hierarchical FoundationStereo inference",
    )
    parser.add_argument(
        "--valid-iters",
        type=int,
        default=32,
        help="Number of FoundationStereo refinement iterations (default: 32)",
    )
    parser.add_argument(
        "--disable-depth-edge-filter",
        action="store_true",
        help="Disable inverse-depth discontinuity filtering before stereo coherence checks and fusion",
    )
    parser.add_argument(
        "--depth-edge-percentile",
        type=float,
        default=95.0,
        help="Reject pixels at or above this inverse-depth gradient percentile (default: 95)",
    )
    parser.add_argument(
        "--depth-edge-threshold",
        type=float,
        default=None,
        help="Use a fixed inverse-depth gradient threshold instead of percentile filtering",
    )
    parser.add_argument(
        "--depth-edge-dilation",
        type=int,
        default=5,
        help="Dilate detected depth discontinuities with an NxN kernel (default: 5)",
    )
    parser.add_argument(
        "--disable-coherence-filter",
        action="store_true",
        help="Disable left/right stereo coherence filtering",
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.05,
        help="Inverse-depth disagreement threshold for left/right coherence filtering (default: 0.05)",
    )
    parser.add_argument(
        "--coherence-dilation",
        type=int,
        default=3,
        help="Dilate stereo-inconsistent regions with an NxN kernel (default: 3)",
    )
    parser.add_argument(
        "--disable-chunk-density-filter",
        action="store_true",
        help="Disable relative voxel-density filtering on fused chunks",
    )
    parser.add_argument(
        "--chunk-density-relative",
        type=float,
        default=0.25,
        help="Keep chunk voxels with local density at least this fraction of the chunk maximum (default: 0.25)",
    )
    parser.add_argument(
        "--chunk-density-radius-voxels",
        type=int,
        default=2,
        help="Voxel-neighborhood radius for chunk density filtering (default: 2)",
    )
    parser.add_argument(
        "--chunk-density-min-neighbors",
        type=int,
        default=4,
        help="Minimum local occupied-voxel count for chunk density filtering (default: 4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_folder = Path(args.source_folder).expanduser().resolve()
    if not source_folder.is_dir():
        raise RuntimeError(f"Source folder is not a directory: {source_folder}")
    output_dir, fused_path = run_foundation_stereo_depth(
        source_folder,
        ckpt_path=Path(args.ckpt_dir).expanduser().resolve(),
        device_name=args.device,
        max_pairs=args.max_images,
        image_stride=args.image_stride,
        full_res_depth=args.full_res_depth,
        overwrite_sparse=args.overwrite_sparse,
        overwrite_depth=args.overwrite_depth,
        depth_short_edge=args.depth_short_edge,
        scale=args.scale,
        hiera=args.hiera,
        valid_iters=args.valid_iters,
        depth_edge_filter=not args.disable_depth_edge_filter,
        depth_edge_percentile=args.depth_edge_percentile,
        depth_edge_threshold=args.depth_edge_threshold,
        depth_edge_dilation=args.depth_edge_dilation,
        coherence_filter=not args.disable_coherence_filter,
        coherence_threshold=args.coherence_threshold,
        coherence_dilation=args.coherence_dilation,
        voxel_size=args.voxel_size,
        fusion_chunk_size=args.fusion_chunk_size,
        voxel_backend=args.voxel_backend,
        chunk_density_filter=not args.disable_chunk_density_filter,
        chunk_density_relative=args.chunk_density_relative,
        chunk_density_radius_voxels=args.chunk_density_radius_voxels,
        chunk_density_min_neighbors=args.chunk_density_min_neighbors,
    )
    print(f"[done] FoundationStereo depth output: {output_dir}")
    print(f"[done] FoundationStereo fused dense point cloud: {fused_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
