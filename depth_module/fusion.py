from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .colmap import SparseDepthItem, depth_paths, item_depth_size, normalized_intrinsics_for_size, output_stem
from .depth_filter import inverse_depth_discontinuity_mask
from .depth_utils import backproject_depth, camera_to_world, load_rgb, write_depth_preview


MAX_DENSITY_GRID_VOXELS = 64_000_000


def add_timing(timings: dict[str, float], key: str, elapsed: float) -> None:
    timings[key] = timings.get(key, 0.0) + elapsed


def resolve_voxel_backend(voxel_backend: str) -> str:
    if voxel_backend == "auto":
        try:
            import torch
        except ImportError:
            return "numpy"
        return "torch-cuda" if torch.cuda.is_available() else "numpy"
    return voxel_backend


def pack_voxel_keys_numpy(keys: np.ndarray) -> np.ndarray:
    mins = keys.min(axis=0)
    shifted = keys - mins
    extents = shifted.max(axis=0).astype(np.int64) + 1
    return shifted[:, 0] + shifted[:, 1] * extents[0] + shifted[:, 2] * extents[0] * extents[1]


def voxel_filter_numpy(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    keys = np.floor(points / voxel_size).astype(np.int64)
    packed_keys = pack_voxel_keys_numpy(keys)
    _, unique_indices = np.unique(packed_keys, return_index=True)
    return points[unique_indices].astype(np.float32), colors[unique_indices].astype(np.uint8)


def voxel_filter_torch_cuda(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    timings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if points.size == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)

    device = torch.device("cuda")
    copy_start = perf_counter()
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device)
    colors_t = torch.as_tensor(colors, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    add_timing(timings, "voxel_cuda_copy_in", perf_counter() - copy_start)

    key_start = perf_counter()
    keys = torch.floor(points_t / voxel_size).to(torch.int64)
    mins = keys.amin(dim=0)
    shifted = keys - mins
    extents = shifted.amax(dim=0) + 1
    packed = shifted[:, 0] + shifted[:, 1] * extents[0] + shifted[:, 2] * extents[0] * extents[1]
    torch.cuda.synchronize(device)
    add_timing(timings, "voxel_cuda_key_build", perf_counter() - key_start)

    sort_start = perf_counter()
    sorted_keys, order = torch.sort(packed)
    keep = torch.empty_like(sorted_keys, dtype=torch.bool)
    keep[0] = True
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    unique_indices = order[keep]
    filtered_points_t = points_t[unique_indices]
    filtered_colors_t = colors_t[unique_indices]
    torch.cuda.synchronize(device)
    add_timing(timings, "voxel_cuda_sort_unique", perf_counter() - sort_start)

    copy_out_start = perf_counter()
    filtered_points = filtered_points_t.cpu().numpy().astype(np.float32)
    filtered_colors = filtered_colors_t.cpu().numpy().astype(np.uint8)
    torch.cuda.synchronize(device)
    add_timing(timings, "voxel_cuda_copy_out", perf_counter() - copy_out_start)
    return filtered_points, filtered_colors


def voxel_filter(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    *,
    backend: str,
    timings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "numpy":
        return voxel_filter_numpy(points, colors, voxel_size)
    if backend == "torch-cuda":
        try:
            return voxel_filter_torch_cuda(points, colors, voxel_size, timings)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "no nvidia driver" not in message:
                raise
            print("[warn] CUDA voxel filter failed; retrying with numpy")
            return voxel_filter_numpy(points, colors, voxel_size)
    raise RuntimeError(f"Unsupported voxel backend: {backend}")


def shifted_voxel_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mins = keys.min(axis=0)
    shifted = keys - mins
    extents = shifted.max(axis=0).astype(np.int64) + 1
    volume = int(extents[0]) * int(extents[1]) * int(extents[2])
    return shifted, extents, volume


def density_count_dtype(radius_voxels: int):
    neighborhood_size = (2 * radius_voxels + 1) ** 3
    return np.uint16 if neighborhood_size <= np.iinfo(np.uint16).max else np.uint32


def local_voxel_density_counts_dense_grid(shifted: np.ndarray, extents: np.ndarray, radius_voxels: int) -> np.ndarray:
    from scipy import ndimage

    count_dtype = density_count_dtype(radius_voxels)
    shape = tuple(int(extent) for extent in extents)
    occupied = np.zeros(shape, dtype=np.uint8)
    occupied[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = 1
    kernel = np.ones((2 * radius_voxels + 1,) * 3, dtype=count_dtype)
    counts_grid = ndimage.convolve(occupied.astype(count_dtype), kernel, mode="constant", cval=0)
    return counts_grid[shifted[:, 0], shifted[:, 1], shifted[:, 2]].astype(np.int32)


def local_voxel_density_counts_sparse_search(shifted: np.ndarray, extents: np.ndarray, radius_voxels: int) -> np.ndarray:
    packed = shifted[:, 0] + shifted[:, 1] * extents[0] + shifted[:, 2] * extents[0] * extents[1]
    sorted_packed = np.sort(packed)
    counts = np.zeros(len(shifted), dtype=np.int32)
    indices = np.arange(len(shifted))
    for dx in range(-radius_voxels, radius_voxels + 1):
        nx = shifted[:, 0] + dx
        valid_x = (nx >= 0) & (nx < extents[0])
        if not valid_x.any():
            continue
        for dy in range(-radius_voxels, radius_voxels + 1):
            ny = shifted[:, 1] + dy
            valid_xy = valid_x & (ny >= 0) & (ny < extents[1])
            if not valid_xy.any():
                continue
            for dz in range(-radius_voxels, radius_voxels + 1):
                nz = shifted[:, 2] + dz
                valid = valid_xy & (nz >= 0) & (nz < extents[2])
                if not valid.any():
                    continue
                valid_indices = indices[valid]
                queries = nx[valid] + ny[valid] * extents[0] + nz[valid] * extents[0] * extents[1]
                positions = np.searchsorted(sorted_packed, queries)
                in_range = positions < len(sorted_packed)
                found = np.zeros(len(queries), dtype=bool)
                found[in_range] = sorted_packed[positions[in_range]] == queries[in_range]
                counts[valid_indices[found]] += 1
    return counts


def local_voxel_density_counts(keys: np.ndarray, radius_voxels: int) -> tuple[np.ndarray, str]:
    shifted, extents, volume = shifted_voxel_keys(keys)
    if volume <= MAX_DENSITY_GRID_VOXELS:
        return local_voxel_density_counts_dense_grid(shifted, extents, radius_voxels), "dense_grid"
    return local_voxel_density_counts_sparse_search(shifted, extents, radius_voxels), "sparse_search"


def chunk_relative_density_filter(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    voxel_size: float,
    relative_density: float,
    radius_voxels: int,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, str]:
    if points.size == 0:
        return points.reshape(0, 3), colors.reshape(0, 3), 0, 0, min_neighbors, "empty"
    keys = np.floor(points / voxel_size).astype(np.int64)
    counts, density_backend = local_voxel_density_counts(keys, radius_voxels)
    max_count = int(counts.max(initial=0))
    threshold = max(min_neighbors, int(np.ceil(max_count * relative_density)))
    keep = counts >= threshold
    return (
        points[keep].astype(np.float32),
        colors[keep].astype(np.uint8),
        int(np.count_nonzero(~keep)),
        max_count,
        threshold,
        density_backend,
    )


def export_point_cloud(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    import trimesh

    if points.size == 0:
        raise RuntimeError("No valid dense points available for point cloud export")
    trimesh.PointCloud(points, colors).export(output_path)


def flush_chunk(
    chunk_points: list[np.ndarray],
    chunk_colors: list[np.ndarray],
    global_points: list[np.ndarray],
    global_colors: list[np.ndarray],
    *,
    voxel_size: float,
    chunk_index: int,
    backend: str,
    pcds_dir: Path,
    chunk_density_filter: bool,
    chunk_density_relative: float,
    chunk_density_radius_voxels: int,
    chunk_density_min_neighbors: int,
    timings: dict[str, float],
) -> int:
    if not chunk_points:
        return 0
    points = np.concatenate(chunk_points, axis=0)
    colors = np.concatenate(chunk_colors, axis=0)
    input_count = len(points)
    voxel_start = perf_counter()
    points, colors = voxel_filter(points, colors, voxel_size, backend=backend, timings=timings)
    add_timing(timings, "chunk_voxel_filter", perf_counter() - voxel_start)

    density_start = perf_counter()
    density_removed = 0
    density_max_count = 0
    density_threshold = 0
    density_backend = "disabled"
    if chunk_density_filter:
        points, colors, density_removed, density_max_count, density_threshold, density_backend = chunk_relative_density_filter(
            points,
            colors,
            voxel_size=voxel_size,
            relative_density=chunk_density_relative,
            radius_voxels=chunk_density_radius_voxels,
            min_neighbors=chunk_density_min_neighbors,
        )
    add_timing(timings, "chunk_density_filter", perf_counter() - density_start)
    if points.size == 0:
        print(f"[fuse] chunk={chunk_index} input_points={input_count} chunk_voxels=0 density_removed={density_removed}")
        return 0

    export_start = perf_counter()
    export_point_cloud(points, colors, pcds_dir / f"chunk_{chunk_index:04d}.ply")
    add_timing(timings, "export_chunk_pcd", perf_counter() - export_start)

    merge_start = perf_counter()
    global_points.append(points)
    global_colors.append(colors)
    add_timing(timings, "global_merge", perf_counter() - merge_start)
    print(
        f"[fuse] chunk={chunk_index} input_points={input_count} chunk_voxels={len(points)} "
        f"density_removed={density_removed} density_max={density_max_count} "
        f"density_threshold={density_threshold} density_backend={density_backend}"
    )
    return len(points)


def run_fuse_pcd(
    source_folder: Path,
    items: list[SparseDepthItem],
    *,
    voxel_size: float,
    chunk_size: int,
    voxel_backend: str,
    full_res_depth: bool,
    depth_edge_filter: bool,
    depth_edge_percentile: float,
    depth_edge_threshold: float | None,
    depth_edge_dilation: int,
    chunk_density_filter: bool,
    chunk_density_relative: float,
    chunk_density_radius_voxels: int,
    chunk_density_min_neighbors: int,
) -> Path:
    if voxel_size <= 0:
        raise RuntimeError(f"Voxel size must be positive: {voxel_size}")
    if chunk_size < 1:
        raise RuntimeError(f"Chunk size must be >= 1: {chunk_size}")
    backend = resolve_voxel_backend(voxel_backend)

    undistorted_dir = source_folder / "undistorted"
    depth_dir = undistorted_dir / ("depth_full_res" if full_res_depth else "depth")
    pcds_dir = undistorted_dir / "pcds"
    fused_path = undistorted_dir / "fused_dense_point_cloud.ply"
    if not depth_dir.is_dir():
        raise RuntimeError(f"Missing dense depth folder: {depth_dir}")

    pcds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Voxel backend: {backend}")
    timings: dict[str, float] = {}
    global_points: list[np.ndarray] = []
    global_colors: list[np.ndarray] = []
    chunk_points: list[np.ndarray] = []
    chunk_colors: list[np.ndarray] = []
    chunk_index = 0
    chunk_image_count = 0

    for item in items:
        dense_npy, _, _ = depth_paths(depth_dir, item.image)
        if not dense_npy.is_file():
            raise RuntimeError(f"Missing dense depth for {item.image.name}: {dense_npy}")
        width, height = item_depth_size(item)
        image_rgb = load_rgb(item.image_path, (width, height))
        dense_depth = np.load(dense_npy).astype(np.float32)
        if dense_depth.shape != image_rgb.shape[:2]:
            raise RuntimeError(
                f"Dense depth shape {dense_depth.shape} does not match image {item.image.name} shape {image_rgb.shape[:2]}"
            )
        raw_stem = Path(item.image.name).with_suffix("").as_posix().replace("/", "__")
        rejected_edges = 0
        if depth_edge_filter:
            reject_mask = inverse_depth_discontinuity_mask(
                dense_depth,
                percentile=depth_edge_percentile,
                fixed_threshold=depth_edge_threshold,
                dilation=depth_edge_dilation,
            )
            rejected_edges = int(np.count_nonzero(reject_mask))
            if rejected_edges:
                dense_depth = dense_depth.copy()
                dense_depth[reject_mask] = 0.0
            cv2.imwrite(str(pcds_dir / f"edge_mask_{item.image.id:08d}_{raw_stem}.png"), reject_mask.astype(np.uint8) * 255)

        filtered_preview_path = depth_dir / f"{output_stem(item.image)}_vis_filtered.png"
        write_depth_preview(filtered_preview_path, dense_depth)

        points_camera = backproject_depth(dense_depth, normalized_intrinsics_for_size(item.camera, width, height))
        valid = np.isfinite(points_camera).all(axis=-1) & (points_camera[..., 2] > 0)
        points_camera = points_camera[valid]
        colors = image_rgb[valid]
        export_point_cloud(points_camera.astype(np.float32), colors.astype(np.uint8), pcds_dir / f"raw_{item.image.id:08d}_{raw_stem}.ply")
        points_world = camera_to_world(points_camera, item).astype(np.float32)
        export_point_cloud(points_world, colors.astype(np.uint8), pcds_dir / f"world_{item.image.id:08d}_{raw_stem}.ply")
        print(f"[fuse] {item.image.name}: valid_points={len(points_world)} edge_reject={rejected_edges}")
        chunk_points.append(points_world)
        chunk_colors.append(colors.astype(np.uint8))
        chunk_image_count += 1
        if chunk_image_count >= chunk_size:
            flush_chunk(
                chunk_points,
                chunk_colors,
                global_points,
                global_colors,
                voxel_size=voxel_size,
                chunk_index=chunk_index,
                backend=backend,
                pcds_dir=pcds_dir,
                chunk_density_filter=chunk_density_filter,
                chunk_density_relative=chunk_density_relative,
                chunk_density_radius_voxels=chunk_density_radius_voxels,
                chunk_density_min_neighbors=chunk_density_min_neighbors,
                timings=timings,
            )
            chunk_points = []
            chunk_colors = []
            chunk_image_count = 0
            chunk_index += 1

    if chunk_points:
        flush_chunk(
            chunk_points,
            chunk_colors,
            global_points,
            global_colors,
            voxel_size=voxel_size,
            chunk_index=chunk_index,
            backend=backend,
            pcds_dir=pcds_dir,
            chunk_density_filter=chunk_density_filter,
            chunk_density_relative=chunk_density_relative,
            chunk_density_radius_voxels=chunk_density_radius_voxels,
            chunk_density_min_neighbors=chunk_density_min_neighbors,
            timings=timings,
        )

    if not global_points:
        raise RuntimeError("No dense points available after fusion")
    points = np.concatenate(global_points, axis=0)
    colors = np.concatenate(global_colors, axis=0)
    final_points, final_colors = voxel_filter(points, colors, voxel_size, backend=backend, timings=timings)
    export_point_cloud(final_points, final_colors, fused_path)
    print(f"[done] Fused dense point cloud: {fused_path}")
    return fused_path
