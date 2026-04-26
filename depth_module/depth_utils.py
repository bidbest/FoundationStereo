from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .colmap import SparseDepthItem, depth_paths, item_depth_size, output_stem, write_depth_png, write_depth_preview


DENSE_MODE_MARKER = ".dense_depth_mode.txt"


def load_rgb(image_path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if size is None:
        return image_rgb
    width, height = size
    if image_rgb.shape[:2] == (height, width):
        return image_rgb
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def dense_depth_is_complete(
    output_dir: Path,
    independent_dir: Path,
    projected_dir: Path,
    diff_dir: Path,
    items: list[SparseDepthItem],
    *,
    expected_marker: str,
    coherence_filter: bool,
) -> bool:
    marker_path = output_dir / DENSE_MODE_MARKER
    if not marker_path.is_file() or marker_path.read_text(encoding="utf-8").strip() != expected_marker:
        return False

    for item in items:
        final_npy, final_png, final_preview = depth_paths(output_dir, item.image)
        independent_npy, independent_png, independent_preview = depth_paths(independent_dir, item.image)
        required_paths = [
            final_npy,
            final_png,
            final_preview,
            independent_npy,
            independent_png,
            independent_preview,
        ]
        if coherence_filter:
            projected_npy, projected_png, projected_preview = depth_paths(projected_dir, item.image)
            required_paths.extend([projected_npy, projected_png, projected_preview])
            required_paths.append(diff_dir / f"diff_depth_{output_stem(item.image)}.png")
        if any(not path.is_file() for path in required_paths):
            return False
        width, height = item_depth_size(item)
        try:
            final_shape = np.load(final_npy, mmap_mode="r").shape
            independent_shape = np.load(independent_npy, mmap_mode="r").shape
        except ValueError:
            return False
        if final_shape != (height, width) or independent_shape != (height, width):
            return False
        if coherence_filter:
            try:
                projected_shape = np.load(projected_npy, mmap_mode="r").shape
            except ValueError:
                return False
            if projected_shape != (height, width):
                return False
    return True


def dense_mode_marker(
    *,
    coherence_filter: bool,
    coherence_threshold: float,
    coherence_dilation: int,
    edge_filter: bool,
    edge_percentile: float,
    edge_threshold: float | None,
    edge_dilation: int,
    valid_iters: int,
    scale: float,
    hiera: bool,
    ckpt_dir: str,
) -> str:
    threshold_label = "none" if edge_threshold is None else f"{edge_threshold:.8g}"
    return (
        f"foundation_stereo coherence={int(coherence_filter)} "
        f"coherence_threshold={coherence_threshold:.8g} coherence_dilation={coherence_dilation} "
        f"edge_filter={int(edge_filter)} edge_percentile={edge_percentile:.8g} "
        f"edge_threshold={threshold_label} edge_dilation={edge_dilation} "
        f"valid_iters={valid_iters} scale={scale:.8g} hiera={int(hiera)} ckpt={ckpt_dir}"
    )


def save_depth_outputs(output_dir: Path, item: SparseDepthItem, suffix: str, depth: np.ndarray) -> None:
    stem = output_stem(item.image)
    np.save(output_dir / f"{stem}{suffix}.npy", depth)
    write_depth_png(output_dir / f"{stem}{suffix}.png", depth)
    write_depth_preview(output_dir / f"{stem}{suffix}_vis.png", depth)


def save_final_depth_outputs(output_dir: Path, item: SparseDepthItem, depth: np.ndarray) -> None:
    npy_path, png_path, preview_path = depth_paths(output_dir, item.image)
    np.save(npy_path, depth)
    write_depth_png(png_path, depth)
    write_depth_preview(preview_path, depth)


def write_diff_heatmap(path: Path, diff: np.ndarray, valid: np.ndarray, threshold: float) -> None:
    heatmap_input = np.zeros(diff.shape, dtype=np.uint8)
    if valid.any():
        scale = max(threshold * 4.0, 1e-6)
        heatmap_input[valid] = np.clip(diff[valid] / scale * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_input, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    cv2.imwrite(str(path), heatmap)


def backproject_depth(depth: np.ndarray, intrinsics_normalized: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    fx = intrinsics_normalized[0, 0] * width
    fy = intrinsics_normalized[1, 1] * height
    cx = intrinsics_normalized[0, 2] * width
    cy = intrinsics_normalized[1, 2] * height
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def camera_to_world(points_camera: np.ndarray, item: SparseDepthItem) -> np.ndarray:
    rotation = item.image.qvec2rotmat()
    return (points_camera - item.image.tvec) @ rotation


def world_to_camera(points_world: np.ndarray, item: SparseDepthItem) -> np.ndarray:
    rotation = item.image.qvec2rotmat()
    return points_world @ rotation.T + item.image.tvec
