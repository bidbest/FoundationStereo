from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .colmap import (
    SparseDepthItem,
    StereoPairItem,
    camera_intrinsics,
    depth_paths,
    flatten_pairs,
    item_depth_size,
    output_stem,
    prepare_sparse_depth,
    scale_intrinsics,
    select_stereo_pairs,
)
from .depth_filter import inverse_depth_discontinuity_mask
from .depth_utils import (
    DENSE_MODE_MARKER,
    dense_depth_is_complete,
    dense_mode_marker,
    load_rgb,
    save_depth_outputs,
    save_final_depth_outputs,
    write_diff_heatmap,
)
from .fusion import run_fuse_pcd


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("FoundationStereo requires CUDA; no CUDA device is available")
        return torch.device("cuda")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FoundationStereo requested CUDA, but no CUDA device is available")
        return torch.device("cuda")
    raise RuntimeError(f"Unsupported FoundationStereo device: {device_name}")


def scaled_size(width: int, height: int, short_edge: int | None) -> tuple[int, int]:
    if short_edge is None:
        return width, height
    if short_edge < 1:
        raise RuntimeError(f"Depth short edge must be >= 1: {short_edge}")
    current_short = min(width, height)
    if current_short <= short_edge:
        return width, height
    scale = short_edge / current_short
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def rectified_output_is_complete(rectified_dir: Path, basenames: list[str]) -> bool:
    metadata_path = rectified_dir / "rectification_metadata.json"
    if not metadata_path.is_file():
        return False
    left_dir = rectified_dir / "left"
    right_dir = rectified_dir / "right"
    if not left_dir.is_dir() or not right_dir.is_dir():
        return False
    for basename in basenames:
        if not (left_dir / basename).is_file() or not (right_dir / basename).is_file():
            return False
    return True


def load_rectification_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rectified_pairs(source_folder: Path, pairs: list[StereoPairItem]) -> tuple[list[StereoPairItem], dict]:
    rectified_dir = source_folder / "undistorted" / "stereo_rectified_images"
    basenames = [pair.basename for pair in pairs]
    if not rectified_output_is_complete(rectified_dir, basenames):
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.stereo_rectify import prepare_stereo_rectified_images

        prepare_stereo_rectified_images(source_folder, overwrite=True)
    metadata = load_rectification_metadata(rectified_dir / "rectification_metadata.json")
    left_dir = rectified_dir / "left"
    right_dir = rectified_dir / "right"
    resolved_pairs = [
        replace(
            pair,
            left_rectified_path=left_dir / pair.basename,
            right_rectified_path=right_dir / pair.basename,
        )
        for pair in pairs
    ]
    return resolved_pairs, metadata


def load_model(ckpt_path: Path, device, *, valid_iters: int, scale: float, hiera: bool):
    import torch
    from omegaconf import OmegaConf

    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from core.foundation_stereo import FoundationStereo

    if not ckpt_path.is_file():
        raise RuntimeError(f"Missing FoundationStereo checkpoint: {ckpt_path}")
    cfg_path = ckpt_path.parent / "cfg.yaml"
    if not cfg_path.is_file():
        raise RuntimeError(f"Missing FoundationStereo cfg.yaml next to checkpoint: {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = valid_iters
    cfg["scale"] = scale
    cfg["hiera"] = 1 if hiera else 0

    torch.autograd.set_grad_enabled(False)
    model = FoundationStereo(OmegaConf.create(cfg))
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def run_foundation_forward(
    model,
    device,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    valid_iters: int,
    scale: float,
    hiera: bool,
) -> np.ndarray:
    import torch

    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from core.utils.utils import InputPadder

    target_height, target_width = left_rgb.shape[:2]
    if scale <= 0 or scale > 1:
        raise RuntimeError(f"FoundationStereo scale must be in (0, 1]: {scale}")
    if scale < 1.0:
        infer_width = max(1, int(round(target_width * scale)))
        infer_height = max(1, int(round(target_height * scale)))
        left_infer = cv2.resize(left_rgb, (infer_width, infer_height), interpolation=cv2.INTER_AREA)
        right_infer = cv2.resize(right_rgb, (infer_width, infer_height), interpolation=cv2.INTER_AREA)
    else:
        left_infer = left_rgb
        right_infer = right_rgb

    left_tensor = torch.as_tensor(left_infer, device=device).float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_infer, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    autocast_enabled = device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        if hiera:
            disparity = model.run_hierachical(
                left_tensor,
                right_tensor,
                iters=valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )
        else:
            disparity = model.forward(left_tensor, right_tensor, iters=valid_iters, test_mode=True)
    disparity = padder.unpad(disparity.float()).data.detach().cpu().numpy().reshape(left_infer.shape[:2])
    if disparity.shape[::-1] != (target_width, target_height):
        disparity = cv2.resize(disparity, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return disparity.astype(np.float32)


def run_pair_inference(
    model,
    device,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    valid_iters: int,
    scale: float,
    hiera: bool,
) -> tuple[np.ndarray, np.ndarray]:
    left_disparity = run_foundation_forward(
        model,
        device,
        left_rgb,
        right_rgb,
        valid_iters=valid_iters,
        scale=scale,
        hiera=hiera,
    )
    right_flip_disparity = run_foundation_forward(
        model,
        device,
        np.ascontiguousarray(np.flip(right_rgb, axis=1)),
        np.ascontiguousarray(np.flip(left_rgb, axis=1)),
        valid_iters=valid_iters,
        scale=scale,
        hiera=hiera,
    )
    right_disparity = np.ascontiguousarray(np.flip(right_flip_disparity, axis=1))
    return left_disparity, right_disparity


def disparity_to_depth(disparity: np.ndarray, *, fx: float, baseline: float, scale: float) -> np.ndarray:
    depth = np.zeros(disparity.shape, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    if valid.any():
        depth[valid] = (fx * scale * baseline / disparity[valid]).astype(np.float32)
    return depth


def horizontal_project_opposite_depth(
    source_depth: np.ndarray,
    reference_depth: np.ndarray,
    *,
    fx: float,
    baseline: float,
    direction: int,
) -> np.ndarray:
    if direction not in {-1, 1}:
        raise RuntimeError(f"Invalid horizontal projection direction: {direction}")
    disparity = np.zeros(reference_depth.shape, dtype=np.float32)
    valid = np.isfinite(reference_depth) & (reference_depth > 0)
    disparity[valid] = fx * baseline / reference_depth[valid]
    height, width = reference_depth.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = x + direction * disparity
    map_y = y
    return cv2.remap(
        source_depth.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_inverse_depth_coherence(
    current_depth: np.ndarray,
    projected_depth: np.ndarray,
    *,
    threshold: float,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_valid = np.isfinite(current_depth) & (current_depth > 0)
    projected_valid = np.isfinite(projected_depth) & (projected_depth > 0)
    compare_valid = current_valid & projected_valid
    diff = np.zeros(current_depth.shape, dtype=np.float32)
    if compare_valid.any():
        diff[compare_valid] = np.abs((1.0 / current_depth[compare_valid]) - (1.0 / projected_depth[compare_valid]))
    reject = compare_valid & (diff > threshold)
    if dilation > 1 and reject.any():
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        reject = cv2.dilate(reject.astype(np.uint8), kernel, iterations=1).astype(bool)
        reject &= current_valid
    filtered = current_depth.copy()
    filtered[reject] = 0.0
    return filtered, diff, compare_valid


def backproject_depth_matrix(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def warp_rectified_depth_to_original(
    depth_rectified: np.ndarray,
    rectified_intrinsics: np.ndarray,
    rectified_to_original_rotation: np.ndarray,
    original_intrinsics: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    target_width, target_height = output_size
    target_depth = np.full((target_height, target_width), np.inf, dtype=np.float32)
    valid = np.isfinite(depth_rectified) & (depth_rectified > 0)
    if not valid.any():
        target_depth.fill(0.0)
        return target_depth

    points_rectified = backproject_depth_matrix(depth_rectified, rectified_intrinsics)[valid]
    points_original = points_rectified @ rectified_to_original_rotation.T
    positive = np.isfinite(points_original).all(axis=1) & (points_original[:, 2] > 0)
    if not positive.any():
        target_depth.fill(0.0)
        return target_depth
    points_original = points_original[positive]
    z = points_original[:, 2]
    u = np.rint(points_original[:, 0] * original_intrinsics[0, 0] / z + original_intrinsics[0, 2]).astype(np.int64)
    v = np.rint(points_original[:, 1] * original_intrinsics[1, 1] / z + original_intrinsics[1, 2]).astype(np.int64)
    in_view = (u >= 0) & (u < target_width) & (v >= 0) & (v < target_height)
    if in_view.any():
        np.minimum.at(target_depth, (v[in_view], u[in_view]), z[in_view].astype(np.float32))
    target_depth[~np.isfinite(target_depth)] = 0.0
    return target_depth


def save_depth_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_foundation_stereo_depth(
    source_folder: Path,
    *,
    ckpt_path: Path,
    device_name: str,
    max_pairs: int | None,
    image_stride: int,
    full_res_depth: bool,
    overwrite_sparse: bool,
    overwrite_depth: bool,
    depth_short_edge: int,
    scale: float,
    hiera: bool,
    valid_iters: int,
    depth_edge_filter: bool,
    depth_edge_percentile: float,
    depth_edge_threshold: float | None,
    depth_edge_dilation: int,
    coherence_filter: bool,
    coherence_threshold: float,
    coherence_dilation: int,
    voxel_size: float,
    fusion_chunk_size: int,
    voxel_backend: str,
    chunk_density_filter: bool,
    chunk_density_relative: float,
    chunk_density_radius_voxels: int,
    chunk_density_min_neighbors: int,
) -> tuple[Path, Path]:
    if image_stride < 1:
        raise RuntimeError(f"Image stride must be >= 1: {image_stride}")
    if max_pairs is not None and max_pairs < 1:
        raise RuntimeError(f"Max stereo pairs must be >= 1: {max_pairs}")
    if valid_iters < 1:
        raise RuntimeError(f"valid_iters must be >= 1: {valid_iters}")
    if not 0 < scale <= 1:
        raise RuntimeError(f"FoundationStereo scale must be in (0, 1]: {scale}")
    if depth_short_edge < 1:
        raise RuntimeError(f"Depth short edge must be >= 1: {depth_short_edge}")
    if coherence_threshold <= 0:
        raise RuntimeError(f"Coherence threshold must be positive: {coherence_threshold}")
    if coherence_dilation < 1:
        raise RuntimeError(f"Coherence dilation must be >= 1: {coherence_dilation}")
    if not 0 < depth_edge_percentile <= 100:
        raise RuntimeError(f"Depth edge percentile must be in (0, 100]: {depth_edge_percentile}")
    if depth_edge_threshold is not None and depth_edge_threshold <= 0:
        raise RuntimeError(f"Depth edge threshold must be positive: {depth_edge_threshold}")
    if depth_edge_dilation < 1:
        raise RuntimeError(f"Depth edge dilation must be >= 1: {depth_edge_dilation}")

    target_short_edge = None if full_res_depth else depth_short_edge
    pairs, _ = select_stereo_pairs(
        source_folder,
        max_pairs=max_pairs,
        image_stride=image_stride,
        target_short_edge=target_short_edge,
    )
    pairs, rectification_metadata = build_rectified_pairs(source_folder, pairs)
    items = flatten_pairs(pairs)
    prepare_sparse_depth(source_folder, items, overwrite=overwrite_sparse)

    undistorted_dir = source_folder / "undistorted"
    output_dir = undistorted_dir / ("depth_full_res" if full_res_depth else "depth")
    independent_dir = undistorted_dir / ("depth_full_res_independent" if full_res_depth else "depth_independent")
    projected_dir = undistorted_dir / ("depth_full_res_projected" if full_res_depth else "depth_projected")
    diff_dir = undistorted_dir / ("depth_full_res_diff" if full_res_depth else "depth_diff")
    output_dir.mkdir(parents=True, exist_ok=True)
    independent_dir.mkdir(parents=True, exist_ok=True)
    if coherence_filter:
        projected_dir.mkdir(parents=True, exist_ok=True)
        diff_dir.mkdir(parents=True, exist_ok=True)

    expected_marker = dense_mode_marker(
        coherence_filter=coherence_filter,
        coherence_threshold=coherence_threshold,
        coherence_dilation=coherence_dilation,
        edge_filter=depth_edge_filter,
        edge_percentile=depth_edge_percentile,
        edge_threshold=depth_edge_threshold,
        edge_dilation=depth_edge_dilation,
        valid_iters=valid_iters,
        scale=scale,
        hiera=hiera,
        ckpt_dir=str(ckpt_path),
    )
    if not overwrite_depth and dense_depth_is_complete(
        output_dir,
        independent_dir,
        projected_dir,
        diff_dir,
        items,
        expected_marker=expected_marker,
        coherence_filter=coherence_filter,
    ):
        print(f"[skip] Reusing existing dense depth: {output_dir}")
        fused_path = run_fuse_pcd(
            source_folder,
            items,
            voxel_size=voxel_size,
            chunk_size=fusion_chunk_size,
            voxel_backend=voxel_backend,
            full_res_depth=full_res_depth,
            depth_edge_filter=depth_edge_filter,
            depth_edge_percentile=depth_edge_percentile,
            depth_edge_threshold=depth_edge_threshold,
            depth_edge_dilation=depth_edge_dilation,
            chunk_density_filter=chunk_density_filter,
            chunk_density_relative=chunk_density_relative,
            chunk_density_radius_voxels=chunk_density_radius_voxels,
            chunk_density_min_neighbors=chunk_density_min_neighbors,
        )
        return output_dir, fused_path

    device = resolve_device(device_name)
    load_start = perf_counter()
    model = load_model(
        ckpt_path,
        device,
        valid_iters=valid_iters,
        scale=scale,
        hiera=hiera,
    )
    print(f"[time] foundation_model_load={perf_counter() - load_start:.3f}s")

    rect_common_width, rect_common_height = rectification_metadata["common_crop_size"]
    rectified_depth_width, rectified_depth_height = scaled_size(rect_common_width, rect_common_height, target_short_edge)
    left_original_intrinsics_full = np.asarray(rectification_metadata["left_camera"]["intrinsics"], dtype=np.float32)
    right_original_intrinsics_full = np.asarray(rectification_metadata["right_camera"]["intrinsics"], dtype=np.float32)
    projection_left = np.asarray(rectification_metadata["rectification"]["projection_left"], dtype=np.float32)
    projection_right = np.asarray(rectification_metadata["rectification"]["projection_right"], dtype=np.float32)
    rectified_intrinsics_base = projection_left[:, :3]
    rectified_intrinsics = scale_intrinsics(
        rectified_intrinsics_base,
        rectified_depth_width,
        rectified_depth_height,
        rect_common_width,
        rect_common_height,
    )
    left_rectified_to_original = np.asarray(rectification_metadata["rectification"]["rotation_left"], dtype=np.float32).T
    right_rectified_to_original = np.asarray(rectification_metadata["rectification"]["rotation_right"], dtype=np.float32).T
    baseline = float(rectification_metadata["rig_validation"]["baseline_m"])
    fx_rectified = float(rectified_intrinsics[0, 0])
    metadata_payload = {
        "module": "FoundationStereo",
        "checkpoint": str(ckpt_path),
        "rectification_metadata": str(undistorted_dir / "stereo_rectified_images" / "rectification_metadata.json"),
        "pair_count": len(pairs),
        "rectified_inference_size": [rectified_depth_width, rectified_depth_height],
        "baseline_m": baseline,
        "rectified_fx": fx_rectified,
        "scale": scale,
        "valid_iters": valid_iters,
        "hiera": bool(hiera),
        "coherence_filter": bool(coherence_filter),
        "coherence_threshold": coherence_threshold,
        "coherence_dilation": coherence_dilation,
        "edge_filter": bool(depth_edge_filter),
        "edge_percentile": depth_edge_percentile,
        "edge_threshold": depth_edge_threshold,
        "edge_dilation": depth_edge_dilation,
    }
    save_depth_metadata(undistorted_dir / "foundation_stereo_metadata.json", metadata_payload)

    for pair in pairs:
        pair_start = perf_counter()
        left_rectified_rgb = load_rgb(pair.left_rectified_path, (rectified_depth_width, rectified_depth_height))
        right_rectified_rgb = load_rgb(pair.right_rectified_path, (rectified_depth_width, rectified_depth_height))
        left_disp, right_disp = run_pair_inference(
            model,
            device,
            left_rectified_rgb,
            right_rectified_rgb,
            valid_iters=valid_iters,
            scale=scale,
            hiera=hiera,
        )

        left_depth_rect = disparity_to_depth(left_disp, fx=fx_rectified, baseline=baseline, scale=scale)
        right_depth_rect = disparity_to_depth(right_disp, fx=fx_rectified, baseline=baseline, scale=scale)

        if depth_edge_filter:
            left_mask = inverse_depth_discontinuity_mask(
                left_depth_rect,
                percentile=depth_edge_percentile,
                fixed_threshold=depth_edge_threshold,
                dilation=depth_edge_dilation,
            )
            right_mask = inverse_depth_discontinuity_mask(
                right_depth_rect,
                percentile=depth_edge_percentile,
                fixed_threshold=depth_edge_threshold,
                dilation=depth_edge_dilation,
            )
            if left_mask.any():
                left_depth_rect = left_depth_rect.copy()
                left_depth_rect[left_mask] = 0.0
            if right_mask.any():
                right_depth_rect = right_depth_rect.copy()
                right_depth_rect[right_mask] = 0.0

        projected_right_rect = horizontal_project_opposite_depth(
            right_depth_rect,
            left_depth_rect,
            fx=fx_rectified,
            baseline=baseline,
            direction=-1,
        )
        projected_left_rect = horizontal_project_opposite_depth(
            left_depth_rect,
            right_depth_rect,
            fx=fx_rectified,
            baseline=baseline,
            direction=1,
        )
        if coherence_filter:
            left_final_rect, left_diff_rect, left_valid_rect = apply_inverse_depth_coherence(
                left_depth_rect,
                projected_right_rect,
                threshold=coherence_threshold,
                dilation=coherence_dilation,
            )
            right_final_rect, right_diff_rect, right_valid_rect = apply_inverse_depth_coherence(
                right_depth_rect,
                projected_left_rect,
                threshold=coherence_threshold,
                dilation=coherence_dilation,
            )
        else:
            left_final_rect = left_depth_rect
            right_final_rect = right_depth_rect
            left_diff_rect = np.zeros_like(left_depth_rect)
            right_diff_rect = np.zeros_like(right_depth_rect)
            left_valid_rect = np.zeros_like(left_depth_rect, dtype=bool)
            right_valid_rect = np.zeros_like(right_depth_rect, dtype=bool)

        left_width, left_height = item_depth_size(pair.left)
        right_width, right_height = item_depth_size(pair.right)
        left_original_intrinsics = scale_intrinsics(
            left_original_intrinsics_full,
            left_width,
            left_height,
            pair.left.camera.width,
            pair.left.camera.height,
        )
        right_original_intrinsics = scale_intrinsics(
            right_original_intrinsics_full,
            right_width,
            right_height,
            pair.right.camera.width,
            pair.right.camera.height,
        )
        left_independent = warp_rectified_depth_to_original(
            left_depth_rect,
            rectified_intrinsics,
            left_rectified_to_original,
            left_original_intrinsics,
            (left_width, left_height),
        )
        right_independent = warp_rectified_depth_to_original(
            right_depth_rect,
            rectified_intrinsics,
            right_rectified_to_original,
            right_original_intrinsics,
            (right_width, right_height),
        )
        left_projected = warp_rectified_depth_to_original(
            projected_right_rect,
            rectified_intrinsics,
            left_rectified_to_original,
            left_original_intrinsics,
            (left_width, left_height),
        )
        right_projected = warp_rectified_depth_to_original(
            projected_left_rect,
            rectified_intrinsics,
            right_rectified_to_original,
            right_original_intrinsics,
            (right_width, right_height),
        )
        left_final = warp_rectified_depth_to_original(
            left_final_rect,
            rectified_intrinsics,
            left_rectified_to_original,
            left_original_intrinsics,
            (left_width, left_height),
        )
        right_final = warp_rectified_depth_to_original(
            right_final_rect,
            rectified_intrinsics,
            right_rectified_to_original,
            right_original_intrinsics,
            (right_width, right_height),
        )

        save_depth_outputs(independent_dir, pair.left, "", left_independent)
        save_depth_outputs(independent_dir, pair.right, "", right_independent)
        save_final_depth_outputs(output_dir, pair.left, left_final)
        save_final_depth_outputs(output_dir, pair.right, right_final)

        if coherence_filter:
            save_depth_outputs(projected_dir, pair.left, "", left_projected)
            save_depth_outputs(projected_dir, pair.right, "", right_projected)
            left_compare_valid = (left_independent > 0) & (left_projected > 0)
            right_compare_valid = (right_independent > 0) & (right_projected > 0)
            left_diff = np.zeros_like(left_independent, dtype=np.float32)
            right_diff = np.zeros_like(right_independent, dtype=np.float32)
            if left_compare_valid.any():
                left_diff[left_compare_valid] = np.abs(
                    (1.0 / left_independent[left_compare_valid]) - (1.0 / left_projected[left_compare_valid])
                )
            if right_compare_valid.any():
                right_diff[right_compare_valid] = np.abs(
                    (1.0 / right_independent[right_compare_valid]) - (1.0 / right_projected[right_compare_valid])
                )
            write_diff_heatmap(
                diff_dir / f"diff_depth_{output_stem(pair.left.image)}.png",
                left_diff,
                left_compare_valid,
                coherence_threshold,
            )
            write_diff_heatmap(
                diff_dir / f"diff_depth_{output_stem(pair.right.image)}.png",
                right_diff,
                right_compare_valid,
                coherence_threshold,
            )

        left_rejected = int(np.count_nonzero((left_independent > 0) & (left_final <= 0)))
        right_rejected = int(np.count_nonzero((right_independent > 0) & (right_final <= 0)))
        print(
            f"[dense] pair={pair.basename} total={perf_counter() - pair_start:.3f}s "
            f"left_reject={left_rejected} right_reject={right_rejected} "
            f"left_valid={int(np.count_nonzero(left_final > 0))} "
            f"right_valid={int(np.count_nonzero(right_final > 0))}"
        )

    (output_dir / DENSE_MODE_MARKER).write_text(expected_marker + "\n", encoding="utf-8")
    print(f"[done] Dense depth output: {output_dir}")

    fused_path = run_fuse_pcd(
        source_folder,
        items,
        voxel_size=voxel_size,
        chunk_size=fusion_chunk_size,
        voxel_backend=voxel_backend,
        full_res_depth=full_res_depth,
        depth_edge_filter=depth_edge_filter,
        depth_edge_percentile=depth_edge_percentile,
        depth_edge_threshold=depth_edge_threshold,
        depth_edge_dilation=depth_edge_dilation,
        chunk_density_filter=chunk_density_filter,
        chunk_density_relative=chunk_density_relative,
        chunk_density_radius_voxels=chunk_density_radius_voxels,
        chunk_density_min_neighbors=chunk_density_min_neighbors,
    )
    return output_dir, fused_path
