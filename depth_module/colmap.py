from __future__ import annotations

import collections
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}


class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


@dataclass(frozen=True)
class SparseDepthItem:
    image: Image
    camera: Camera
    image_path: Path
    sparse_depth_path: Path
    depth_width: int | None = None
    depth_height: int | None = None


@dataclass(frozen=True)
class StereoPairItem:
    basename: str
    left: SparseDepthItem
    right: SparseDepthItem
    left_rectified_path: Path
    right_rectified_path: Path


def read_next_bytes(handle, num_bytes: int, format_char_sequence: str):
    if num_bytes == 0:
        return ()
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise RuntimeError("Unexpected end of COLMAP model file")
    return struct.unpack("<" + format_char_sequence, data)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = Camera(
                id=camera_id,
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.array(tuple(map(float, elems[4:])), dtype=np.float64),
            )
    return cameras


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as handle:
        num_cameras = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(handle, 24, "iiQQ")
            model = CAMERA_MODEL_IDS[model_id]
            params = read_next_bytes(handle, 8 * model.num_params, "d" * model.num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model.model_name,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_images_text(path: Path) -> dict[int, Image]:
    images = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_line = handle.readline().split()
            if point_line:
                xys = np.column_stack(
                    [tuple(map(float, point_line[0::3])), tuple(map(float, point_line[1::3]))]
                )
                point3d_ids = np.array(tuple(map(int, point_line[2::3])), dtype=np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3d_ids = np.empty((0,), dtype=np.int64)
            image_id = int(elems[0])
            images[image_id] = Image(
                id=image_id,
                qvec=np.array(tuple(map(float, elems[1:5])), dtype=np.float64),
                tvec=np.array(tuple(map(float, elems[5:8])), dtype=np.float64),
                camera_id=int(elems[8]),
                name=elems[9],
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_images_binary(path: Path) -> dict[int, Image]:
    images = {}
    with path.open("rb") as handle:
        num_images = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(handle, 64, "idddddddi")
            image_id = props[0]
            image_name = ""
            current_char = read_next_bytes(handle, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(handle, 1, "c")[0]
            num_points2d = read_next_bytes(handle, 8, "Q")[0]
            values = read_next_bytes(handle, 24 * num_points2d, "ddq" * num_points2d)
            if values:
                xys = np.column_stack(
                    [tuple(map(float, values[0::3])), tuple(map(float, values[1::3]))]
                )
                point3d_ids = np.array(tuple(map(int, values[2::3])), dtype=np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3d_ids = np.empty((0,), dtype=np.int64)
            images[image_id] = Image(
                id=image_id,
                qvec=np.array(props[1:5], dtype=np.float64),
                tvec=np.array(props[5:8], dtype=np.float64),
                camera_id=props[8],
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    points = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            points[point_id] = Point3D(
                id=point_id,
                xyz=np.array(tuple(map(float, elems[1:4])), dtype=np.float64),
                rgb=np.array(tuple(map(int, elems[4:7])), dtype=np.uint8),
                error=float(elems[7]),
                image_ids=np.array(tuple(map(int, elems[8::2])), dtype=np.int32),
                point2D_idxs=np.array(tuple(map(int, elems[9::2])), dtype=np.int32),
            )
    return points


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    points = {}
    with path.open("rb") as handle:
        num_points = read_next_bytes(handle, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(handle, 43, "QdddBBBd")
            track_length = read_next_bytes(handle, 8, "Q")[0]
            track = read_next_bytes(handle, 8 * track_length, "ii" * track_length)
            point_id = props[0]
            points[point_id] = Point3D(
                id=point_id,
                xyz=np.array(props[1:4], dtype=np.float64),
                rgb=np.array(props[4:7], dtype=np.uint8),
                error=float(props[7]),
                image_ids=np.array(tuple(map(int, track[0::2])), dtype=np.int32),
                point2D_idxs=np.array(tuple(map(int, track[1::2])), dtype=np.int32),
            )
    return points


def read_colmap_model(model_dir: Path):
    if (model_dir / "cameras.bin").is_file():
        return (
            read_cameras_binary(model_dir / "cameras.bin"),
            read_images_binary(model_dir / "images.bin"),
            read_points3d_binary(model_dir / "points3D.bin"),
        )
    if (model_dir / "cameras.txt").is_file():
        return (
            read_cameras_text(model_dir / "cameras.txt"),
            read_images_text(model_dir / "images.txt"),
            read_points3d_text(model_dir / "points3D.txt"),
        )
    raise RuntimeError(f"No COLMAP model found in {model_dir}")


def camera_intrinsics(camera: Camera) -> np.ndarray:
    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera.model in {"RADIAL", "RADIAL_FISHEYE", "OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise RuntimeError(f"Unsupported camera model for intrinsics: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def scale_intrinsics(intrinsics: np.ndarray, width: int, height: int, base_width: int, base_height: int) -> np.ndarray:
    scaled = intrinsics.copy().astype(np.float32)
    scaled[0, 0] *= width / base_width
    scaled[0, 2] *= width / base_width
    scaled[1, 1] *= height / base_height
    scaled[1, 2] *= height / base_height
    return scaled


def normalized_intrinsics(camera: Camera) -> np.ndarray:
    return normalized_intrinsics_for_size(camera, camera.width, camera.height)


def normalized_intrinsics_for_size(camera: Camera, width: int, height: int) -> np.ndarray:
    intrinsics = camera_intrinsics(camera)
    intrinsics[0, 0] *= width / camera.width
    intrinsics[0, 2] *= width / camera.width
    intrinsics[1, 1] *= height / camera.height
    intrinsics[1, 2] *= height / camera.height
    intrinsics[0, 0] /= width
    intrinsics[0, 2] /= width
    intrinsics[1, 1] /= height
    intrinsics[1, 2] /= height
    return intrinsics


def item_depth_size(item: SparseDepthItem) -> tuple[int, int]:
    return item.depth_width or item.camera.width, item.depth_height or item.camera.height


def sparse_depth_item_at_size(
    item: SparseDepthItem,
    sparse_depth_path: Path,
    width: int,
    height: int,
) -> SparseDepthItem:
    return SparseDepthItem(
        image=item.image,
        camera=item.camera,
        image_path=item.image_path,
        sparse_depth_path=sparse_depth_path,
        depth_width=width,
        depth_height=height,
    )


def scaled_size(width: int, height: int, short_edge: int) -> tuple[int, int]:
    if short_edge < 1:
        raise RuntimeError(f"Depth short edge must be >= 1: {short_edge}")
    current_short_edge = min(width, height)
    if current_short_edge <= short_edge:
        return width, height
    scale = short_edge / current_short_edge
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def select_stereo_pairs(
    source_folder: Path,
    *,
    max_pairs: int | None,
    image_stride: int,
    target_short_edge: int | None,
) -> tuple[list[StereoPairItem], dict[int, Camera]]:
    undistorted_dir = source_folder / "undistorted"
    model_dir = undistorted_dir / "sparse"
    images_dir = undistorted_dir / "images"
    sparse_depth_dir = undistorted_dir / "sparse_depth"
    if not model_dir.is_dir():
        raise RuntimeError(f"Missing COLMAP sparse model: {model_dir}")
    if not images_dir.is_dir():
        raise RuntimeError(f"Missing undistorted images folder: {images_dir}")

    cameras, images, _ = read_colmap_model(model_dir)
    left_images = {}
    right_images = {}
    for image in images.values():
        path = images_dir / image.name
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image.name.startswith("left/"):
            left_images[Path(image.name).name] = image
        elif image.name.startswith("right/"):
            right_images[Path(image.name).name] = image

    if set(left_images) != set(right_images):
        left_only = sorted(set(left_images) - set(right_images))
        right_only = sorted(set(right_images) - set(left_images))
        raise RuntimeError(
            "Stereo registration mismatch between left and right images: "
            f"left_only={left_only[:5]} right_only={right_only[:5]}"
        )

    basenames = sorted(left_images)[::image_stride]
    if max_pairs is not None:
        basenames = basenames[:max_pairs]
    pairs: list[StereoPairItem] = []
    for basename in basenames:
        left_image = left_images[basename]
        right_image = right_images[basename]
        left_path = images_dir / left_image.name
        right_path = images_dir / right_image.name
        left_item = SparseDepthItem(
            image=left_image,
            camera=cameras[left_image.camera_id],
            image_path=left_path,
            sparse_depth_path=Path(),
        )
        right_item = SparseDepthItem(
            image=right_image,
            camera=cameras[right_image.camera_id],
            image_path=right_path,
            sparse_depth_path=Path(),
        )
        if target_short_edge is not None:
            left_item = sparse_depth_item_at_size(
                left_item,
                Path(),
                *scaled_size(left_item.camera.width, left_item.camera.height, target_short_edge),
            )
            right_item = sparse_depth_item_at_size(
                right_item,
                Path(),
                *scaled_size(right_item.camera.width, right_item.camera.height, target_short_edge),
            )
        pairs.append(
            StereoPairItem(
                basename=basename,
                left=sparse_depth_item_at_size(
                    left_item, depth_paths(sparse_depth_dir, left_item.image)[0], *item_depth_size(left_item)
                ),
                right=sparse_depth_item_at_size(
                    right_item, depth_paths(sparse_depth_dir, right_item.image)[0], *item_depth_size(right_item)
                ),
                left_rectified_path=Path(),
                right_rectified_path=Path(),
            )
        )
    if not pairs:
        raise RuntimeError(f"No registered stereo pairs found in {images_dir}")
    return pairs, cameras


def flatten_pairs(pairs: list[StereoPairItem]) -> list[SparseDepthItem]:
    items: list[SparseDepthItem] = []
    for pair in pairs:
        items.append(pair.left)
        items.append(pair.right)
    return items


def output_stem(image: Image) -> str:
    stem = Path(image.name).with_suffix("").as_posix().replace("/", "__")
    return f"{image.id:08d}_{stem}"


def depth_paths(output_dir: Path, image: Image) -> tuple[Path, Path, Path]:
    stem = output_stem(image)
    return output_dir / f"{stem}.npy", output_dir / f"{stem}.png", output_dir / f"{stem}_vis.png"


def write_depth_png(path: Path, depth: np.ndarray) -> None:
    millimeters = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535)
    cv2.imwrite(str(path), millimeters.astype(np.uint16))


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        cv2.imwrite(str(path), np.zeros((*depth.shape, 3), dtype=np.uint8))
        return
    near = np.percentile(depth[valid], 1)
    far = np.percentile(depth[valid], 99)
    normalized = np.clip((depth - near) / (far - near + 1e-8) * 255, 0, 255)
    preview = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imwrite(str(path), preview)


def sparse_depth_is_complete(output_dir: Path, selected_items: list[SparseDepthItem]) -> bool:
    for item in selected_items:
        npy_path, png_path, preview_path = depth_paths(output_dir, item.image)
        if not npy_path.is_file() or not png_path.is_file() or not preview_path.is_file():
            return False
        width, height = item_depth_size(item)
        try:
            shape = np.load(npy_path, mmap_mode="r").shape
        except ValueError:
            return False
        if shape != (height, width):
            return False
    return True


def project_sparse_depth(image: Image, camera: Camera, points3d: dict[int, Point3D]) -> np.ndarray:
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    valid_track = image.point3D_ids >= 0
    if not valid_track.any():
        depth.fill(0.0)
        return depth

    point_ids = image.point3D_ids[valid_track]
    xys = image.xys[valid_track]
    xyz_world = []
    xy_pixels = []
    for point_id, xy in zip(point_ids, xys):
        point = points3d.get(int(point_id))
        if point is None:
            continue
        xyz_world.append(point.xyz)
        xy_pixels.append(xy)

    if not xyz_world:
        depth.fill(0.0)
        return depth

    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    xy_pixels = np.rint(np.asarray(xy_pixels, dtype=np.float64)).astype(np.int64)
    xyz_camera = xyz_world @ image.qvec2rotmat().T + image.tvec
    z = xyz_camera[:, 2].astype(np.float32)
    valid = (
        (z > 0)
        & (xy_pixels[:, 0] >= 0)
        & (xy_pixels[:, 0] < camera.width)
        & (xy_pixels[:, 1] >= 0)
        & (xy_pixels[:, 1] < camera.height)
    )
    pixels = xy_pixels[valid]
    z = z[valid]
    np.minimum.at(depth, (pixels[:, 1], pixels[:, 0]), z)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def prepare_sparse_depth(
    source_folder: Path,
    items: list[SparseDepthItem],
    *,
    overwrite: bool = False,
) -> None:
    undistorted_dir = source_folder / "undistorted"
    model_dir = undistorted_dir / "sparse"
    output_dir = undistorted_dir / "sparse_depth"
    if not model_dir.is_dir():
        raise RuntimeError(f"Missing COLMAP sparse model: {model_dir}")
    cameras, images, points3d = read_colmap_model(model_dir)
    _ = cameras, images
    if not overwrite and sparse_depth_is_complete(output_dir, items):
        print(f"[skip] Reusing existing sparse depth: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing sparse depth to: {output_dir}")
    for item in items:
        npy_path, png_path, preview_path = depth_paths(output_dir, item.image)
        depth = project_sparse_depth(item.image, item.camera, points3d)
        width, height = item_depth_size(item)
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        np.save(npy_path, depth)
        write_depth_png(png_path, depth)
        write_depth_preview(preview_path, depth)
        print(f"[sparse] {item.image.name}: {width}x{height} {int(np.count_nonzero(depth > 0))} pixels")
