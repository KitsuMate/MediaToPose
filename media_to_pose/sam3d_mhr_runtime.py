"""Release-safe NumPy runtime for the extracted SAM 3D Body MHR model.

The runtime consumes the deterministic artifacts written by :mod:`extract_mhr`
and has no PyTorch or PyMomentum dependency.  ``forward`` mirrors the scripted
``mhr_model.pt`` contract: vertices and skeleton translations are returned in
the model's native centimetres, quaternions use XYZW order, and scale is the
last skeleton channel.  ``MhrOutput.camera_space`` performs the separate
metres conversion and Y/Z axis flip used by the ViT-H MHR head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LBS_MAGIC = 0x4C425300
LBS_VERSION = 3
CORRECTIVES_MAGIC = 0x43524543
CORRECTIVES_VERSION = 1
HAND_RUNTIME_MAGIC = 0x48414E44
HAND_RUNTIME_VERSION = 1

N_JOINTS = 127
N_VERTICES = 18_439
N_SHAPE_COEFFICIENTS = 45
N_FACE_COEFFICIENTS = 72
N_MODEL_PARAMETERS = 204
N_PARAMETER_COLUMNS = 249
N_KEYPOINTS = 70


class MhrRuntimeError(RuntimeError):
    """An MHR artifact or runtime input violates the known ViT-H contract."""


@dataclass(frozen=True, slots=True)
class MhrCameraOutput:
    """Plugin-facing MHR geometry in metres with the camera Y/Z axis flip."""

    vertices: np.ndarray
    keypoints: np.ndarray
    joint_coords: np.ndarray
    joint_rotations: np.ndarray


@dataclass(frozen=True, slots=True)
class MhrOutput:
    """Raw output matching ``mhr_model.pt`` plus derived MHR fields.

    ``vertices``, ``keypoints``, and ``joint_coords`` are in centimetres.
    ``skeleton`` has the upstream layout ``[tx, ty, tz, qx, qy, qz, qw, s]``.
    """

    vertices: np.ndarray
    skeleton: np.ndarray
    keypoints: np.ndarray
    joint_coords: np.ndarray
    joint_quaternions: np.ndarray
    joint_scales: np.ndarray
    joint_rotations: np.ndarray

    def camera_space(self) -> MhrCameraOutput:
        """Return the coordinates used by the ViT-H head and Blender provider."""

        vertices = self.vertices * np.float32(0.01)
        keypoints = self.keypoints * np.float32(0.01)
        joint_coords = self.joint_coords * np.float32(0.01)
        vertices = vertices.copy()
        keypoints = keypoints.copy()
        joint_coords = joint_coords.copy()
        vertices[..., 1:3] *= np.float32(-1.0)
        keypoints[..., 1:3] *= np.float32(-1.0)
        joint_coords[..., 1:3] *= np.float32(-1.0)
        return MhrCameraOutput(
            vertices=vertices,
            keypoints=keypoints,
            joint_coords=joint_coords,
            # This intentionally matches MHRHead: coordinate arrays are flipped,
            # while its quaternion-derived global rotation matrices are not.
            joint_rotations=self.joint_rotations,
        )


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _array_view(
    raw: np.memmap,
    offset: int,
    dtype: str,
    shape: tuple[int, ...],
    label: str,
) -> tuple[np.ndarray, int]:
    item_dtype = np.dtype(dtype)
    byte_count = _product(shape) * item_dtype.itemsize
    end = offset + byte_count
    if end > raw.nbytes:
        raise MhrRuntimeError(
            f"{label} is truncated: need byte {end}, file has {raw.nbytes}"
        )
    array = np.ndarray(shape, dtype=item_dtype, buffer=raw, offset=offset, order="C")
    array.flags.writeable = False
    return array, end


def _open_mapping(path: Path, label: str) -> np.memmap:
    if not path.is_file():
        raise MhrRuntimeError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise MhrRuntimeError(f"Empty {label}: {path}")
    return np.memmap(path, mode="r", dtype=np.uint8)


def _as_batch(value: object, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise MhrRuntimeError(
            f"{label} must have shape [batch, {width}], got {array.shape}"
        )
    if array.shape[0] < 1:
        raise MhrRuntimeError(f"{label} batch must not be empty")
    if not np.isfinite(array).all():
        raise MhrRuntimeError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for broadcast-compatible XYZW arrays."""

    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            bx * aw + bw * ax + bz * ay - by * az,
            by * aw - bz * ax + bw * ay + bx * az,
            bz * aw + by * ax - bx * ay + bw * az,
            bw * aw - bx * ax - by * ay - bz * az,
        ),
        axis=-1,
    )


def _quaternion_rotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Rotate vectors by unit XYZW quaternions without forming matrices."""

    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    temporary = np.float32(2.0) * np.cross(q_xyz, value)
    return value + q_w * temporary + np.cross(q_xyz, temporary)


def _euler_xyz_to_quaternion(angles: np.ndarray) -> np.ndarray:
    """PyMomentum intrinsic XYZ Euler angles to XYZW quaternions."""

    half = angles * np.float32(0.5)
    sine = np.sin(half)
    cosine = np.cos(half)
    zeros = np.zeros_like(sine[..., 0])
    qx = np.stack((sine[..., 0], zeros, zeros, cosine[..., 0]), axis=-1)
    qy = np.stack((zeros, sine[..., 1], zeros, cosine[..., 1]), axis=-1)
    qz = np.stack((zeros, zeros, sine[..., 2], cosine[..., 2]), axis=-1)
    return _quaternion_multiply(_quaternion_multiply(qz, qy), qx)


def quaternion_xyzw_to_matrix(quaternions: object) -> np.ndarray:
    """Convert normalized XYZW quaternions to rotation matrices."""

    values = np.asarray(quaternions, dtype=np.float32)
    if values.shape[-1:] != (4,):
        raise MhrRuntimeError(
            f"quaternions must end in four XYZW values, got {values.shape}"
        )
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= np.float32(1e-12)) or not np.isfinite(values).all():
        raise MhrRuntimeError("quaternions contain a zero norm, NaN, or infinity")
    values = values / norm
    x, y, z, w = np.moveaxis(values, -1, 0)
    two = np.float32(2.0)
    return np.stack(
        (
            np.float32(1.0) - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            np.float32(1.0) - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            np.float32(1.0) - two * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*values.shape[:-1], 3, 3)


def _sparse_row_starts(rows: np.ndarray, output_rows: int, label: str):
    if rows.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.int32)
    if np.any(rows < 0) or np.any(rows >= output_rows):
        raise MhrRuntimeError(f"{label} has an out-of-range row index")
    if np.any(rows[1:] < rows[:-1]):
        raise MhrRuntimeError(f"{label} rows are not sorted")
    starts = np.flatnonzero(
        np.concatenate((np.ones(1, dtype=bool), rows[1:] != rows[:-1]))
    )
    return starts, np.asarray(rows[starts], dtype=np.int32)


class MhrModel:
    """Memory-mapped, batch-capable NumPy implementation of ViT-H MHR."""

    def __init__(
        self,
        *,
        lbs_map: np.memmap,
        correctives_map: np.memmap,
        keypoint_map: np.memmap,
        hand_runtime_map: np.memmap,
        arrays: dict[str, np.ndarray],
        correctives: dict[str, np.ndarray | int],
        keypoints: dict[str, np.ndarray | int],
        hand_runtime: dict[str, np.ndarray],
    ) -> None:
        # Retain the mappings because all read-only array views use their buffers.
        self._lbs_map = lbs_map
        self._correctives_map = correctives_map
        self._keypoint_map = keypoint_map
        self._hand_runtime_map = hand_runtime_map
        self._arrays = arrays
        self._correctives = correctives
        self._keypoints = keypoints
        self._hand_runtime = hand_runtime

        self._corr1_starts, self._corr1_rows = _sparse_row_starts(
            correctives["rows1"], int(correctives["hidden"]), "corrective layer 1"
        )
        self._corr2_starts, self._corr2_rows = _sparse_row_starts(
            correctives["rows2"], int(correctives["output"]), "corrective layer 2"
        )
        self._keypoint_starts, self._keypoint_rows = _sparse_row_starts(
            keypoints["rows"], int(keypoints["output"]), "keypoint mapping"
        )

    def close(self) -> None:
        """Release every memory mapping owned by this model.

        Array views retain their mapping until the view dictionaries are
        cleared, so drop those references before closing the underlying mmap
        handles.  The method is deliberately idempotent because estimator
        construction and accelerator fallback both use it for cleanup.
        """

        mappings = tuple(
            mapping
            for mapping in (
                getattr(self, "_lbs_map", None),
                getattr(self, "_correctives_map", None),
                getattr(self, "_keypoint_map", None),
                getattr(self, "_hand_runtime_map", None),
            )
            if mapping is not None
        )
        self._arrays = {}
        self._correctives = {}
        self._keypoints = {}
        self._hand_runtime = {}
        self._lbs_map = None
        self._correctives_map = None
        self._keypoint_map = None
        self._hand_runtime_map = None
        for mapping in mappings:
            handle = getattr(mapping, "_mmap", None)
            if handle is not None and not handle.closed:
                handle.close()

    @classmethod
    def from_directory(cls, directory: str | Path) -> MhrModel:
        """Load the complete immutable MHR and hand-fusion artifact set."""

        root = Path(directory)
        return cls.load(
            root / "body_model.lbs",
            root / "correctives.bin",
            root / "keypoint_mapping.bin",
            root / "hand_runtime.bin",
        )

    @classmethod
    def load(
        cls,
        lbs_path: str | Path,
        correctives_path: str | Path,
        keypoint_mapping_path: str | Path,
        hand_runtime_path: str | Path,
    ) -> MhrModel:
        """Load and strictly validate the four extracted runtime artifacts."""

        lbs_map, arrays = cls._load_lbs(Path(lbs_path))
        correctives_map, correctives = cls._load_correctives(Path(correctives_path))
        keypoint_map, keypoints = cls._load_keypoints(Path(keypoint_mapping_path))
        hand_runtime_map, hand_runtime = cls._load_hand_runtime(Path(hand_runtime_path))
        return cls(
            lbs_map=lbs_map,
            correctives_map=correctives_map,
            keypoint_map=keypoint_map,
            hand_runtime_map=hand_runtime_map,
            arrays=arrays,
            correctives=correctives,
            keypoints=keypoints,
            hand_runtime=hand_runtime,
        )

    @staticmethod
    def _load_lbs(path: Path) -> tuple[np.memmap, dict[str, np.ndarray]]:
        raw = _open_mapping(path, "MHR LBS artifact")
        if raw.nbytes < 32:
            raise MhrRuntimeError(f"MHR LBS header is truncated: {path}")
        header = np.frombuffer(raw, dtype="<u4", count=8)
        magic, version, joints, skin, vertices, shapes, faces, columns = map(
            int, header
        )
        expected = (
            LBS_MAGIC,
            LBS_VERSION,
            N_JOINTS,
            51_337,
            N_VERTICES,
            N_SHAPE_COEFFICIENTS,
            N_FACE_COEFFICIENTS,
            N_PARAMETER_COLUMNS,
        )
        actual = (magic, version, joints, skin, vertices, shapes, faces, columns)
        if actual != expected:
            raise MhrRuntimeError(
                f"Unsupported MHR LBS header in {path}: got {actual}, expected {expected}"
            )

        offset = 32
        specs = (
            ("parameter_transform", "<f4", (joints * 7, columns)),
            ("joint_offsets", "<f4", (joints, 3)),
            ("joint_prerotations", "<f4", (joints, 4)),
            ("joint_parents", "<i4", (joints,)),
            ("inverse_bind_pose", "<f4", (joints, 8)),
            ("skin_joint_indices", "<i4", (skin,)),
            ("skin_weights", "<f4", (skin,)),
            ("skin_vertex_indices", "<i4", (skin,)),
            ("base_shape", "<f4", (vertices, 3)),
            ("shape_vectors", "<f4", (shapes, vertices, 3)),
            ("face_vectors", "<f4", (faces, vertices, 3)),
            ("scale_mean", "<f4", (68,)),
            ("scale_components", "<f4", (28, 68)),
            ("hand_pose_mean", "<f4", (54,)),
            ("hand_pose_components", "<f4", (54, 54)),
            ("hand_joint_indices_left", "<i4", (27,)),
            ("hand_joint_indices_right", "<i4", (27,)),
        )
        arrays: dict[str, np.ndarray] = {}
        for name, dtype, shape in specs:
            arrays[name], offset = _array_view(raw, offset, dtype, shape, name)
        if offset != raw.nbytes:
            raise MhrRuntimeError(
                f"MHR LBS artifact has {raw.nbytes - offset} unexpected trailing bytes"
            )
        parents = arrays["joint_parents"]
        if int(parents[0]) >= 0 or np.any(parents[1:] >= np.arange(1, joints)):
            raise MhrRuntimeError("MHR joint parents are not topologically ordered")
        if np.any(arrays["skin_joint_indices"] < 0) or np.any(
            arrays["skin_joint_indices"] >= joints
        ):
            raise MhrRuntimeError("MHR skin data has an out-of-range joint")
        if np.any(arrays["skin_vertex_indices"] < 0) or np.any(
            arrays["skin_vertex_indices"] >= vertices
        ):
            raise MhrRuntimeError("MHR skin data has an out-of-range vertex")
        return raw, arrays

    @staticmethod
    def _load_correctives(path: Path):
        raw = _open_mapping(path, "MHR correctives artifact")
        if raw.nbytes < 36:
            raise MhrRuntimeError(f"MHR correctives header is truncated: {path}")
        magic, version = map(int, np.frombuffer(raw, dtype="<u4", count=2))
        header = tuple(map(int, np.frombuffer(raw, dtype="<i4", count=7, offset=8)))
        joints, vertices, features, hidden, output, nnz1, nnz2 = header
        expected_prefix = (N_JOINTS, N_VERTICES, 750, 3000, N_VERTICES * 3)
        if (
            magic != CORRECTIVES_MAGIC
            or version != CORRECTIVES_VERSION
            or header[:5] != expected_prefix
            or nnz1 < 0
            or nnz2 < 0
        ):
            raise MhrRuntimeError(
                f"Unsupported MHR correctives header in {path}: "
                f"magic={magic:#x}, version={version}, values={header}"
            )
        offset = 36
        values: dict[str, np.ndarray | int] = {
            "features": features,
            "hidden": hidden,
            "output": output,
        }
        for name, dtype, shape in (
            ("rows1", "<i4", (nnz1,)),
            ("columns1", "<i4", (nnz1,)),
            ("values1", "<f4", (nnz1,)),
            ("rows2", "<i4", (nnz2,)),
            ("columns2", "<i4", (nnz2,)),
            ("values2", "<f4", (nnz2,)),
        ):
            values[name], offset = _array_view(raw, offset, dtype, shape, name)
        if offset != raw.nbytes:
            raise MhrRuntimeError(
                f"MHR correctives artifact has {raw.nbytes - offset} trailing bytes"
            )
        if np.any(values["columns1"] < 0) or np.any(values["columns1"] >= features):
            raise MhrRuntimeError("Corrective layer 1 has an out-of-range column")
        if np.any(values["columns2"] < 0) or np.any(values["columns2"] >= hidden):
            raise MhrRuntimeError("Corrective layer 2 has an out-of-range column")
        return raw, values

    @staticmethod
    def _load_keypoints(path: Path):
        raw = _open_mapping(path, "MHR keypoint mapping")
        if raw.nbytes < 12:
            raise MhrRuntimeError(f"MHR keypoint header is truncated: {path}")
        output, columns, nnz = map(int, np.frombuffer(raw, dtype="<u4", count=3))
        if output != N_KEYPOINTS or columns != N_VERTICES + N_JOINTS:
            raise MhrRuntimeError(
                f"Unsupported MHR keypoint shape {(output, columns)} in {path}"
            )
        entry_dtype = np.dtype([("row", "<i4"), ("column", "<i4"), ("value", "<f4")])
        entries, end = _array_view(raw, 12, entry_dtype, (nnz,), "keypoints")
        if end != raw.nbytes:
            raise MhrRuntimeError(
                f"MHR keypoint mapping has {raw.nbytes - end} trailing bytes"
            )
        rows = entries["row"]
        column_values = entries["column"]
        if np.any(column_values < 0) or np.any(column_values >= columns):
            raise MhrRuntimeError("MHR keypoint mapping has an out-of-range column")
        return raw, {
            "output": output,
            "columns_count": columns,
            "rows": rows,
            "columns": column_values,
            "values": entries["value"],
        }

    @staticmethod
    def _load_hand_runtime(path: Path):
        raw = _open_mapping(path, "MHR hand runtime artifact")
        if raw.nbytes < 16:
            raise MhrRuntimeError(f"MHR hand runtime header is truncated: {path}")
        magic, version, joints, nonhand_count = map(
            int, np.frombuffer(raw, dtype="<u4", count=4)
        )
        expected = (HAND_RUNTIME_MAGIC, HAND_RUNTIME_VERSION, N_JOINTS, 145)
        actual = (magic, version, joints, nonhand_count)
        if actual != expected:
            raise MhrRuntimeError(
                f"Unsupported MHR hand runtime header in {path}: "
                f"got {actual}, expected {expected}"
            )
        offset = 16
        arrays: dict[str, np.ndarray] = {}
        for name, dtype, shape in (
            ("joint_rotation", "<f4", (joints, 3, 3)),
            ("right_wrist_coords", "<f4", (3,)),
            ("root_coords", "<f4", (3,)),
            ("local_to_world_wrist", "<f4", (3, 3)),
            ("nonhand_param_indices", "<i4", (nonhand_count,)),
        ):
            arrays[name], offset = _array_view(raw, offset, dtype, shape, name)
        if offset != raw.nbytes:
            raise MhrRuntimeError(
                "MHR hand runtime artifact has "
                f"{raw.nbytes - offset} unexpected trailing bytes"
            )
        indices = arrays["nonhand_param_indices"]
        if (
            np.any(indices < 0)
            or np.any(indices >= N_MODEL_PARAMETERS)
            or len(np.unique(indices)) != len(indices)
        ):
            raise MhrRuntimeError(
                "MHR hand runtime contains invalid non-hand parameter indices"
            )
        for name in (
            "joint_rotation",
            "right_wrist_coords",
            "root_coords",
            "local_to_world_wrist",
        ):
            if not np.isfinite(arrays[name]).all():
                raise MhrRuntimeError(f"MHR hand runtime {name} is not finite")
        return raw, arrays

    @property
    def scale_mean(self) -> np.ndarray:
        """The extracted ViT-H scale PCA mean (read-only)."""

        return self._arrays["scale_mean"]

    @property
    def scale_components(self) -> np.ndarray:
        """The extracted ViT-H scale PCA components (read-only)."""

        return self._arrays["scale_components"]

    @property
    def hand_pose_mean(self) -> np.ndarray:
        """The extracted ViT-H hand-pose PCA mean (read-only)."""

        return self._arrays["hand_pose_mean"]

    @property
    def hand_pose_components(self) -> np.ndarray:
        """The extracted ViT-H hand-pose PCA components (read-only)."""

        return self._arrays["hand_pose_components"]

    @property
    def hand_joint_indices_left(self) -> np.ndarray:
        return self._arrays["hand_joint_indices_left"]

    @property
    def hand_joint_indices_right(self) -> np.ndarray:
        return self._arrays["hand_joint_indices_right"]

    @property
    def joint_rotation(self) -> np.ndarray:
        """Body bind-pose joint frames used by wrist IK (read-only)."""

        return self._hand_runtime["joint_rotation"]

    @property
    def right_wrist_coords(self) -> np.ndarray:
        return self._hand_runtime["right_wrist_coords"]

    @property
    def root_coords(self) -> np.ndarray:
        return self._hand_runtime["root_coords"]

    @property
    def local_to_world_wrist(self) -> np.ndarray:
        return self._hand_runtime["local_to_world_wrist"]

    @property
    def nonhand_param_indices(self) -> np.ndarray:
        return self._hand_runtime["nonhand_param_indices"]

    @staticmethod
    def _sparse_linear(
        inputs: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
        values: np.ndarray,
        starts: np.ndarray,
        unique_rows: np.ndarray,
        output_width: int,
    ) -> np.ndarray:
        output = np.zeros((inputs.shape[0], output_width), dtype=np.float32)
        if rows.size == 0:
            return output
        weighted = inputs[:, columns] * values[None, :]
        output[:, unique_rows] = np.add.reduceat(weighted, starts, axis=1)
        return output

    def _pose_correctives(self, joint_parameters: np.ndarray) -> np.ndarray:
        angles = joint_parameters[:, 2:, 3:6]
        x, y, z = np.moveaxis(angles, -1, 0)
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)
        features = np.stack(
            (
                cz * cy - np.float32(1.0),
                sz * cy,
                -sy,
                -sz * cx + cz * sy * sx,
                cz * cx + sz * sy * sx - np.float32(1.0),
                cy * sx,
            ),
            axis=-1,
        ).reshape(joint_parameters.shape[0], -1)
        corr = self._correctives
        hidden = self._sparse_linear(
            features,
            corr["rows1"],
            corr["columns1"],
            corr["values1"],
            self._corr1_starts,
            self._corr1_rows,
            int(corr["hidden"]),
        )
        np.maximum(hidden, np.float32(0.0), out=hidden)
        return self._sparse_linear(
            hidden,
            corr["rows2"],
            corr["columns2"],
            corr["values2"],
            self._corr2_starts,
            self._corr2_rows,
            int(corr["output"]),
        ).reshape(joint_parameters.shape[0], N_VERTICES, 3)

    def _extract_keypoints(
        self, vertices: np.ndarray, joint_coords: np.ndarray
    ) -> np.ndarray:
        mapping = self._keypoints
        model_points = np.concatenate((vertices, joint_coords), axis=1)
        return (
            self._sparse_linear(
                model_points.transpose(0, 2, 1).reshape(-1, model_points.shape[1]),
                mapping["rows"],
                mapping["columns"],
                mapping["values"],
                self._keypoint_starts,
                self._keypoint_rows,
                int(mapping["output"]),
            )
            .reshape(vertices.shape[0], 3, N_KEYPOINTS)
            .transpose(0, 2, 1)
        )

    def forward(
        self,
        identity_coeffs: object,
        model_parameters: object,
        face_expression_coeffs: object,
        *,
        apply_correctives: bool = True,
    ) -> MhrOutput:
        """Evaluate MHR for a batch with the same inputs as ``mhr_model.pt``."""

        identity = _as_batch(identity_coeffs, N_SHAPE_COEFFICIENTS, "identity_coeffs")
        parameters = _as_batch(model_parameters, N_MODEL_PARAMETERS, "model_parameters")
        expressions = _as_batch(
            face_expression_coeffs,
            N_FACE_COEFFICIENTS,
            "face_expression_coeffs",
        )
        batch = identity.shape[0]
        if parameters.shape[0] != batch or expressions.shape[0] != batch:
            raise MhrRuntimeError(
                "identity, model-parameter, and expression batches must match"
            )

        arrays = self._arrays
        unposed = arrays["base_shape"][None, :, :].copy()
        unposed = unposed + np.einsum(
            "bi,ivc->bvc", identity, arrays["shape_vectors"], optimize=True
        )
        unposed = unposed + np.einsum(
            "bi,ivc->bvc", expressions, arrays["face_vectors"], optimize=True
        )

        input_vector = np.zeros((batch, N_PARAMETER_COLUMNS), dtype=np.float32)
        input_vector[:, :N_MODEL_PARAMETERS] = parameters
        joint_parameters = (input_vector @ arrays["parameter_transform"].T).reshape(
            batch, N_JOINTS, 7
        )
        if apply_correctives:
            unposed += self._pose_correctives(joint_parameters)

        translation_local = (
            arrays["joint_offsets"][None, :, :] + joint_parameters[..., :3]
        )
        rotation_local = _quaternion_multiply(
            arrays["joint_prerotations"][None, :, :],
            _euler_xyz_to_quaternion(joint_parameters[..., 3:6]),
        )
        scale_local = np.exp(joint_parameters[..., 6] * np.float32(0.6931471824645996))

        translation_global = np.empty((batch, N_JOINTS, 3), dtype=np.float32)
        rotation_global = np.empty((batch, N_JOINTS, 4), dtype=np.float32)
        scale_global = np.empty((batch, N_JOINTS), dtype=np.float32)
        parents = arrays["joint_parents"]
        for joint in range(N_JOINTS):
            parent = int(parents[joint])
            if parent < 0:
                translation_global[:, joint] = translation_local[:, joint]
                rotation_global[:, joint] = rotation_local[:, joint]
                scale_global[:, joint] = scale_local[:, joint]
            else:
                scale_global[:, joint] = scale_global[:, parent] * scale_local[:, joint]
                rotation_global[:, joint] = _quaternion_multiply(
                    rotation_global[:, parent], rotation_local[:, joint]
                )
                translation_global[:, joint] = translation_global[:, parent] + (
                    scale_global[:, parent, None]
                    * _quaternion_rotate(
                        rotation_global[:, parent], translation_local[:, joint]
                    )
                )

        inverse_bind = arrays["inverse_bind_pose"]
        skin_scale = scale_global * inverse_bind[None, :, 7]
        skin_rotation = _quaternion_multiply(
            rotation_global, inverse_bind[None, :, 3:7]
        )
        skin_translation = translation_global + scale_global[..., None] * (
            _quaternion_rotate(rotation_global, inverse_bind[None, :, :3])
        )

        joint_indices = arrays["skin_joint_indices"]
        vertex_indices = arrays["skin_vertex_indices"]
        rotated = _quaternion_rotate(
            skin_rotation[:, joint_indices], unposed[:, vertex_indices]
        )
        contributions = arrays["skin_weights"][None, :, None] * (
            skin_translation[:, joint_indices]
            + skin_scale[:, joint_indices, None] * rotated
        )
        vertices = np.zeros((batch, N_VERTICES, 3), dtype=np.float32)
        for batch_index in range(batch):
            np.add.at(vertices[batch_index], vertex_indices, contributions[batch_index])

        skeleton = np.concatenate(
            (
                translation_global,
                rotation_global,
                scale_global[..., None],
            ),
            axis=-1,
        )
        keypoints = self._extract_keypoints(vertices, translation_global)
        rotations = quaternion_xyzw_to_matrix(rotation_global)
        for name, value in (
            ("vertices", vertices),
            ("skeleton", skeleton),
            ("keypoints", keypoints),
            ("joint rotations", rotations),
        ):
            if not np.isfinite(value).all():
                raise MhrRuntimeError(f"MHR {name} contains NaN or infinity")
        return MhrOutput(
            vertices=vertices,
            skeleton=skeleton,
            keypoints=keypoints,
            joint_coords=translation_global,
            joint_quaternions=rotation_global,
            joint_scales=scale_global,
            joint_rotations=rotations,
        )


__all__ = [
    "MhrCameraOutput",
    "MhrModel",
    "MhrOutput",
    "MhrRuntimeError",
    "quaternion_xyzw_to_matrix",
]
