"""Provider-neutral camera-intrinsics helpers and AnyCalib aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from ..types import CameraIntrinsicsObservation


def _fovs(width: int, height: int, fx: float, fy: float) -> tuple[float, float]:
    return (
        math.degrees(2.0 * math.atan(float(width) / (2.0 * float(fx)))),
        math.degrees(2.0 * math.atan(float(height) / (2.0 * float(fy)))),
    )


def intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    confidence: float = 1.0,
    source: str,
    bundle_revision: str = "",
    precision: str = "",
) -> CameraIntrinsicsObservation:
    hfov, vfov = _fovs(width, height, fx, fy)
    return CameraIntrinsicsObservation(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        horizontal_fov=hfov,
        vertical_fov=vfov,
        confidence=float(confidence),
        source=source,
        bundle_revision=bundle_revision,
        precision=precision,
    )


def fallback(width: int, height: int) -> CameraIntrinsicsObservation:
    focal = math.hypot(int(width), int(height))
    return intrinsics(
        width,
        height,
        focal,
        focal,
        float(width) * 0.5,
        float(height) * 0.5,
        confidence=0.25,
        source="FALLBACK",
    )


def manual(width: int, height: int, vertical_fov: float) -> CameraIntrinsicsObservation:
    value = float(vertical_fov)
    if not math.isfinite(value) or not 1.0 <= value <= 179.0:
        raise ValueError("Manual vertical FOV must be between 1 and 179 degrees")
    focal = float(height) * 0.5 / math.tan(math.radians(value) * 0.5)
    return intrinsics(
        width,
        height,
        focal,
        focal,
        float(width) * 0.5,
        float(height) * 0.5,
        source="MANUAL",
    )


def from_mapping(value: Mapping[str, object]) -> CameraIntrinsicsObservation:
    width = int(value["width"])
    height = int(value["height"])
    return intrinsics(
        width,
        height,
        float(value["fx"]),
        float(value["fy"]),
        float(value["cx"]),
        float(value["cy"]),
        confidence=float(value.get("confidence", 1.0)),
        source=str(value.get("source", "FALLBACK")),
        bundle_revision=str(value.get("bundle_revision", "")),
        precision=str(value.get("precision", "")),
    )


def to_mapping(value: CameraIntrinsicsObservation) -> dict[str, object]:
    return {
        "width": int(value.width),
        "height": int(value.height),
        "fx": float(value.fx),
        "fy": float(value.fy),
        "cx": float(value.cx),
        "cy": float(value.cy),
        "horizontal_fov": float(value.horizontal_fov),
        "vertical_fov": float(value.vertical_fov),
        "confidence": float(value.confidence),
        "source": str(value.source),
        "bundle_revision": str(value.bundle_revision),
        "precision": str(value.precision),
        "valid": bool(value.valid),
    }


def scale_to(
    value: CameraIntrinsicsObservation, width: int, height: int
) -> CameraIntrinsicsObservation:
    sx = float(width) / float(value.width)
    sy = float(height) / float(value.height)
    return intrinsics(
        width,
        height,
        value.fx * sx,
        value.fy * sy,
        value.cx * sx,
        value.cy * sy,
        confidence=value.confidence,
        source=value.source,
        bundle_revision=value.bundle_revision,
        precision=value.precision,
    )


def mirrored(value: CameraIntrinsicsObservation) -> CameraIntrinsicsObservation:
    return intrinsics(
        value.width,
        value.height,
        value.fx,
        value.fy,
        # Intrinsics use pixel-center coordinates (0.5 .. width - 0.5), so a
        # horizontal image flip maps x to width - x.
        float(value.width) - value.cx,
        value.cy,
        confidence=value.confidence,
        source=value.source,
        bundle_revision=value.bundle_revision,
        precision=value.precision,
    )


def matrix(value: CameraIntrinsicsObservation, np):
    return np.asarray(
        ((value.fx, 0.0, value.cx), (0.0, value.fy, value.cy), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )


def consensus(
    values: Iterable[CameraIntrinsicsObservation],
    *,
    minimum: int = 1,
    maximum_fov_range: float = 8.0,
) -> CameraIntrinsicsObservation | None:
    accepted = [value for value in values if value.valid and value.source == "ANYCALIB"]
    if len(accepted) < int(minimum):
        return None
    width, height = accepted[0].width, accepted[0].height
    accepted = [
        scale_to(value, width, height)
        for value in accepted
        if value.width > 0 and value.height > 0
    ]
    vfovs = sorted(value.vertical_fov for value in accepted)
    if not vfovs or vfovs[-1] - vfovs[0] > float(maximum_fov_range):
        return None
    ordered = sorted(accepted, key=lambda value: value.vertical_fov)
    middle = ordered[len(ordered) // 2]

    def median(items):
        sorted_items = sorted(float(item) for item in items)
        midpoint = len(sorted_items) // 2
        if len(sorted_items) % 2:
            return sorted_items[midpoint]
        return 0.5 * (sorted_items[midpoint - 1] + sorted_items[midpoint])

    fields = ("fx", "fy", "cx", "cy", "confidence")
    medians = {
        name: median(getattr(item, name) for item in accepted) for name in fields
    }
    return intrinsics(
        width,
        height,
        medians["fx"],
        medians["fy"],
        medians["cx"],
        medians["cy"],
        confidence=medians["confidence"],
        source="ANYCALIB",
        bundle_revision=middle.bundle_revision,
        precision=middle.precision,
    )


def image_consensus(
    values: Sequence[CameraIntrinsicsObservation],
    *,
    maximum_primary_deviation: float = 1.5,
) -> CameraIntrinsicsObservation | None:
    """Use transformed views only when they corroborate the unmodified image."""

    accepted = [value for value in values if value.valid and value.source == "ANYCALIB"]
    if not accepted:
        return None
    primary = accepted[0]
    stable = [
        value
        for value in accepted
        if abs(value.vertical_fov - primary.vertical_fov)
        <= float(maximum_primary_deviation)
        and abs(value.horizontal_fov - primary.horizontal_fov)
        <= float(maximum_primary_deviation)
    ]
    if len(stable) < 2:
        return primary
    return consensus(
        stable,
        minimum=2,
        maximum_fov_range=2.0 * float(maximum_primary_deviation),
    )


def fit_tangent_field(
    tangent_field,
    *,
    original_width: int,
    original_height: int,
    scale_xy: Sequence[float] = (1.0, 1.0),
    shift_xy: Sequence[float] = (0.0, 0.0),
    np,
    bundle_revision: str = "",
    precision: str = "",
) -> CameraIntrinsicsObservation:
    """Fit pinhole intrinsics to AnyCalib's dense tangent-coordinate field."""

    field = np.asarray(tangent_field, dtype=np.float64)
    if field.ndim == 4:
        field = field[0]
    if field.ndim == 3 and field.shape[0] == 2:
        field = field.transpose(1, 2, 0)
    if field.ndim != 3 or field.shape[-1] != 2 or not np.isfinite(field).all():
        raise ValueError(f"Invalid AnyCalib tangent field shape: {field.shape}")
    height, width = field.shape[:2]
    # AnyCalib predicts logarithmic tangent coordinates at the optical axis.
    # Convert them to the pinhole plane X/Z,Y/Z before the linear fit.
    angles = np.linalg.norm(field, axis=-1, keepdims=True)
    tangent_scale = np.where(
        angles > 1.0e-9,
        np.tan(angles) / np.maximum(angles, 1.0e-9),
        1.0,
    )
    projected = field * tangent_scale
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
        indexing="xy",
    )

    def solve(pixel, tangent):
        # Match Pinhole.fit: the dense prediction is already the robust observation,
        # so the official initializer uses every ray within the admissible 170° FoV.
        active = np.isfinite(tangent) & (angles.reshape(-1) < math.radians(85.0))
        design = np.stack((tangent[active], np.ones(active.sum())), axis=1)
        result, _residuals, rank, _singular = np.linalg.lstsq(
            design, pixel[active], rcond=None
        )
        if rank < 2 or not np.isfinite(result).all():
            raise ValueError("AnyCalib pinhole fit was singular")
        residual = pixel - (result[0] * tangent + result[1])
        return (
            float(result[0]),
            float(result[1]),
            float(np.sqrt(np.mean(np.square(residual[active])))),
        )

    fx, cx, error_x = solve(grid_x.reshape(-1), projected[..., 0].reshape(-1))
    fy, cy, error_y = solve(grid_y.reshape(-1), projected[..., 1].reshape(-1))

    # Preserve AnyCalib's default ten-step Gauss-Newton refinement. Residuals are
    # measured in each currently fitted ray's tangent plane, not in pixels.
    observed = np.concatenate(
        (
            field
            * np.where(
                angles > 1.0e-9,
                np.sin(angles) / np.maximum(angles, 1.0e-9),
                1.0,
            ),
            np.cos(angles),
        ),
        axis=-1,
    ).reshape(-1, 3)
    pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)

    def tangent_basis(unit_vectors):
        x0, x1, x2 = unit_vectors.T
        pole = np.abs(x1) >= 1.0 - np.finfo(np.float64).eps
        parallel = np.stack(
            (np.where(pole, 1.0, x2), np.zeros_like(x0), np.where(pole, 0.0, -x0)),
            axis=-1,
        )
        parallel /= np.maximum(
            np.linalg.norm(parallel, axis=-1, keepdims=True), 1.0e-15
        )
        meridian = np.cross(unit_vectors, parallel)
        return np.stack((parallel, meridian), axis=-1)

    def fitted_residual_and_jac(params):
        focal = params[:2]
        center = params[2:]
        plane = (pixels - center) / focal
        homogeneous = np.concatenate(
            (plane, np.ones((plane.shape[0], 1), dtype=np.float64)), axis=-1
        )
        fitted = homogeneous / np.linalg.norm(homogeneous, axis=-1, keepdims=True)
        basis = tangent_basis(fitted)
        cosine = np.sum(fitted * observed, axis=-1, keepdims=True)
        not_parallel = cosine < 1.0 - 1.0e-4
        distance = 2.0 * np.arcsin(
            np.clip(
                0.5 * np.linalg.norm(fitted - observed, axis=-1, keepdims=True),
                0.0,
                1.0 - np.finfo(np.float64).eps,
            )
        )
        theta = np.where(not_parallel, distance, 1.0)
        scaled_observed = np.where(
            not_parallel,
            theta / np.sin(theta) * observed,
            observed,
        )
        residual = np.sum(basis * scaled_observed[..., None], axis=-2)

        # Unit3.jac_logmap_wrt_refvecs, ported directly from upstream.
        basis_t = np.swapaxes(basis, -2, -1)
        by_xyt = (
            np.sum(basis_t * observed[:, None, :], axis=-1, keepdims=True)
            * (fitted - observed)[:, None, :]
        )
        cosine_j = cosine[:, None, :]
        not_parallel_j = not_parallel[:, None, :]
        guarded_cosine = np.where(not_parallel_j, cosine_j, 0.0)
        cosec2 = 1.0 / (1.0 - guarded_cosine**2)
        theta_cosec = distance[:, None, :] * np.sqrt(cosec2)
        term1_scale = np.where(
            not_parallel_j,
            cosec2 * (1.0 - guarded_cosine * theta_cosec),
            1.0 / 3.0,
        )
        term1 = term1_scale * by_xyt

        x0, x1, x2 = fitted.T
        y0, y1, y2 = observed.T
        inverse_norm_cubed = 1.0 / np.maximum(basis_t[:, 1, 1] ** 3, 1.0e-15)
        valid = inverse_norm_cubed < 1.0 / np.finfo(np.float64).eps
        term2 = np.zeros_like(basis_t)
        xy0_xy2 = x0 * y0 + x2 * y2
        x012 = x0 * x1 * x2
        term2[:, 0, 0] = -x2 * xy0_xy2 * inverse_norm_cubed
        term2[:, 0, 2] = x0 * xy0_xy2 * inverse_norm_cubed
        term2[:, 1, 0] = (
            x0**3 * y1 + x012 * y2 + x0 * x2**2 * y1 - x1 * x2**2 * y0
        ) * inverse_norm_cubed
        term2[:, 1, 1] = -xy0_xy2 / np.maximum(basis_t[:, 1, 1], 1.0e-15)
        term2[:, 1, 2] = (
            -(x0**2) * x1 * y2 + x0**2 * x2 * y1 + x012 * y0 + x2**3 * y1
        ) * inverse_norm_cubed
        term2 = np.where(not_parallel_j, theta_cosec * term2, term2)
        jac_log = term1 + term2

        inverse_norm = 1.0 / np.linalg.norm(homogeneous, axis=-1)
        identity = np.zeros((3, 2), dtype=np.float64)
        identity[0, 0] = identity[1, 1] = 1.0
        db_dplane = inverse_norm[:, None, None] * (
            identity[None]
            - inverse_norm[:, None, None] ** 2
            * homogeneous[:, :, None]
            * plane[:, None, :]
        )
        inverse_focal = -1.0 / focal
        dplane_dparams = np.zeros((plane.shape[0], 2, 4), dtype=np.float64)
        dplane_dparams[:, 0, 0] = plane[:, 0] * inverse_focal[0]
        dplane_dparams[:, 1, 1] = plane[:, 1] * inverse_focal[1]
        dplane_dparams[:, 0, 2] = inverse_focal[0]
        dplane_dparams[:, 1, 3] = inverse_focal[1]
        jac_bearing = np.matmul(db_dplane, dplane_dparams)
        return residual, np.matmul(jac_log, jac_bearing), valid

    params = np.asarray((fx, fy, cx, cy), dtype=np.float64)
    residual0, _jac0, valid0 = fitted_residual_and_jac(params)
    cost0 = float(np.mean(np.sum(residual0[valid0] ** 2, axis=-1)))
    for _iteration in range(10):
        residual, jacobian, valid = fitted_residual_and_jac(params)
        flat_jacobian = jacobian[valid].reshape(-1, 4)
        flat_residual = residual[valid].reshape(-1)
        normal = flat_jacobian.T @ flat_jacobian
        gradient = flat_jacobian.T @ (-flat_residual)
        try:
            delta = np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            delta = np.zeros(4, dtype=np.float64)
        if np.isfinite(delta).all():
            params += delta
    final_residual, _final_jacobian, final_valid = fitted_residual_and_jac(params)
    final_cost = float(np.mean(np.sum(final_residual[final_valid] ** 2, axis=-1)))
    if np.isfinite(params).all() and final_cost < cost0:
        fx, fy, cx, cy = (float(value) for value in params)
    sx, sy = (float(item) for item in scale_xy)
    shift_x, shift_y = (float(item) for item in shift_xy)
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("Invalid AnyCalib preprocessing scale")
    fx /= sx
    fy /= sy
    cx = (cx - shift_x) / sx
    cy = (cy - shift_y) / sy
    error = math.hypot(error_x, error_y) / max(math.hypot(width, height), 1.0)
    confidence = max(0.05, min(1.0, 1.0 - error * 20.0))
    value = intrinsics(
        original_width,
        original_height,
        fx,
        fy,
        cx,
        cy,
        confidence=confidence,
        source="ANYCALIB",
        bundle_revision=bundle_revision,
        precision=precision,
    )
    if (
        not value.valid
        or not 5.0 <= value.vertical_fov <= 140.0
        or not -0.25 * original_width <= value.cx <= 1.25 * original_width
        or not -0.25 * original_height <= value.cy <= 1.25 * original_height
    ):
        raise ValueError("AnyCalib returned implausible camera intrinsics")
    return value
