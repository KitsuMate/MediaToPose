"""Robust, time-aware helpers for offline animation stabilization.

The functions here are symmetric in time and deliberately avoid SciPy. They
operate on ``geometry`` (``mathutils`` in Blender) values and use real
timestamps.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence

from .geometry import Quaternion, Vector


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def _mad(values: Sequence[float]) -> tuple[float, float]:
    center = _median(values)
    spread = 1.4826 * _median(abs(float(value) - center) for value in values)
    return center, max(spread, 1.0e-8)


def _quiet_mad(values: Sequence[float]) -> tuple[float, float]:
    """Estimate ordinary motion without allowing a spike to mask itself."""
    ordered = sorted(float(value) for value in values)
    quiet = ordered[: max(2, (len(ordered) + 1) // 2)]
    return _mad(quiet)


def _interpolation_factor(times: Sequence[float], index: int) -> float:
    span = float(times[index + 1]) - float(times[index - 1])
    if span <= 1.0e-8:
        return 0.5
    return min(
        max((float(times[index]) - float(times[index - 1])) / span, 0.0),
        1.0,
    )


def _continuous_quaternions(values: Sequence[Quaternion]) -> list[Quaternion]:
    result: list[Quaternion] = []
    for value in values:
        current = value.normalized()
        if result and current.dot(result[-1]) < 0.0:
            current.negate()
        result.append(current)
    return result


def _quaternion_vector(reference: Quaternion, value: Quaternion) -> Vector:
    delta = reference.rotation_difference(value).normalized()
    if delta.w < 0.0:
        delta.negate()
    angle = min(float(delta.angle), math.pi)
    if angle <= 1.0e-10:
        return Vector((0.0, 0.0, 0.0))
    return Vector(delta.axis) * angle


def _vector_quaternion(value: Vector) -> Quaternion:
    angle = value.length
    if angle <= 1.0e-10:
        return Quaternion()
    return Quaternion(value / angle, angle)


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a tiny dense system with partial pivoting."""
    size = len(vector)
    augmented = [
        row[:] + [float(value)] for row, value in zip(matrix, vector, strict=True)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-12:
            raise ArithmeticError("singular local regression")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[index][-1] for index in range(size)]


def _polynomial_fit(
    offsets: Sequence[float],
    values: Sequence[float],
    weights: Sequence[float],
    degree: int,
) -> list[float]:
    degree = min(int(degree), len(values) - 1)
    while degree >= 0:
        size = degree + 1
        matrix = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size
        for offset, value, weight in zip(offsets, values, weights, strict=True):
            powers = [1.0]
            for _index in range(2 * degree):
                powers.append(powers[-1] * float(offset))
            for row in range(size):
                rhs[row] += weight * float(value) * powers[row]
                for column in range(size):
                    matrix[row][column] += weight * powers[row + column]
        for index in range(size):
            matrix[index][index] += 1.0e-10
        try:
            return _solve_linear(matrix, rhs)
        except ArithmeticError:
            degree -= 1
    return [float(values[len(values) // 2])]


def _local_fit_components(
    values: Sequence[tuple[float, ...]],
    times: Sequence[float],
    confidences: Sequence[float],
    index: int,
    half_width: float,
) -> tuple[float, ...]:
    center_time = float(times[index])
    selected = _window_indices(times, index, half_width)
    offsets = [float(times[item]) - center_time for item in selected]
    samples = [values[item] for item in selected]
    selected_confidences = [confidences[item] for item in selected]
    return _fit_selected_components(
        samples,
        offsets,
        selected_confidences,
        half_width,
    )


def _window_indices(
    times: Sequence[float],
    index: int,
    half_width: float,
) -> list[int]:
    center_time = float(times[index])
    left = bisect_left(times, center_time - half_width - 1.0e-9)
    right = bisect_right(times, center_time + half_width + 1.0e-9)
    selected = list(range(left, right))
    if len(selected) < 3:
        selected = sorted(
            set(
                selected
                + sorted(
                    range(len(times)),
                    key=lambda item: abs(float(times[item]) - center_time),
                )[: min(3, len(times))]
            )
        )
    return selected


def _fit_selected_components(
    samples: Sequence[tuple[float, ...]],
    offsets: Sequence[float],
    confidences: Sequence[float],
    half_width: float,
) -> tuple[float, ...]:
    base_weights = []
    for confidence, offset in zip(confidences, offsets, strict=True):
        distance = min(abs(offset) / max(half_width, 1.0e-8), 1.0)
        tricube = (1.0 - distance**3) ** 3
        base_weights.append(
            max(tricube, 1.0e-4) * max(min(float(confidence), 1.0), 0.02)
        )
    weights = base_weights[:]
    coefficients: list[list[float]] = []
    # Initial quadratic fit followed by two Huber reweighting passes.
    for iteration in range(3):
        coefficients = [
            _polynomial_fit(
                offsets,
                [sample[axis] for sample in samples],
                weights,
                2,
            )
            for axis in range(len(samples[0]))
        ]
        if iteration == 2:
            break
        residuals = []
        for offset, sample in zip(offsets, samples, strict=True):
            fitted = tuple(
                sum(
                    coefficient[power] * offset**power
                    for power in range(len(coefficient))
                )
                for coefficient in coefficients
            )
            residuals.append(
                math.sqrt(
                    sum(
                        (actual - estimate) ** 2
                        for actual, estimate in zip(sample, fitted, strict=True)
                    )
                )
            )
        _center, scale = _mad(residuals)
        threshold = max(1.5 * scale, 1.0e-8)
        weights = [
            base * (1.0 if residual <= threshold else threshold / residual)
            for base, residual in zip(base_weights, residuals, strict=True)
        ]
    return tuple(coefficient[0] for coefficient in coefficients)


def _half_width(times: Sequence[float], strength: float) -> float:
    intervals = [
        float(right) - float(left)
        for left, right in zip(times, times[1:], strict=False)
        if float(right) > float(left)
    ]
    median_interval = _median(intervals) if intervals else 1.0 / 30.0
    return max(
        2.0 * median_interval,
        0.05 + 0.15 * min(max(strength, 0.0), 1.0),
    )


def _weak_runs(confidences: Sequence[float], threshold: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, confidence in enumerate(confidences):
        weak = float(confidence) < threshold
        if weak and start is None:
            start = index
        elif not weak and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(confidences) - 1))
    return runs


def _weak_reference_factor(
    times: Sequence[float], index: int, left: int, right: int
) -> float:
    span = float(times[right]) - float(times[left])
    if span <= 1.0e-8:
        return 0.0
    if span <= 0.5:
        return min(max((float(times[index]) - float(times[left])) / span, 0.0), 1.0)
    # Long unreliable intervals retain the last known value. Only the final
    # 150 ms ease into the next reliable observation, avoiding both a default
    # pose fallback and a discontinuity when tracking returns.
    ease_start = max(float(times[left]), float(times[right]) - 0.15)
    ease_span = float(times[right]) - ease_start
    if ease_span <= 1.0e-8:
        return 1.0
    factor = min(max((float(times[index]) - ease_start) / ease_span, 0.0), 1.0)
    return factor * factor * (3.0 - 2.0 * factor)


def bridge_low_confidence_vectors(
    values: Sequence[Vector],
    times: Sequence[float],
    confidences: Sequence[float],
    threshold: float = 0.35,
) -> list[Vector]:
    """Bridge weak runs without ever falling back to a default vector."""

    result = [value.copy() for value in values]
    if len(values) != len(times) or len(values) != len(confidences):
        raise ValueError("values, times, and confidences must have equal lengths")
    for start, stop in _weak_runs(confidences, threshold):
        left = start - 1 if start > 0 else None
        right = stop + 1 if stop + 1 < len(values) else None
        if left is None and right is None:
            continue
        for index in range(start, stop + 1):
            if left is None:
                reference = values[right].copy()
            elif right is None:
                reference = values[left].copy()
            else:
                reference = values[left].lerp(
                    values[right],
                    _weak_reference_factor(times, index, left, right),
                )
            trust = min(max(float(confidences[index]) / threshold, 0.0), 1.0) ** 2
            result[index] = reference.lerp(values[index], trust)
    return result


def bridge_low_confidence_quaternions(
    values: Sequence[Quaternion],
    times: Sequence[float],
    confidences: Sequence[float],
    threshold: float = 0.35,
) -> list[Quaternion]:
    """Bridge weak rotation runs using hemisphere-safe interpolation/holds."""

    continuous = _continuous_quaternions(values)
    if len(values) != len(times) or len(values) != len(confidences):
        raise ValueError("values, times, and confidences must have equal lengths")
    result = [value.copy() for value in continuous]
    for start, stop in _weak_runs(confidences, threshold):
        left = start - 1 if start > 0 else None
        right = stop + 1 if stop + 1 < len(values) else None
        if left is None and right is None:
            continue
        for index in range(start, stop + 1):
            if left is None:
                reference = continuous[right].copy()
            elif right is None:
                reference = continuous[left].copy()
            else:
                reference = (
                    continuous[left]
                    .slerp(
                        continuous[right],
                        _weak_reference_factor(times, index, left, right),
                    )
                    .normalized()
                )
            trust = min(max(float(confidences[index]) / threshold, 0.0), 1.0) ** 2
            result[index] = reference.slerp(continuous[index], trust).normalized()
    return _continuous_quaternions(result)


def reduce_scalar_outliers(
    values: Sequence[float],
    times: Sequence[float],
    confidences: Sequence[float] | None = None,
) -> tuple[list[float], int]:
    if len(values) < 3:
        return [float(value) for value in values], 0
    confidences = confidences or [1.0] * len(values)
    residuals = [0.0] * len(values)
    predictions = [float(value) for value in values]
    for index in range(1, len(values) - 1):
        factor = _interpolation_factor(times, index)
        prediction = (
            float(values[index - 1])
            + (float(values[index + 1]) - float(values[index - 1])) * factor
        )
        predictions[index] = prediction
        residuals[index] = abs(float(values[index]) - prediction)
    result = [float(value) for value in values]
    replaced = 0
    for index in range(1, len(values) - 1):
        window = residuals[max(1, index - 3) : min(len(values) - 1, index + 4)]
        center, spread = _quiet_mad(window)
        reconnect = abs(float(values[index + 1]) - float(values[index - 1]))
        residual = residuals[index]
        if float(confidences[index]) < 0.2 or (
            residual > center + 3.5 * spread and reconnect < residual
        ):
            result[index] = predictions[index]
            replaced += 1
    return result, replaced


def reduce_vector_outliers(
    values: Sequence[Vector],
    times: Sequence[float],
    confidences: Sequence[float] | None = None,
) -> tuple[list[Vector], int]:
    if len(values) < 3:
        return [value.copy() for value in values], 0
    confidences = confidences or [1.0] * len(values)
    values = bridge_low_confidence_vectors(values, times, confidences)
    residuals = [0.0] * len(values)
    predictions = [value.copy() for value in values]
    for index in range(1, len(values) - 1):
        prediction = values[index - 1].lerp(
            values[index + 1], _interpolation_factor(times, index)
        )
        predictions[index] = prediction
        residuals[index] = (values[index] - prediction).length
    result = [value.copy() for value in values]
    replaced = 0
    for index in range(1, len(values) - 1):
        window = residuals[max(1, index - 3) : min(len(values) - 1, index + 4)]
        center, spread = _quiet_mad(window)
        reconnect = (values[index + 1] - values[index - 1]).length
        residual = residuals[index]
        if float(confidences[index]) < 0.2 or (
            residual > center + 3.5 * spread and reconnect < residual
        ):
            result[index] = predictions[index]
            replaced += 1
    return result, replaced


def reduce_quaternion_outliers(
    values: Sequence[Quaternion],
    times: Sequence[float],
    confidences: Sequence[float] | None = None,
) -> tuple[list[Quaternion], int]:
    continuous = _continuous_quaternions(values)
    if len(values) < 3:
        return continuous, 0
    confidences = confidences or [1.0] * len(values)
    continuous = bridge_low_confidence_quaternions(continuous, times, confidences)
    residuals = [0.0] * len(values)
    predictions = [value.copy() for value in continuous]
    for index in range(1, len(values) - 1):
        prediction = (
            continuous[index - 1]
            .slerp(continuous[index + 1], _interpolation_factor(times, index))
            .normalized()
        )
        predictions[index] = prediction
        residuals[index] = prediction.rotation_difference(continuous[index]).angle
    result = [value.copy() for value in continuous]
    replaced = 0
    for index in range(1, len(values) - 1):
        window = residuals[max(1, index - 3) : min(len(values) - 1, index + 4)]
        center, spread = _quiet_mad(window)
        reconnect = (
            continuous[index - 1].rotation_difference(continuous[index + 1]).angle
        )
        residual = residuals[index]
        if float(confidences[index]) < 0.2 or (
            residual > center + 3.5 * spread and reconnect < residual
        ):
            result[index] = predictions[index]
            replaced += 1
    return _continuous_quaternions(result), replaced


def reduce_short_quaternion_bursts(
    values: Sequence[Quaternion],
    times: Sequence[float],
    confidences: Sequence[float] | None = None,
    *,
    maximum_duration: float = 0.3,
    minimum_excursion: float = math.radians(35.0),
) -> tuple[list[Quaternion], int]:
    """Bridge short rotation excursions whose clean boundaries agree.

    The ordinary outlier pass compares one sample with its immediate
    neighbours. A multi-sample detector failure therefore validates itself
    and survives. This pass searches only short, bounded intervals and
    requires the interior path to depart substantially from the direct
    boundary interpolation. Sustained turns have no agreeing far boundary
    and are left untouched.
    """

    continuous = _continuous_quaternions(values)
    if len(values) < 3 or len(values) != len(times):
        return continuous, 0
    confidences = confidences or [1.0] * len(values)
    if len(confidences) != len(values):
        raise ValueError("values, times, and confidences must have equal lengths")
    candidates = []
    for left in range(len(values) - 2):
        for right in range(left + 2, len(values)):
            duration = float(times[right]) - float(times[left])
            if duration > maximum_duration + 1.0e-9:
                break
            if duration <= 1.0e-8:
                continue
            boundary_delta = (
                continuous[left].rotation_difference(continuous[right]).angle
            )
            predictions = []
            residuals = []
            for index in range(left + 1, right):
                factor = min(
                    max(
                        (float(times[index]) - float(times[left])) / duration,
                        0.0,
                    ),
                    1.0,
                )
                prediction = (
                    continuous[left].slerp(continuous[right], factor).normalized()
                )
                predictions.append(prediction)
                residuals.append(
                    prediction.rotation_difference(continuous[index]).angle
                )
            peak = max(residuals, default=0.0)
            if peak < minimum_excursion:
                continue
            boundary_limit = max(math.radians(25.0), peak * 0.35)
            if boundary_delta > boundary_limit:
                continue
            path = sum(
                continuous[index - 1].rotation_difference(continuous[index]).angle
                for index in range(left + 1, right + 1)
            )
            if path <= boundary_delta + minimum_excursion * 1.5:
                continue
            weak = min(float(confidences[index]) for index in range(left + 1, right))
            score = peak - boundary_delta + max(0.0, 0.35 - weak)
            candidates.append((score, left, right, predictions))
    result = [value.copy() for value in continuous]
    replaced_indices = set()
    occupied = set()
    for _score, left, right, predictions in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        span = set(range(left, right + 1))
        if span & occupied:
            continue
        interior = set(range(left + 1, right))
        for index, prediction in zip(range(left + 1, right), predictions, strict=True):
            result[index] = prediction
        replaced_indices.update(interior)
        occupied.update(span)
    return _continuous_quaternions(result), len(replaced_indices)


def stabilize_quaternion_jumps(
    values: Sequence[Quaternion],
    times: Sequence[float],
    maximum_speed: float,
    *,
    reacquire_seconds: float = 0.8,
    candidate_tolerance: float = math.radians(12.0),
) -> tuple[list[Quaternion], int]:
    """Hold implausible rotation jumps until the new pose proves stable.

    MediaPipe hand landmarks have no per-joint visibility score, so a malformed
    finger can arrive with confidence 1.0 for several frames. A simple speed
    clamp turns that burst into a large but slower deformation. This stateful
    pass instead keeps the last trusted pose, accepts ordinary motion directly,
    and only eases toward a discontinuous candidate after it remains coherent
    for the configured reacquisition interval.
    """

    continuous = _continuous_quaternions(values)
    if len(values) != len(times):
        raise ValueError("values and times must have equal lengths")
    if len(values) < 2 or maximum_speed <= 0.0:
        return continuous, 0

    result = [continuous[0].copy()]
    trusted = continuous[0].copy()
    candidate = None
    candidate_since = 0.0
    held = 0
    delay = max(float(reacquire_seconds), 0.0)
    tolerance = max(float(candidate_tolerance), 0.0)
    for index in range(1, len(continuous)):
        current = continuous[index]
        elapsed = max(float(times[index]) - float(times[index - 1]), 0.0)
        maximum_step = float(maximum_speed) * elapsed
        angle = trusted.rotation_difference(current).angle
        if maximum_step > 1.0e-8 and angle <= maximum_step:
            trusted = current.copy()
            candidate = None
            candidate_since = 0.0
            result.append(trusted.copy())
            continue

        if (
            candidate is None
            or candidate.rotation_difference(current).angle > tolerance
        ):
            candidate = current.copy()
            candidate_since = float(times[index])
        stable_for = float(times[index]) - candidate_since
        if stable_for + 1.0e-9 < delay or maximum_step <= 1.0e-8:
            result.append(trusted.copy())
            held += 1
            continue

        factor = min(maximum_step / max(angle, 1.0e-12), 1.0)
        trusted = trusted.slerp(current, factor).normalized()
        result.append(trusted.copy())
        held += 1

    return _continuous_quaternions(result), held


def limit_quaternion_speed(
    values: Sequence[Quaternion],
    times: Sequence[float],
    maximum_speed: float,
) -> tuple[list[Quaternion], int]:
    """Clamp physically implausible angular steps without changing direction."""

    continuous = _continuous_quaternions(values)
    if len(values) != len(times):
        raise ValueError("values and times must have equal lengths")
    if len(values) < 2 or maximum_speed <= 0.0:
        return continuous, 0
    result = [continuous[0].copy()]
    limited = 0
    for index in range(1, len(continuous)):
        elapsed = max(float(times[index]) - float(times[index - 1]), 0.0)
        maximum_step = float(maximum_speed) * elapsed
        angle = result[-1].rotation_difference(continuous[index]).angle
        if maximum_step > 1.0e-8 and angle > maximum_step:
            result.append(
                result[-1].slerp(continuous[index], maximum_step / angle).normalized()
            )
            limited += 1
        else:
            result.append(continuous[index].copy())
    return _continuous_quaternions(result), limited


def smooth_scalars(
    values: Sequence[float],
    times: Sequence[float],
    confidences: Sequence[float] | None,
    strength: float,
) -> list[float]:
    influence = min(max(float(strength), 0.0), 1.0)
    if len(values) < 3 or influence <= 0.0:
        return [float(value) for value in values]
    confidences = confidences or [1.0] * len(values)
    tuples = [(float(value),) for value in values]
    half_width = _half_width(times, influence)
    fitted = [
        _local_fit_components(tuples, times, confidences, index, half_width)[0]
        for index in range(len(values))
    ]
    return [
        float(original) + (estimate - float(original)) * influence
        for original, estimate in zip(values, fitted, strict=True)
    ]


def smooth_vectors(
    values: Sequence[Vector],
    times: Sequence[float],
    confidences: Sequence[float] | None,
    strength: float,
) -> list[Vector]:
    influence = min(max(float(strength), 0.0), 1.0)
    if len(values) < 3 or influence <= 0.0:
        return [value.copy() for value in values]
    confidences = confidences or [1.0] * len(values)
    tuples = [tuple(float(component) for component in value) for value in values]
    half_width = _half_width(times, influence)
    fitted = [
        Vector(_local_fit_components(tuples, times, confidences, index, half_width))
        for index in range(len(values))
    ]
    return [
        original.lerp(estimate, influence)
        for original, estimate in zip(values, fitted, strict=True)
    ]


def smooth_quaternions(
    values: Sequence[Quaternion],
    times: Sequence[float],
    confidences: Sequence[float] | None,
    strength: float,
) -> list[Quaternion]:
    influence = min(max(float(strength), 0.0), 1.0)
    continuous = _continuous_quaternions(values)
    if len(values) < 3 or influence <= 0.0:
        return continuous
    confidences = confidences or [1.0] * len(values)
    half_width = _half_width(times, influence)
    result = []
    for index, reference in enumerate(continuous):
        selected = _window_indices(times, index, half_width)
        tangent_values = [
            tuple(_quaternion_vector(reference, continuous[item])) for item in selected
        ]
        fitted = Vector(
            _fit_selected_components(
                tangent_values,
                [float(times[item]) - float(times[index]) for item in selected],
                [confidences[item] for item in selected],
                half_width,
            )
        )
        estimate = (reference @ _vector_quaternion(fitted)).normalized()
        candidate = reference.slerp(estimate, influence).normalized()
        if result and candidate.dot(result[-1]) < 0.0:
            candidate.negate()
        result.append(candidate)
    return result


def smooth_quaternions_bounded(
    values: Sequence[Quaternion],
    times: Sequence[float],
    confidences: Sequence[float] | None,
    strength: float,
) -> list[Quaternion]:
    """Smooth rotations without quadratic extrapolation outside the window.

    Finger and wrist channels are particularly sensitive to a small tangent-
    space overshoot. Average neighboring logarithmic rotation vectors with
    positive weights only, then blend from the observed center sample. This is
    symmetric and confidence-aware but cannot invent the polynomial extrema
    used by the general motion smoother.
    """
    influence = min(max(float(strength), 0.0), 1.0)
    continuous = _continuous_quaternions(values)
    if len(values) < 3 or influence <= 0.0:
        return continuous
    confidences = confidences or [1.0] * len(values)
    half_width = _half_width(times, influence)
    result = []
    for index, reference in enumerate(continuous):
        selected = _window_indices(times, index, half_width)
        accumulated = Vector((0.0, 0.0, 0.0))
        total = 0.0
        center_time = float(times[index])
        for item in selected:
            distance = min(
                abs(float(times[item]) - center_time) / max(half_width, 1.0e-8),
                1.0,
            )
            weight = (1.0 - distance**3) ** 3
            weight *= max(min(float(confidences[item]), 1.0), 0.02)
            accumulated += _quaternion_vector(reference, continuous[item]) * weight
            total += weight
        estimate = (
            reference @ _vector_quaternion(accumulated / max(total, 1.0e-12))
        ).normalized()
        candidate = reference.slerp(estimate, influence).normalized()
        if result and candidate.dot(result[-1]) < 0.0:
            candidate.negate()
        result.append(candidate)
    return result
