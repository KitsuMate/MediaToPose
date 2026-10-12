"""Vector, quaternion and matrix types for modules that must run without Blender.

Inside Blender these are the ``mathutils`` types, so results are bit-for-bit
what the rest of the add-on computes. Plain Python (the unit tier, tools)
gets a small pure-Python implementation of the subset of the ``mathutils``
API that the motion and cleanup modules use. Both share the same
conventions: quaternions are ``(w, x, y, z)``, matrices are row-major and
multiply column vectors, and ``slerp`` takes the shortest path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

_EPSILON = 1.1920929e-07


class _Vector:
    __slots__ = ("_values",)

    def __init__(self, values: Iterable[float] = (0.0, 0.0, 0.0)):
        self._values = [float(value) for value in values]

    # Sequence protocol
    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value) -> None:
        self._values[index] = float(value)

    def __repr__(self) -> str:
        return f"Vector({tuple(self._values)!r})"

    def __eq__(self, other) -> bool:
        try:
            return len(other) == len(self) and all(
                a == float(b) for a, b in zip(self._values, other, strict=True)
            )
        except TypeError:
            return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    @property
    def x(self) -> float:
        return self._values[0]

    @x.setter
    def x(self, value: float) -> None:
        self._values[0] = float(value)

    @property
    def y(self) -> float:
        return self._values[1]

    @y.setter
    def y(self, value: float) -> None:
        self._values[1] = float(value)

    @property
    def z(self) -> float:
        return self._values[2]

    @z.setter
    def z(self, value: float) -> None:
        self._values[2] = float(value)

    # Arithmetic
    def _zip(self, other) -> Iterator[tuple[float, float]]:
        if len(other) != len(self):
            raise ValueError("vector sizes differ")
        return zip(self._values, (float(value) for value in other), strict=True)

    def __add__(self, other) -> _Vector:
        return _Vector(a + b for a, b in self._zip(other))

    __radd__ = __add__

    def __sub__(self, other) -> _Vector:
        return _Vector(a - b for a, b in self._zip(other))

    def __rsub__(self, other) -> _Vector:
        return _Vector(b - a for a, b in self._zip(other))

    def __mul__(self, scalar) -> _Vector:
        if isinstance(scalar, (int, float)):
            return _Vector(value * float(scalar) for value in self._values)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> _Vector:
        return _Vector(value / float(scalar) for value in self._values)

    def __neg__(self) -> _Vector:
        return _Vector(-value for value in self._values)

    def __iadd__(self, other) -> _Vector:
        self._values = [a + b for a, b in self._zip(other)]
        return self

    def __isub__(self, other) -> _Vector:
        self._values = [a - b for a, b in self._zip(other)]
        return self

    def __imul__(self, scalar) -> _Vector:
        self._values = [value * float(scalar) for value in self._values]
        return self

    def __itruediv__(self, scalar) -> _Vector:
        self._values = [value / float(scalar) for value in self._values]
        return self

    # mathutils API subset
    def copy(self) -> _Vector:
        return _Vector(self._values)

    def to_tuple(self) -> tuple[float, ...]:
        return tuple(self._values)

    @property
    def length(self) -> float:
        return math.sqrt(sum(value * value for value in self._values))

    @property
    def length_squared(self) -> float:
        return sum(value * value for value in self._values)

    magnitude = length

    def dot(self, other) -> float:
        return sum(a * b for a, b in self._zip(other))

    def cross(self, other) -> _Vector:
        ax, ay, az = self._values
        bx, by, bz = (float(value) for value in other)
        return _Vector((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx))

    def normalized(self) -> _Vector:
        length = self.length
        if length <= 0.0:
            return self.copy()
        return self / length

    def normalize(self) -> None:
        self._values = list(self.normalized())

    def lerp(self, other, factor: float) -> _Vector:
        factor = float(factor)
        return _Vector(a + (b - a) * factor for a, b in self._zip(other))

    def angle(self, other, fallback=None) -> float:
        lengths = self.length * _Vector(other).length
        if lengths <= 0.0:
            if fallback is not None:
                return fallback
            raise ValueError("zero length vectors have no valid angle")
        return math.acos(min(max(self.dot(other) / lengths, -1.0), 1.0))


class _Quaternion:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, values: Iterable[float] | None = None, angle=None):
        if values is None:
            self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
            return
        if angle is not None:
            axis = _Vector(values)
            length = axis.length
            if length <= 0.0:
                self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
                return
            half = 0.5 * float(angle)
            scale = math.sin(half) / length
            self.w = math.cos(half)
            self.x, self.y, self.z = (value * scale for value in axis)
            return
        self.w, self.x, self.y, self.z = (float(value) for value in values)

    def __iter__(self) -> Iterator[float]:
        return iter((self.w, self.x, self.y, self.z))

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index):
        return (self.w, self.x, self.y, self.z)[index]

    def __repr__(self) -> str:
        return f"Quaternion({(self.w, self.x, self.y, self.z)!r})"

    def __eq__(self, other) -> bool:
        try:
            return tuple(self) == tuple(float(value) for value in other)
        except TypeError:
            return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> _Quaternion:
        return _Quaternion(tuple(self))

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.dot(self))

    def dot(self, other) -> float:
        return sum(a * float(b) for a, b in zip(self, other, strict=True))

    def normalized(self) -> _Quaternion:
        length = self.magnitude
        if length <= 0.0:
            return _Quaternion()
        return _Quaternion(value / length for value in self)

    def normalize(self) -> None:
        self.w, self.x, self.y, self.z = self.normalized()

    def negate(self) -> None:
        self.w, self.x, self.y, self.z = (-value for value in self)

    def conjugated(self) -> _Quaternion:
        return _Quaternion((self.w, -self.x, -self.y, -self.z))

    def inverted(self) -> _Quaternion:
        scale = 1.0 / max(self.dot(self), 1.0e-30)
        return _Quaternion(value * scale for value in self.conjugated())

    def __matmul__(self, other):
        if isinstance(other, _Quaternion):
            aw, ax, ay, az = self
            bw, bx, by, bz = other
            return _Quaternion(
                (
                    aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                )
            )
        return self.to_matrix() @ _Vector(other)

    def __mul__(self, scalar) -> _Quaternion:
        return _Quaternion(value * float(scalar) for value in self)

    __rmul__ = __mul__

    def __add__(self, other) -> _Quaternion:
        return _Quaternion(a + float(b) for a, b in zip(self, other, strict=True))

    def __neg__(self) -> _Quaternion:
        return _Quaternion(-value for value in self)

    def rotation_difference(self, other) -> _Quaternion:
        return self.inverted() @ _Quaternion(tuple(other))

    @property
    def angle(self) -> float:
        w = self.normalized().w
        value = 2.0 * math.acos(min(max(w, -1.0), 1.0))
        return value if math.isfinite(value) else 0.0

    @property
    def axis(self) -> _Vector:
        normalized = self.normalized()
        half = math.acos(min(max(normalized.w, -1.0), 1.0))
        sine = math.sin(half)
        if abs(sine) < _EPSILON:
            sine = 1.0
        axis = _Vector((normalized.x, normalized.y, normalized.z)) / sine
        if axis.length <= 0.0 or not all(math.isfinite(value) for value in axis):
            return _Vector((1.0, 0.0, 0.0))
        return axis

    def slerp(self, other, factor: float) -> _Quaternion:
        first = self.normalized()
        second = _Quaternion(tuple(other)).normalized()
        cosom = first.dot(second)
        if cosom < 0.0:
            cosom = -cosom
            first = -first
        factor = float(factor)
        if 1.0 - cosom > 0.0001:
            omega = math.acos(min(cosom, 1.0))
            sinom = math.sin(omega)
            left = math.sin((1.0 - factor) * omega) / sinom
            right = math.sin(factor * omega) / sinom
        else:
            left, right = 1.0 - factor, factor
        return _Quaternion(
            left * a + right * b for a, b in zip(first, second, strict=True)
        )

    def to_matrix(self) -> _Matrix:
        w, x, y, z = self
        return _Matrix(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
            )
        )


class _Matrix:
    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[Sequence[float]]):
        self._rows = [[float(value) for value in row] for row in rows]

    def __iter__(self) -> Iterator[_Vector]:
        return (_Vector(row) for row in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index) -> _Vector:
        return _Vector(self._rows[index])

    def __repr__(self) -> str:
        return f"Matrix({self._rows!r})"

    @classmethod
    def Identity(cls, size: int) -> _Matrix:  # noqa: N802 - mathutils API
        return cls(
            [
                [1.0 if row == column else 0.0 for column in range(size)]
                for row in range(size)
            ]
        )

    def copy(self) -> _Matrix:
        return _Matrix(self._rows)

    def transposed(self) -> _Matrix:
        return _Matrix(list(zip(*self._rows, strict=True)))

    def __matmul__(self, other):
        if isinstance(other, _Matrix):
            columns = list(zip(*other._rows, strict=True))
            return _Matrix(
                [
                    [
                        sum(a * b for a, b in zip(row, column, strict=True))
                        for column in columns
                    ]
                    for row in self._rows
                ]
            )
        vector = [float(value) for value in other]
        return _Vector(
            sum(a * b for a, b in zip(row, vector, strict=True)) for row in self._rows
        )

    def to_3x3(self) -> _Matrix:
        return _Matrix([row[:3] for row in self._rows[:3]])

    def to_quaternion(self) -> _Quaternion:
        rows = self.to_3x3()._rows
        columns = [list(column) for column in zip(*rows, strict=True)]
        for column in columns:
            length = math.sqrt(sum(value * value for value in column))
            if length > 0.0:
                for index in range(3):
                    column[index] /= length
        m = [list(row) for row in zip(*columns, strict=True)]
        trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0.0:
            s = 2.0 * math.sqrt(1.0 + trace)
            result = (
                0.25 * s,
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
            )
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(max(1.0 + m[0][0] - m[1][1] - m[2][2], 0.0))
            result = (
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
            )
        elif m[1][1] > m[2][2]:
            s = 2.0 * math.sqrt(max(1.0 + m[1][1] - m[0][0] - m[2][2], 0.0))
            result = (
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
            )
        else:
            s = 2.0 * math.sqrt(max(1.0 + m[2][2] - m[0][0] - m[1][1], 0.0))
            result = (
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s,
            )
        quaternion = _Quaternion(result)
        if quaternion.w < 0.0:
            quaternion.negate()
        return quaternion.normalized()


try:  # Blender: use the native types so results match the rest of the add-on.
    from mathutils import Matrix, Quaternion, Vector
except ImportError:  # Plain Python: unit tests and tools.
    Matrix = _Matrix  # type: ignore[assignment,misc]
    Quaternion = _Quaternion  # type: ignore[assignment,misc]
    Vector = _Vector  # type: ignore[assignment,misc]


__all__ = ["Matrix", "Quaternion", "Vector"]
