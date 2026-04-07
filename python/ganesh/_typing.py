from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatVectorLike: TypeAlias = Sequence[float] | npt.NDArray[np.float64]
    FloatMatrixLike: TypeAlias = Sequence[Sequence[float]] | npt.NDArray[np.float64]
else:
    FloatVectorLike: TypeAlias = Sequence[float]
    FloatMatrixLike: TypeAlias = Sequence[Sequence[float]]

BoundsLike: TypeAlias = Sequence[tuple[float | None, float | None]]


class MCMCDiagnostics(TypedDict):
    r_hat: FloatVectorLike
    ess: FloatVectorLike
    acceptance_rates: FloatVectorLike


class Point(TypedDict):
    x: float
    fx: FloatVectorLike


class Particle(TypedDict):
    position: Point
    velocity: FloatVectorLike
    best: Point


class Swarm(TypedDict):
    particles: list[Particle]
    topology: str
    update_method: str
    boundary_method: str
    position_initializer: dict
    velocity_initializer: dict
