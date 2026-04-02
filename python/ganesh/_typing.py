from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    FloatVectorLike: TypeAlias = Sequence[float] | npt.NDArray[np.float64]
    FloatMatrixLike: TypeAlias = Sequence[Sequence[float]] | npt.NDArray[np.float64]
else:
    FloatVectorLike: TypeAlias = Sequence[float]
    FloatMatrixLike: TypeAlias = Sequence[Sequence[float]]

BoundsLike: TypeAlias = Sequence[tuple[float | None, float | None]]
