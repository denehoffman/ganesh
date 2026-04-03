from __future__ import annotations

try:
    from ._ganesh import (  # ty:ignore[unresolved-import]
        GaneshConfigError,
        GaneshError,
        GaneshNumericalError,
    )
except ImportError:

    class GaneshError(Exception):
        pass

    class GaneshConfigError(GaneshError):
        pass

    class GaneshNumericalError(GaneshError):
        pass


__all__ = ['GaneshConfigError', 'GaneshError', 'GaneshNumericalError']
