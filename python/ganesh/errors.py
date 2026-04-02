from __future__ import annotations

try:
    from ._ganesh import GaneshConfigError, GaneshError, GaneshNumericalError  # ty:ignore[unresolved-import]
except ImportError:

    class GaneshError(Exception):
        pass

    class GaneshConfigError(GaneshError):
        pass

    class GaneshNumericalError(GaneshError):
        pass


__all__ = ['GaneshConfigError', 'GaneshError', 'GaneshNumericalError']
