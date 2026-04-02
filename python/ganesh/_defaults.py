from __future__ import annotations


class DefaultType:
    """
    Sentinel representing library-provided defaults.

    This sentinel is used in option objects where ``None`` has a separate
    meaning, such as omitting a terminator entirely. Passing ``Default`` asks
    Ganesh to construct the corresponding built-in default object.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return 'Default'


Default = DefaultType()


def default_to_factory(value, factory):
    return factory() if value is Default else value


__all__ = ['Default', 'DefaultType', 'default_to_factory']
