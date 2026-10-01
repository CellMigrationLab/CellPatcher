"""Transition support for the ``repeat`` -> ``biological_repeat`` rename.

The biological-replicate field of :class:`~cellpatcher.PatchRegion`,
:class:`~cellpatcher.CommonConfig`, :class:`~cellpatcher.ConditionRepeatFolder`
and :class:`~cellpatcher.ConditionRepeatResult` (and the matching
``patches.csv`` column) is now called ``biological_repeat``, so it cannot be
mistaken for the MorphoBricks ``technical_repeat`` column. The old ``repeat``
name keeps working for a transition period: as a constructor keyword and as
an attribute, each use emits one ``DeprecationWarning`` and reads or writes
``biological_repeat``; an old ``patches.csv`` with a ``repeat`` column is read
as ``biological_repeat`` (one ``DeprecationWarning`` per file).
"""

import functools
import warnings

OLD_REPEAT_NAME = "repeat"
NEW_REPEAT_NAME = "biological_repeat"


def warn_repeat_renamed(where: str, stacklevel: int = 2) -> None:
    """Emit the rename ``DeprecationWarning`` for one use of the old name.

    *stacklevel* follows :func:`warnings.warn`, counted from the caller of
    this function, so the warning points at the user's code.
    """
    warnings.warn(
        f"{where}: '{OLD_REPEAT_NAME}' was renamed to '{NEW_REPEAT_NAME}' and the old name will be "
        f"removed in a future release; use '{NEW_REPEAT_NAME}' instead.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def accept_deprecated_repeat(cls):
    """Class decorator: keep ``repeat`` as a deprecated alias of ``biological_repeat``.

    Apply it on top of ``@dataclass``. The generated ``__init__`` (and so
    ``dataclasses.fields``/``asdict``/the CSV header) only knows
    ``biological_repeat``; the wrapper maps a ``repeat=`` keyword onto it,
    and a ``repeat`` property reads/writes it. Both warn.
    """
    generated_init = cls.__init__
    class_name = cls.__name__

    @functools.wraps(generated_init)
    def __init__(self, *args, **kwargs):
        if OLD_REPEAT_NAME in kwargs:
            value = kwargs.pop(OLD_REPEAT_NAME)
            warn_repeat_renamed(f"{class_name}({OLD_REPEAT_NAME}=...)")
            existing = kwargs.get(NEW_REPEAT_NAME)
            if existing is not None and existing != value:
                raise TypeError(
                    f"{class_name}() got conflicting values for '{NEW_REPEAT_NAME}' ({existing!r}) and its "
                    f"deprecated alias '{OLD_REPEAT_NAME}' ({value!r}); pass only '{NEW_REPEAT_NAME}'."
                )
            kwargs[NEW_REPEAT_NAME] = value
        generated_init(self, *args, **kwargs)

    def _get(self):
        warn_repeat_renamed(f"{class_name}.{OLD_REPEAT_NAME}")
        return getattr(self, NEW_REPEAT_NAME)

    def _set(self, value):
        warn_repeat_renamed(f"{class_name}.{OLD_REPEAT_NAME}")
        setattr(self, NEW_REPEAT_NAME, value)

    cls.__init__ = __init__
    cls.repeat = property(_get, _set, doc=f"Deprecated alias of ``{NEW_REPEAT_NAME}``.")
    return cls


def upgrade_legacy_repeat_column(fieldnames, rows: list[dict], source: str) -> list[dict]:
    """Map a pre-rename ``repeat`` metadata column onto ``biological_repeat``.

    For CSVs written before the rename (``patches.csv`` with a ``repeat``
    column and no ``biological_repeat`` column): each row gets its value
    under ``biological_repeat`` instead, with one ``DeprecationWarning`` for
    the whole file. Rows of a current CSV are returned unchanged.
    """
    names = list(fieldnames or [])
    if OLD_REPEAT_NAME not in names or NEW_REPEAT_NAME in names:
        return rows
    warnings.warn(
        f"{source}: column '{OLD_REPEAT_NAME}' was renamed to '{NEW_REPEAT_NAME}'; reading it as "
        f"'{NEW_REPEAT_NAME}'. Re-export the patches to write the current column name; support for "
        f"the old column will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )
    return [
        {(NEW_REPEAT_NAME if key == OLD_REPEAT_NAME else key): value for key, value in row.items()}
        for row in rows
    ]
