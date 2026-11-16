# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The Python half of the bridge from sif's log to the logging module.

Every message sif logs goes to the logger ``pysif.<tag>``, where the tag is
the library's own (``io.hdf5``, ``finder.exodus``, ...), so logging's usual
configuration selects and silences it: ``logging.getLogger("pysif.io")``
is every reader and writer. sif's TRACE level is logging level 5, named
"TRACE" unless something else has named it already.

The compiled half (python/src/py_log.c) calls emit() for each message, and
keeps the library's own level at c_level(), so that a message no pysif logger
would accept is never formatted at all. It re-reads c_level() whenever the
logging configuration changes -- see follow_levels() -- and before each call
that releases the GIL. Private: configure the loggers, not this module.
"""

import logging

TRACE = 5

# sif's levels, TRACE through ERROR, as logging's.
_LEVELS = (TRACE, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)

if logging.getLevelName(TRACE) == f"Level {TRACE}":
    logging.addLevelName(TRACE, "TRACE")

_loggers = {}


def emit(level, tag, message):
    """Log one of sif's messages under pysif.<tag>."""
    logger = _loggers.get(tag)
    if logger is None:
        logger = logging.getLogger(f"pysif.{tag}" if tag else "pysif")
        _loggers[tag] = logger
    # No arguments, so a '%' in the message is never taken as a format; and
    # stacklevel=2 credits the line of Python that called into sif, the
    # frame under this one, rather than this file.
    logger.log(_LEVELS[level], message, stacklevel=2)


def c_level():
    """The lowest sif level any pysif logger would accept: 0 (TRACE) to 5
    (none). A logger set below its parent counts, so the floor is the
    minimum over pysif and every pysif.* logger, and logging.disable()
    raises it."""
    manager = logging.Logger.manager
    level = logging.getLogger("pysif").getEffectiveLevel()
    for name, logger in list(manager.loggerDict.items()):
        if name.startswith("pysif.") and isinstance(logger, logging.Logger):
            level = min(level, logger.getEffectiveLevel())
    level = max(level, manager.disable + 1)
    for sif_level, py_level in enumerate(_LEVELS):
        if level <= py_level:
            return sif_level
    return len(_LEVELS)


def follow_levels(resync):
    """Call resync() whenever a logger's level changes.

    logging clears its loggers' level caches, through Manager._clear_cache(),
    on every change that can move an effective level: setLevel(), disable(),
    and the configuration functions, which use setLevel(). Hooking that is
    what keeps the library's level right for calls that do not release the
    GIL, which do not re-read it themselves. Returns whether the hook went
    in; without it the level still follows at every call that releases the
    GIL.
    """
    manager = logging.Logger.manager
    clear_cache = getattr(manager, "_clear_cache", None)
    if clear_cache is None:
        return False

    def _clear_cache():
        clear_cache()
        resync()

    manager._clear_cache = _clear_cache
    return True
