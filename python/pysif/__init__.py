# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""sif: cosmic void finding and analysis.

The finders, the grid-based delta statistics and the settings need
system_init() first, and system_finalise() when done; everything else
works without it.

The data structures live here in the package root -- Field, Grid,
Catalogue and the rest -- and the operations on them are grouped
into submodules: io for reading and writing, finders for void
identification, measure for measurements taken from data, and
model for theoretical predictions.

What the library logs goes to the logging module, under pysif and a
logger per subsystem (pysif.io.hdf5, pysif.finder.exodus, ...), so the
usual configuration applies: logging.getLogger("pysif").setLevel(
logging.INFO) shows progress, which is hidden by default as every
library's is.

pysif.real is the NumPy scalar type matching the precision the
library was built with, so np.zeros(n, dtype=pysif.real) gives
arrays the bindings take without a conversion copy.
"""

# Everything is compiled into _pysif, which also registers the submodules
# (pysif.io, pysif.finders, ...) in sys.modules as it loads. This file only
# makes that the package's public face. It is a package rather than a lone
# extension module so the wheel has somewhere to put other files -- the
# command-line programs among them.
from ._pysif import *  # noqa: F401,F403
from . import _pysif
from . import _log

# What sif logs goes to the logging module from here on, under pysif.<tag>.
_pysif._log_install(_log.emit, _log.c_level)

# The functions defined at the root report _pysif as their module, which
# autodoc reads as "imported from elsewhere" and leaves out of the pysif page.
# They are pysif's, so they say so. The types need nothing: their names are
# spelled "pysif.Field" and so on in C.
for _obj in vars(_pysif).values():
    if getattr(_obj, "__module__", None) == _pysif.__name__:
        _obj.__module__ = __name__
del _obj
