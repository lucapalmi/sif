# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Shared setup for the pysif tests.

These test the bindings -- argument handling, the numpy views, what reaches
Python and how -- against an installed pysif. The library itself is tested in
C, by ctest; nothing here re-checks what those tests already do.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pysif
import pytest

GADGET = Path(__file__).resolve().parent.parent / "data" / "gadget"
FITS = Path(__file__).resolve().parent.parent / "data" / "fits"

# A SIF_HOME of the session's own, so the tests never read or write the
# developer's settings, wisdom or grid cache. Inherited by the subprocesses
# some tests start.
os.environ["SIF_HOME"] = tempfile.mkdtemp(prefix="pysif-test-home-")
atexit.register(shutil.rmtree, os.environ["SIF_HOME"], ignore_errors=True)

# At import rather than in a fixture: HAS_HDF5 below calls into the library
# while the tests are still being collected.
#
# The library's log reaches the logging module, which pytest captures and
# shows with a failing test, so nothing needs silencing here. pytest's
# --log-level (TRACE is 5) shows more of it.
pysif.system_init(skip_tuning=True)

def pytest_unconfigure(config):
    pysif.system_finalise()


def _has_hdf5():
    try:
        pysif.io.gadget_header(str(GADGET / "hdf5_legacy" / "snap_005.hdf5"))
    except RuntimeError:
        return False
    return True


HAS_HDF5 = _has_hdf5()


def _has_fits():
    try:
        pysif.io.read_fits(str(FITS / "catalogue.fits"), x="RA", y="DEC", z="Z")
    except RuntimeError:
        return False
    return True


HAS_FITS = _has_fits()
requires_hdf5 = pytest.mark.skipif(not HAS_HDF5, reason="pysif built without HDF5")
