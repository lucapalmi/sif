# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Shared setup for the pysif tests.

These test the bindings -- argument handling, the numpy views, what reaches
Python and how -- against an installed pysif. The library itself is tested in
C, by ctest; nothing here re-checks what those tests already do.
"""

import os
from pathlib import Path

import pysif
import pytest

GADGET = Path(__file__).resolve().parent.parent / "data" / "gadget"

# At import rather than in a fixture: HAS_HDF5 below calls into the library
# while the tests are still being collected.
#
# Silent by default. The failure tests provoke errors on purpose, and the C
# log is flushed at exit rather than with the test that wrote it, so it would
# only add noise. PYSIF_TEST_LOG_LEVEL=2 brings it back when debugging.
pysif.init(skip_tuning=True,
           log_level=int(os.environ.get("PYSIF_TEST_LOG_LEVEL", "5")))


def pytest_unconfigure(config):
    pysif.finalize()


def _has_hdf5():
    try:
        pysif.io.gadget_header(str(GADGET / "hdf5_legacy" / "snap_005.hdf5"))
    except RuntimeError:
        return False
    return True


HAS_HDF5 = _has_hdf5()
requires_hdf5 = pytest.mark.skipif(not HAS_HDF5, reason="pysif built without HDF5")
