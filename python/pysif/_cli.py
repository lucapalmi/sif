# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""Launchers for the command-line programs a wheel carries.

The programs are native executables in pysif/bin/, next to the extension,
where the wheel's bundled libraries are found. pip does not put that
directory on PATH, so for each program [project.scripts] in pyproject.toml
installs a command that lands here, and each function below replaces the
Python process with the program: from then on it is the program alone, with
its own signals, output and exit status.
"""

import os
import sys


def _exec(name):
    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", name)
    if not os.access(exe, os.X_OK):
        sys.exit(f"{name}: not part of this pysif installation "
                 f"(built with SIF_BUILD_PROGRAMS=OFF?)")
    os.execv(exe, [exe, *sys.argv[1:]])


def exodus():
    _exec("sif-exodus")
