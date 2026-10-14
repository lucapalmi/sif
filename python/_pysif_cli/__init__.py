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

WHY THIS IS NOT pysif._cli
--------------------------
A launcher inside pysif imports pysif to run, and pysif loads the extension
and with it the OpenMP runtime. With OMP_PROC_BIND set -- as most HPC job
scripts set it -- libgomp binds the initial thread to the first place the
moment it loads: this Python process ends up pinned to one core, execv keeps
the mask, and the program's own OpenMP then builds its places from that one
core and runs every thread on it. A 72-thread sif-exodus ran at 184% CPU on
an idle node this way, at 400 s for a rung that takes a few.

So the launchers live in a package of their own that never imports pysif.
find_spec() locates pysif without executing its __init__, and nothing here
may import it either: the program has to start with the process's affinity
untouched.
"""

import importlib.util
import os
import sys


def _exec(name):
    spec = importlib.util.find_spec("pysif")
    dirs = (spec.submodule_search_locations or []) if spec else []
    for d in dirs:
        exe = os.path.join(d, "bin", name)
        if os.access(exe, os.X_OK):
            os.execv(exe, [exe, *sys.argv[1:]])
    sys.exit(f"{name}: not part of this pysif installation "
             f"(built with SIF_BUILD_PROGRAMS=OFF?)")


def exodus():
    _exec("sif-exodus")
