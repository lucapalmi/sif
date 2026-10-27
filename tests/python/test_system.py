# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The system state: what needs pysif.system_init() says so with a
RuntimeError, what does not works without it, and the settings refuse what
the config file cannot hold."""

import subprocess
import sys
import textwrap

import pysif
import pytest

# conftest.py initialises the session, so a process that never does has to be
# a separate one.
UNINITIALISED = textwrap.dedent("""
    import numpy as np
    import pysif

    grid = pysif.Grid(n_cells=16, box_length=100.0)
    radii = np.array([20.0, 15.0], dtype=pysif.real)

    calls = {
        "spherical": lambda: pysif.finders.spherical(grid, radii, -0.2),
        "delta_moments_grid": lambda: pysif.measure.delta_moments_grid(
            grid, radii, 2),
        "set_setting": lambda: pysif.set_setting("grid_cache_enabled", "1"),
    }
    for name, call in calls.items():
        try:
            call()
        except RuntimeError as e:
            assert "system_init" in str(e), (name, str(e))
            print(name, "RuntimeError")
        else:
            print(name, "no error")

    # The one call that falls back rather than raising.
    print("fallback", pysif.get_setting("key", "fallback"))
""")


def test_uninitialised_calls_raise_runtime_error():
    out = subprocess.run([sys.executable, "-c", UNINITIALISED],
                         capture_output=True, text=True, check=True).stdout
    # The library logs to stdout at info level, "[INFO] tag: ..."; keep only
    # what the script printed.
    lines = [line for line in out.splitlines() if not line.startswith("[")]
    assert lines == [
        "spherical RuntimeError",
        "delta_moments_grid RuntimeError",
        "set_setting RuntimeError",
        "fallback fallback",
    ]


@pytest.mark.parametrize("key, value, match", [
    ("a=b", "x", "not a setting"),
    ("fft_tuning_max_gb", "2", "not a setting"),
    ("fft_tuning_max_gib", "x\ny", "config file cannot hold"),
    ("fft_tuning_max_gib", "lots", "expected a number"),
    ("grid_cache_enabled", "maybe", "expected 1/0"),
])
def test_set_setting_refuses(key, value, match):
    with pytest.raises(ValueError, match=match):
        pysif.set_setting(key, value)


def test_setting_override_is_for_this_process():
    saved = pysif.get_setting("fft_tuning_max_gib")
    pysif.set_setting("fft_tuning_max_gib", "3", persist=False)
    assert pysif.get_setting("fft_tuning_max_gib") == "3"
    pysif.set_setting("fft_tuning_max_gib", None, persist=False)
    assert pysif.get_setting("fft_tuning_max_gib") == saved
    with pytest.raises(ValueError, match="persist=False"):
        pysif.set_setting("fft_tuning_max_gib", None)


def test_get_setting_stores_nothing():
    assert pysif.get_setting("not_a_setting") is None
    assert pysif.get_setting("not_a_setting", "x") == "x"


def test_deprecated_names_warn():
    with pytest.warns(FutureWarning, match=r"pysif\.system_init\(\)"):
        pysif.init(skip_tuning=True, log_level=5)  # already up: a no-op
