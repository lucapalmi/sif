# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""The type stubs (python/pysif/*.pyi) against the installed pysif.

Editors read the stubs, never the compiled module, so a stub that has
fallen behind the bindings shows the wrong parameters or an old docstring
without anything failing. packaging/stubs.py holds the two side by side;
this runs it, and checks that it does notice each way they can drift.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "packaging" / "stubs.py"
_spec = importlib.util.spec_from_file_location("stubs", _SCRIPT)
stubs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stubs)


def test_stubs_match_pysif():
    problems = stubs.check()
    assert not problems, "\n".join(
        ["the stubs differ from pysif (packaging/stubs.py):"] + problems)


@pytest.fixture
def copy(tmp_path):
    """The stubs in a directory of their own, to break."""
    for path in stubs.STUBS.glob("*.pyi"):
        shutil.copy(path, tmp_path)
    return tmp_path


def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize("filename, old, new, expected", [
    ("measure.pyi", "def survey_volume(randoms: Field,",
     "def survey_volume(field: Field,", "parameters differ"),
    ("measure.pyi", "ext: float = 5.0,", "ext: float = 4.0,",
     "parameter ext: default (expected 5.0, found 4.0)"),
    ("__init__.pyi", "def translate(self, offset: _Vec3) -> None:\n        \"\"\"Shift every void",
     "def translate(self, *, offset: _Vec3) -> None:\n        \"\"\"Shift every void",
     "parameter offset: kind"),
    ("io.pyi", "    \"\"\"Read an .xgrid binary into a Grid.",
     "    \"\"\"Read an .xgrid file into a Grid.",
     "docstring differs from the compiled one"),
    ("finders.pyi", "def read_checkpoint(", "def _read_checkpoint(",
     "pysif.finders.read_checkpoint: missing from the stub"),
    ("__init__.pyi", "    @property\n    def n_bits(self) -> int:",
     "    def n_bits(self) -> int:",
     "pysif.Bitmask.n_bits: a method in the stub, a property in pysif"),
    ("model.pyi", "def bbks_g(", "def bbks_h(",
     "pysif.model.bbks_h: in the stub, not in pysif"),
])
def test_check_notices(copy, filename, old, new, expected):
    edit(copy / filename, old, new)
    problems = stubs.check(copy)
    assert any(expected in p for p in problems), problems


def test_fix_rewrites_docstrings(copy):
    edit(copy / "io.pyi", "    \"\"\"Read an .xgrid binary into a Grid.",
         "    \"\"\"Something else entirely.\n\n    Over two paragraphs.")
    edit(copy / "__init__.pyi", "        \"\"\"Count the number of set bits.\"\"\"",
         "        \"\"\"Stale.\"\"\"")
    assert stubs.check(copy)
    assert {p.name for p in stubs.fix(copy)} == {"io.pyi", "__init__.pyi"}
    assert stubs.check(copy) == []
    for path in copy.glob("*.pyi"):
        assert path.read_text() == (stubs.STUBS / path.name).read_text()
