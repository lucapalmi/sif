# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.io.describe(), check() and inspect(): what reaches Python of the
library's file descriptions and checks, which test_file_info and
test_file_check exercise format by format in C.
"""

import os
from pathlib import Path

import pysif
import pytest
from conftest import FITS, GADGET, HAS_FITS

SNAPSHOT = GADGET / "f1_legacy" / "snap_005"


def test_describe_snapshot():
    d = pysif.io.describe(SNAPSHOT)
    assert d["format"] == "gadget"
    assert d["n_files"] == 3
    assert d["meta"]["BoxSize"] == 10000.0
    tables = {t["name"]: t for t in d["tables"]}
    assert tables["PartType1"]["n_rows"] == 100
    assert tables["PartType1"]["meta"]["MassTable"] == 0.5
    pos = tables["PartType1"]["columns"][0]
    assert pos == {"name": "POS", "type": "float32", "width": 3, "unit": None}


def test_describe_grid(tmp_path):
    path = tmp_path / "g.xgrid"
    pysif.io.write_grid(str(path), pysif.Grid(4, 10.0))
    d = pysif.io.describe(path, format="xgrid")
    assert d["tables"] == []
    assert d["arrays"][0]["name"] == "cells"
    assert d["arrays"][0]["shape"] == (4, 4, 4)


@pytest.mark.parametrize("form", [str, os.fsencode, Path])
def test_path_forms(form):
    assert pysif.io.describe(form(SNAPSHOT))["format"] == "gadget"
    assert pysif.io.check(path=form(SNAPSHOT))["ok"]


def test_describe_refusals(tmp_path):
    with pytest.raises(FileNotFoundError):
        pysif.io.describe(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="format must be"):
        pysif.io.describe(SNAPSHOT, format="csv")
    (tmp_path / "t.txt").write_text("1 2 3\n")
    with pytest.raises(OSError, match="not an .xfield file"):
        pysif.io.describe(tmp_path / "t.txt", format="xfield")


def test_check_sound():
    c = pysif.io.check(SNAPSHOT)
    assert c["ok"] and c["n_errors"] == 0 and c["n_warnings"] == 0
    assert c["problems"] == []
    assert c["content"] == "GADGET snapshot"
    table = next(t for t in c["info"]["tables"] if t["name"] == "PartType1")
    ids = next(col for col in table["columns"] if col["name"] == "ID")
    assert ids["stats"] == {"n": 100, "n_nonfinite": 0, "min": 1000000.0,
                            "max": 1000099.0}


def test_check_problems(tmp_path):
    path = tmp_path / "cat.txt"
    path.write_text("#n=3\n#cx cy cz r\n1 2 3 4\n5 6 7 -1\n")
    c = pysif.io.check(path)
    assert not c["ok"]
    assert c["content"] == "sif catalogue"
    messages = [p["message"] for p in c["problems"]]
    assert all(p["severity"] == "error" for p in c["problems"])
    assert any("expected every radius > 0" in m for m in messages)
    assert any("n = 3, the table has 2 rows" in m for m in messages)
    r = c["info"]["tables"][0]["columns"][3]["stats"]
    assert r["min"] == -1.0 and r["max"] == 4.0


def test_check_unreadable(tmp_path):
    # Not an exception: the check says why.
    c = pysif.io.check(tmp_path / "missing.txt")
    assert c["info"] is None and not c["ok"]
    assert "No such file or directory" in c["problems"][0]["message"]


def test_check_text_column(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("x,name\n1,a\n2,b\n")
    columns = pysif.io.check(path)["info"]["tables"][0]["columns"]
    assert columns[1]["stats"] == {"n": 2, "n_nonfinite": 0, "min": None,
                                   "max": None}


@pytest.mark.skipif(not HAS_FITS, reason="pysif built without FITS")
def test_check_fits_nulls():
    c = pysif.io.check(FITS / "catalogue.fits")
    galaxies = c["info"]["tables"][0]
    znull = next(col for col in galaxies["columns"] if col["name"] == "ZNULL")
    assert znull["stats"]["n_nonfinite"] == 10


def test_inspect(capsys):
    assert pysif.io.inspect(SNAPSHOT) is None
    out = capsys.readouterr().out
    assert "table PartType1, 100 rows, 3 columns" in out
    assert "non-finite" not in out
    pysif.io.inspect(SNAPSHOT, check=True)
    out = capsys.readouterr().out
    assert "non-finite" in out and "result    sound" in out
    with pytest.raises(TypeError):
        pysif.io.inspect(SNAPSHOT, None, True)  # check is keyword-only
