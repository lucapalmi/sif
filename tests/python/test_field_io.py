# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""pysif.io.read_field_ascii and read_field_binary.

The files are written here with numpy -- structured dtypes for records, one
array after another for blocks -- so the reader is checked against numpy's
idea of a layout rather than against a writer of its own. The C tests
(tests/test_io.c) cover the same ground from C.
"""

import numpy as np
import pysif
import pytest

N = 50


def values(n=N):
    i = np.arange(n, dtype=np.float64)
    return {name: 1000.0 * (k + 1) + 0.25 * i
            for k, name in enumerate(["x", "y", "z", "vx", "vy", "vz", "w"])}


def check(field, cols, n=N):
    want = values(n)
    assert field.n_particles == n
    for name, attr in [("x", "x"), ("y", "y"), ("z", "z"), ("vx", "vx"),
                       ("vy", "vy"), ("vz", "vz"), ("w", "weights")]:
        got = getattr(field, attr)
        if name in cols:
            np.testing.assert_array_equal(got, want[name].astype(got.dtype))
        else:
            assert got is None, f"{attr} should not have been read"


# --- ASCII ---

@pytest.fixture
def table(tmp_path):
    v = values()
    path = tmp_path / "table.txt"
    cols = np.column_stack([v[c] for c in ["x", "y", "z", "vx", "vy", "vz", "w"]])
    np.savetxt(path, cols, fmt="%.4f")
    return str(path)


@pytest.mark.parametrize("fmt", [
    "x y z vx vy vz w", "x,y,z,vx,vy,vz,w", "XYZVXVYVZW", "xyzvxvyvzw"])
def test_ascii_spellings(table, fmt):
    check(pysif.io.read_field_ascii(table, fmt), {"x", "y", "z", "vx", "vy", "vz", "w"})


def test_ascii_skips_and_defaults(table):
    check(pysif.io.read_field_ascii(table, "x y z * * * w"), {"x", "y", "z", "w"})
    # The documented default: no header line skipped.
    check(pysif.io.read_field_ascii(table), {"x", "y", "z"})


@pytest.mark.parametrize("fmt", [
    "xyzuvw", "x y z * * * m", "x y z x", "x y z v", "x y z q", "x y", "x y z *8"])
def test_ascii_bad_formats(table, fmt):
    with pytest.raises(ValueError):
        pysif.io.read_field_ascii(table, fmt)


# --- binary ---

def write_rows(path, dtype, header=b""):
    """One record per particle: numpy's structured array, written as is."""
    v = values()
    rec = np.zeros(N, dtype=dtype)
    for name in dtype.names:
        rec[name] = np.arange(N) if name == "id" else v[name]
    with open(path, "wb") as f:
        f.write(header)
        rec.tofile(f)


def write_blocks(path, names, float_type, header=b""):
    """One column at a time."""
    v = values()
    with open(path, "wb") as f:
        f.write(header)
        for name in names:
            if name == "id":
                np.arange(N, dtype="<i8").tofile(f)
            else:
                v[name].astype(float_type).tofile(f)


LAYOUT = ["x", "y", "z", "id", "vx", "vy", "vz", "w"]
FORMAT = "x y z *8 vx vy vz w"
ALL = {"x", "y", "z", "vx", "vy", "vz", "w"}


@pytest.mark.parametrize("precision, char", [("float32", "f4"), ("float64", "f8")])
@pytest.mark.parametrize("byteorder, prefix", [("little", "<"), ("big", ">"), ("native", "=")])
def test_binary_rows(tmp_path, precision, char, byteorder, prefix):
    path = tmp_path / "rows.bin"
    dtype = np.dtype([(n, "<i8" if n == "id" else prefix + char) for n in LAYOUT])
    write_rows(path, dtype, header=b"\xa5" * 24)
    field = pysif.io.read_field_binary(str(path), FORMAT, precision=precision,
                                       byteorder=byteorder, header_bytes=24)
    check(field, ALL)


@pytest.mark.parametrize("precision, char", [("float32", "f4"), ("float64", "f8")])
@pytest.mark.parametrize("byteorder, prefix", [("little", "<"), ("big", ">")])
def test_binary_blocks(tmp_path, precision, char, byteorder, prefix):
    path = tmp_path / "blocks.bin"
    write_blocks(path, LAYOUT, prefix + char, header=b"\xa5" * 8)
    field = pysif.io.read_field_binary(str(path), FORMAT, layout="blocks",
                                       precision=precision, byteorder=byteorder,
                                       header_bytes=8)
    check(field, ALL)


def test_binary_velocities_into_existing_field(tmp_path):
    path = tmp_path / "blocks.bin"
    write_blocks(path, LAYOUT, "<f4")
    field = pysif.io.read_field_binary(str(path), "x y z *8 * * * *",
                                       layout="blocks", byteorder="little")
    x = field.x  # a live view: adding columns must not move the positions
    again = pysif.io.read_field_binary(str(path), "* * * *8 vx vy vz *",
                                       layout="blocks", byteorder="little",
                                       field=field)
    assert again is field
    check(field, {"x", "y", "z", "vx", "vy", "vz"})
    np.testing.assert_array_equal(x, field.x)


def test_binary_n_particles(tmp_path):
    path = tmp_path / "rows.bin"
    write_rows(path, np.dtype([(n, "<i8" if n == "id" else "<f4") for n in LAYOUT]))
    field = pysif.io.read_field_binary(str(path), FORMAT, byteorder="little",
                                       n_particles=10)
    check(field, ALL, n=10)
    with pytest.raises(OSError):
        pysif.io.read_field_binary(str(path), FORMAT, byteorder="little",
                                   n_particles=N + 1)


@pytest.mark.parametrize("kwargs", [
    dict(header_bytes=4),                     # wrong header
    dict(precision="float64"),                # wrong precision
])
def test_binary_remainder_refused(tmp_path, kwargs):
    path = tmp_path / "rows.bin"
    write_rows(path, np.dtype([(n, "<i8" if n == "id" else "<f4") for n in LAYOUT]))
    with pytest.raises(OSError):
        pysif.io.read_field_binary(str(path), FORMAT, byteorder="little", **kwargs)


@pytest.mark.parametrize("kwargs, exc", [
    (dict(layout="columns"), ValueError),
    (dict(precision="float16"), ValueError),
    (dict(byteorder="network"), ValueError),
    (dict(format="x y z *0 w"), ValueError),
    (dict(format="x y z m"), ValueError),
    (dict(field="not a field"), TypeError),
    (dict(field=pysif.Field(), n_particles=3), ValueError),
])
def test_binary_bad_arguments(tmp_path, kwargs, exc):
    path = tmp_path / "rows.bin"
    write_rows(path, np.dtype([(n, "<i8" if n == "id" else "<f4") for n in LAYOUT]))
    kwargs.setdefault("format", FORMAT)
    with pytest.raises(exc):
        pysif.io.read_field_binary(str(path), **kwargs)


def test_binary_sorted_field_refused(tmp_path):
    path = tmp_path / "blocks.bin"
    write_blocks(path, LAYOUT, "<f4")
    field = pysif.io.read_field_binary(str(path), "x y z *8 * * * *",
                                       layout="blocks", byteorder="little")
    field.sort_morton()
    with pytest.raises(ValueError):
        pysif.io.read_field_binary(str(path), "* * * *8 * * * w",
                                   layout="blocks", byteorder="little", field=field)
