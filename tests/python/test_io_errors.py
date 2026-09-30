# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

"""What a failed read raises: the class, and the C library's reason as the
message.

The reason has to be in the exception itself. The C log goes to the process's
own stderr, which a notebook does not show, and the tests run it silenced --
so a message that only pointed at the log would carry nothing here.
"""

import shutil

import pysif
import pytest
from conftest import FITS, GADGET, HAS_FITS, requires_hdf5

requires_fits = pytest.mark.skipif(not HAS_FITS, reason="pysif built without FITS")


def text(tmp_path, body, name="t.txt"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def truncated(tmp_path, source, n_bytes, name):
    p = tmp_path / name
    p.write_bytes(source.read_bytes()[:n_bytes])
    return str(p)


# --- ASCII ---

@pytest.mark.parametrize("body, fmt, message", [
    ("1 2 3\n7 8\n", "x y z", r":2: 2 columns, format 'x y z' needs 3"),
    ("x y z\n1 2 3\n", "x y z", r":1: column 1 \(x\): 'x' is not a number"),
    ("1 2 3q\n", "x y z", r":1: column 3 \(z\): '3q' is not a number"),
    ("# only a comment\n", "x y z", r": no data rows"),
])
def test_ascii_field_says_where(tmp_path, body, fmt, message):
    path = text(tmp_path, body)
    with pytest.raises(OSError, match=message) as info:
        pysif.io.read_field_ascii(path, fmt)
    assert path in str(info.value)


def test_ascii_field_system_errors(tmp_path):
    missing = str(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError, match="missing.txt: No such file"):
        pysif.io.read_field_ascii(missing)
    with pytest.raises(IsADirectoryError):
        pysif.io.read_field_ascii(str(tmp_path))


def test_ascii_bad_format_is_a_value_error(tmp_path):
    path = text(tmp_path, "1 2 3\n")
    with pytest.raises(ValueError, match=r"format 'x y q': position 5: 'q'"):
        pysif.io.read_field_ascii(path, "x y q")


def test_ascii_catalogue_says_where(tmp_path):
    path = text(tmp_path, "#n=5\n1 2 3 4\n5 6 7 8\n")
    with pytest.raises(OSError, match=r": 2 rows, header says 5 \(truncated\?\)"):
        pysif.io.read_catalogue_ascii(path)
    path = text(tmp_path, "#n=1\n#cx cy cz r\n1 2 x 4\n")
    with pytest.raises(OSError, match=r":3: column 3 \(z\): 'x' is not a number"):
        pysif.io.read_catalogue_ascii(path)


# --- GADGET ---

def test_gadget_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        pysif.io.read_gadget(str(tmp_path / "snap_005"), length="kpc")


def test_gadget_truncated(tmp_path):
    path = truncated(tmp_path, GADGET / "f1_g4_u32" / "snap_005", 2000,
                     "snap_005")
    with pytest.raises(OSError, match=r"record 2 at byte \d+: .*\(truncated\)"):
        pysif.io.read_gadget(path, length="kpc")


def test_gadget_not_a_snapshot(tmp_path):
    path = text(tmp_path, "certainly not a snapshot\n", "snap_005")
    with pytest.raises(OSError, match="not a GADGET snapshot"):
        pysif.io.read_gadget(path, length="kpc")


# --- HDF5 ---

@requires_hdf5
def test_hdf5_catalogue_reasons(tmp_path):
    with pytest.raises(FileNotFoundError):
        pysif.io.read_catalogue_hdf5(str(tmp_path / "missing.h5"))
    with pytest.raises(OSError, match="not an HDF5 file"):
        pysif.io.read_catalogue_hdf5(text(tmp_path, "notes\n"))
    snapshot = str(GADGET / "hdf5_legacy" / "snap_005.hdf5")
    with pytest.raises(OSError, match=r"not a sif file .*GADGET snapshot\?"):
        pysif.io.read_catalogue_hdf5(snapshot)


@requires_hdf5
def test_hdf5_truncated(tmp_path):
    path = truncated(tmp_path, GADGET / "hdf5_legacy" / "snap_005.hdf5", 1500,
                     "s.hdf5")
    with pytest.raises(OSError, match=r"1500 bytes, superblock says \d+ \(truncated\)"):
        pysif.io.read_hdf5(path, x="PartType1/Coordinates[0]",
                           y="PartType1/Coordinates[1]",
                           z="PartType1/Coordinates[2]")


# --- FITS ---

@requires_fits
def test_fits_missing_column_lists_the_table(tmp_path):
    with pytest.raises(ValueError, match=r"column NOPE \(z\): missing; table has RA, DEC"):
        pysif.io.read_fits(str(FITS / "catalogue.fits"), ra="RA", dec="DEC",
                           z="NOPE")


@requires_fits
def test_fits_file_reasons(tmp_path):
    with pytest.raises(FileNotFoundError):
        pysif.io.read_fits(str(tmp_path / "missing.fits"), ra="RA", dec="DEC",
                           z="Z")
    path = truncated(tmp_path, FITS / "catalogue.fits", 4000, "cut.fits")
    with pytest.raises(OSError, match=r"4000 bytes, not a multiple of 2880 \(truncated\)"):
        pysif.io.read_fits(path, ra="RA", dec="DEC", z="Z")
    with pytest.raises(OSError, match="not a FITS file"):
        pysif.io.read_fits(text(tmp_path, "notes\n", "t.fits"), ra="RA",
                           dec="DEC", z="Z")


# --- everywhere ---

def test_no_message_points_at_the_log(tmp_path):
    """The reason is the message; nothing says to go and look for it."""
    calls = [
        lambda: pysif.io.read_field_ascii(text(tmp_path, "1 2\n")),
        lambda: pysif.io.read_catalogue_ascii(text(tmp_path, "#n=3\n")),
        lambda: pysif.io.read_gadget(text(tmp_path, "x" * 40, "g"), length="kpc"),
    ]
    for call in calls:
        with pytest.raises(Exception) as info:
            call()
        assert "log" not in str(info.value)
        assert str(info.value).strip()
