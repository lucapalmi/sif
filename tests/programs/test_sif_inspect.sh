#!/bin/sh
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# sif-inspect from the outside: the command line, and what it makes of each
# kind of file. What the checks find is the library's business (test_io,
# test_gadget, test_hdf5, test_fits); this checks the program around them.
#
#   test_sif_inspect.sh SIF_INSPECT DATA_DIR

set -u

EXE=$1
DATA=$2
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 1

failures=0
fail() {
  echo "FAIL: $*"
  failures=$((failures + 1))
}

# Run the program, keeping its exit status and what it said on stderr.
run() {
  "$EXE" "$@" >out.txt 2>err.txt
  status=$?
}

expect_status() {
  [ "$status" -eq "$1" ] || { fail "$2: exit $status, expected $1"; cat err.txt; }
}

expect_error() {
  grep -qF -- "$1" err.txt || { fail "$2: no \"$1\" in"; cat err.txt; }
}

# --- the command line -----------------------------------------------------

run --version
expect_status 0 "--version"
grep -q '^sif-inspect ' out.txt || fail "--version: $(cat out.txt)"

# What the build reads: HDF5 and FITS files only with their libraries, an
# .xfield only in the build's precision.
if grep -q 'without HDF5' out.txt; then
  HDF5=0
else
  HDF5=1
fi
if grep -q 'without FITS' out.txt; then
  FITS=0
else
  FITS=1
fi
if grep -q 'double precision' out.txt; then
  DOUBLE=1
else
  DOUBLE=0
fi

run --help
expect_status 0 "--help"
grep -q '^usage: sif-inspect' out.txt || fail "--help: $(cat out.txt)"

run
expect_status 2 "no FILE"

run --bogus x.txt
expect_status 2 "unknown option"

run a.txt b.txt
expect_status 2 "two files"
expect_error "one FILE only" "two files"

run --skip-header -1 a.txt
expect_status 2 "a negative --skip-header"

run --delimiter ab a.txt
expect_status 2 "a --delimiter of two characters"

run --format lua a.txt
expect_status 2 "--format lua"

# A configuration is sif-exodus's to check.
echo 'mode = "box"' >run.lua
run run.lua
expect_status 2 "a configuration"
expect_error "sif-exodus --check checks it" "a configuration"

# A one-particle .xfield, which is all the check reads of it without --deep:
# the 64-byte header (magic, version 2, one particle, a box of 1000, no
# weights or velocities, the precision, no checksum, padding), then
# x, y, z = 0.5 in little-endian order.
make_xfield() {
  z4='\000\000\000\000'
  {
    printf 'XFLD\002\000\000\000\001\000\000\000\000\000\000\000'
    printf '\000\000\000\000\000\100\217\100'
    printf "$z4$z4"
    if [ "$DOUBLE" -eq 1 ]; then printf '\001\000\000\000'; else printf "$z4"; fi
    printf "$z4$z4$z4$z4$z4$z4$z4"
    for _ in 1 2 3; do
      if [ "$DOUBLE" -eq 1 ]; then
        printf '\000\000\000\000\000\000\340\077'
      else
        printf '\000\000\000\077'
      fi
    done
  } >"$1"
}
make_xfield tracers.xfield

# --- each kind of file ------------------------------------------------------

expect_out() {
  grep -qF -- "$1" out.txt || { fail "$2: no \"$1\" in"; cat out.txt; }
}

# ASCII: told from its contents, checked row by row, every problem listed.
printf '# x y z w\n1 2 3 1\n4 5 6 1\n7 8 9 1\n' >good.txt
run good.txt
expect_status 0 "an ASCII file"
expect_out "rows          3" "an ASCII file"
expect_error "good.txt: OK" "an ASCII file"

printf 'x y z\n1 2 3\n4 5\n7 8 abc\n1,2,3\n' >bad.txt
run bad.txt
expect_status 2 "a malformed ASCII file"
expect_error "bad.txt:1: no number in the row (a header? skip 1 line)" "a header"
expect_error "bad.txt:3: 2 columns, line 1 has 3" "a short row"
expect_error "bad.txt:4: column 3: 'abc' is not a number" "not a number"
expect_error "(comma-separated? pass the delimiter ',')" "the delimiter hint"
expect_out "problems      5" "the problem count"

run --skip-header 1 --columns "x y z" bad.txt
expect_status 2 "against a format"
expect_error "bad.txt:3: 2 columns, format 'x y z' needs 3" "against a format"
run --skip-header 1 --columns "x y z" --delimiter , bad.txt
expect_status 2 "with a delimiter"
expect_error "'1 2 3' is not a number" "with a delimiter"

run missing.txt
expect_status 2 "a missing file"
expect_error "missing.txt: No such file or directory" "a missing file"

# GADGET, binary and multi-file, by its base name.
run "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "a snapshot"
expect_out "files read    3 of 3" "a snapshot"
run --deep "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "--deep a snapshot"
expect_out "1 x" "--deep a snapshot"

mkdir -p cut
for i in 0 1 2; do cp "$DATA/gadget/f1_legacy/snap_005.$i" cut/; done
head -c 2000 "$DATA/gadget/f1_legacy/snap_005.1" >cut/snap_005.1
run cut/snap_005
expect_status 2 "a truncated snapshot"
expect_error "cut/snap_005.1: record" "a truncated snapshot"
expect_out "files read    2 of 3" "a truncated snapshot"

# .xfield: the header, and with --deep every value.
run tracers.xfield
expect_status 0 "an .xfield"
expect_out "particles     1" "an .xfield"
# Its header carries no checksum, which only reading every value finds.
run --deep tracers.xfield
expect_status 2 "--deep an .xfield"
expect_error "header says 00000000 (corrupt)" "--deep an .xfield"
head -c 70 tracers.xfield >short.xfield
run short.xfield
expect_status 2 "a truncated .xfield"
expect_error "(truncated)" "a truncated .xfield"

if [ "$HDF5" -eq 1 ]; then
  H5="$DATA/gadget/hdf5_legacy/snap_005.hdf5"
  run "$H5"
  expect_status 0 "an HDF5 snapshot"
  expect_out "type         in file" "an HDF5 snapshot is checked as one"
  run --format hdf5 "$H5"
  expect_status 0 "--format hdf5"
  expect_out "/ParticleType1/Coordinates" "--format hdf5 lists the tree"
  expect_out "GADGET snapshot" "--format hdf5 says what it is"
  run --deep --format hdf5 "$H5"
  expect_status 0 "--deep --format hdf5"
  head -c 5000 "$H5" >cut.hdf5
  run cut.hdf5
  expect_status 2 "a truncated HDF5 file"
  expect_error "(truncated)" "a truncated HDF5 file"
fi

if [ "$FITS" -eq 1 ]; then
  run --deep "$DATA/fits/catalogue.fits"
  expect_status 0 "--deep a FITS file"
  expect_out "GALAXIES" "a FITS file"
  expect_out "ZNULL" "--deep ranges the columns"
  head -c 10000 "$DATA/fits/catalogue.fits" >cut.fits
  run cut.fits
  expect_status 2 "a truncated FITS file"
  expect_error "(truncated)" "a truncated FITS file"
fi

run --format xfield good.txt
expect_status 2 "--format that is not what the file is"
expect_error "not an .xfield file" "--format that is not what the file is"

run --columns "x y z" tracers.xfield
expect_status 2 "--columns on a file that is not ASCII"
expect_error "go with an ASCII file" "--columns on a file that is not ASCII"
run --format nonsense good.txt
expect_status 2 "an unknown --format"

if [ "$failures" -ne 0 ]; then
  echo "$failures failure(s)"
  exit 1
fi
echo "sif-inspect: all checks passed"
