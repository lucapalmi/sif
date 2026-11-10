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

run --format lua a.txt
expect_status 2 "--format lua"

# A check reads everything: there is no shallower one to deepen.
run --check --deep a.txt
expect_status 2 "--deep"
expect_error "unknown option '--deep'" "--deep"

# A configuration is sif-exodus's to check.
echo 'mode = "box"' >run.lua
run run.lua
expect_status 2 "a configuration"
expect_error "sif-exodus --check checks it" "a configuration"

# A one-particle .xfield, which is all a description reads of it:
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

# --- the description ------------------------------------------------------

expect_out() {
  grep -qF -- "$1" out.txt || { fail "$2: no \"$1\" in"; cat out.txt; }
}

# The same layout for every format: the file, then its metadata, then its
# tables. Only the formats this build reads are tried.
printf '#n=2\n#box=100\n#cx cy cz r\n1 2 3 4\n5 6 7 8\n' >cat.txt
run cat.txt
expect_status 0 "describe ASCII"
expect_out "format    ASCII, blank-separated" "describe ASCII"
expect_out "table, 2 rows, 4 columns" "describe ASCII"
grep -q '^  box  *100$' out.txt || fail "describe ASCII: no box: $(cat out.txt)"
grep -q '^  r  *number$' out.txt || fail "describe ASCII: no r: $(cat out.txt)"

run tracers.xfield
expect_status 0 "describe .xfield"
expect_out "table particles, 1 row, 3 columns" "describe .xfield"

run "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "describe a snapshot"
expect_out "files     3" "describe a snapshot"
expect_out "table PartType1, 100 rows" "describe a snapshot"
grep -q '^  POS  *float32\[3\]$' out.txt ||
  fail "describe a snapshot: no POS: $(cat out.txt)"

# A truncated file is described from what it says about itself; that it is
# cut short is --check's to say.
mkdir -p cut
for i in 0 1 2; do cp "$DATA/gadget/f1_legacy/snap_005.$i" cut/; done
head -c 2000 "$DATA/gadget/f1_legacy/snap_005.1" >cut/snap_005.1
run cut/snap_005
expect_status 0 "describe a truncated snapshot"

if [ "$HDF5" -eq 1 ]; then
  run --format hdf5 "$DATA/gadget/hdf5_legacy/snap_005.hdf5"
  expect_status 0 "describe a snapshot as HDF5"
  expect_out "Header/BoxSize" "describe a snapshot as HDF5"
  expect_out "table ParticleType0, 40 rows" "describe a snapshot as HDF5"
fi
if [ "$FITS" -eq 1 ]; then
  run "$DATA/fits/catalogue.fits"
  expect_status 0 "describe FITS"
  expect_out "table GALAXIES, 100 rows, 8 columns" "describe FITS"
  grep -q '^  POS  *float64\[3\]  *Mpc/h$' out.txt ||
    fail "describe FITS: no POS: $(cat out.txt)"
fi

run --format xfield cat.txt
expect_status 2 "describe as what it is not"
expect_error "not an .xfield file" "describe as what it is not"

run missing.txt
expect_status 2 "describe a missing file"
expect_error "missing.txt: No such file or directory" "describe a missing file"

head -c 300 "$EXE" >binary.dat
run binary.dat
expect_status 2 "describe binary data"
expect_error "not a format sif reads" "describe binary data"

# --- --check: each kind of file ---------------------------------------------

# The report is the description with each column's range, then the
# problems, then the result; the exit status is 2 for any error.
expect_result() {
  grep -q "^result    $1" out.txt || { fail "$2: no \"result    $1\" in"; cat out.txt; }
}

# ASCII: every row by the layout of the first, every problem with its line.
printf '# x y z w\n1 2 3 1\n4 5 6 1\n7 8 9 1\n' >good.txt
run --check good.txt
expect_status 0 "an ASCII file"
expect_result "sound" "an ASCII file"
grep -q '^  y  *number  *3  *0  *2  *8$' out.txt ||
  fail "an ASCII file: y not ranged: $(cat out.txt)"

printf 'x y z\n1 2 3\n4 5\n7 8 abc\n1,2,3\n' >bad.txt
run --check bad.txt
expect_status 2 "a malformed ASCII file"
expect_out "bad.txt:3: 2 fields (expected 3, as the first row)" "a short row"
expect_out "bad.txt:4: column 3 (z): 'abc' (expected a number)" "not a number"
expect_out "(comma-separated in places?)" "the delimiter hint"
expect_result "4 errors, 0 warnings" "the problem count"

# A catalogue: checked as one, and its radii have to be positive.
printf '#n=2\n#cx cy cz r\n1 2 3 4\n5 6 7 -1\n' >cat.txt
run --check cat.txt
expect_status 2 "a catalogue with a negative radius"
expect_out "checked as sif catalogue" "a catalogue with a negative radius"
expect_out "expected every radius > 0" "a catalogue with a negative radius"

run --check missing.txt
expect_status 2 "a missing file"
expect_out "missing.txt: No such file or directory" "a missing file"

# GADGET, binary and multi-file, by its base name: every block read.
run --check "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "a snapshot"
expect_result "sound" "a snapshot"
expect_out "checked as GADGET snapshot" "a snapshot"
grep -q '^  ID  *uint32  *100  *0  *1000000  *1000099$' out.txt ||
  fail "a snapshot: PartType1 IDs not ranged: $(cat out.txt)"

run --check cut/snap_005
expect_status 2 "a truncated snapshot"
expect_out "cut/snap_005.1: record" "a truncated snapshot"

# .xfield: the header says it carries a checksum of zero, which reading every
# value disproves.
run --check tracers.xfield
expect_status 2 "an .xfield without its checksum"
expect_out "(expected 00000000, from the header: corrupt)" "an .xfield"
head -c 70 tracers.xfield >short.xfield
run --check short.xfield
expect_status 2 "a truncated .xfield"
expect_out "truncated" "a truncated .xfield"

if [ "$HDF5" -eq 1 ]; then
  H5="$DATA/gadget/hdf5_legacy/snap_005.hdf5"
  run --check "$H5"
  expect_status 0 "an HDF5 snapshot"
  expect_out "checked as GADGET snapshot" "an HDF5 snapshot"
  run --check --format hdf5 "$H5"
  expect_status 0 "an HDF5 snapshot as HDF5"
  expect_result "sound" "an HDF5 snapshot as HDF5"
  head -c 5000 "$H5" >cut.hdf5
  run --check cut.hdf5
  expect_status 2 "a truncated HDF5 file"
  expect_out "(truncated)" "a truncated HDF5 file"
fi

if [ "$FITS" -eq 1 ]; then
  run --check "$DATA/fits/catalogue.fits"
  expect_status 0 "a FITS file"
  grep -q '^  ZNULL  *float64  *100  *10 ' out.txt ||
    fail "a FITS file: ZNULL's undefined values not counted: $(cat out.txt)"
  head -c 10000 "$DATA/fits/catalogue.fits" >cut.fits
  run --check cut.fits
  expect_status 2 "a truncated FITS file"
  expect_out "(truncated)" "a truncated FITS file"
fi

run --check --format xfield good.txt
expect_status 2 "--format that is not what the file is"
expect_out "not an .xfield file" "--format that is not what the file is"

run --format nonsense good.txt
expect_status 2 "an unknown --format"

if [ "$failures" -ne 0 ]; then
  echo "$failures failure(s)"
  exit 1
fi
echo "sif-inspect: all checks passed"
