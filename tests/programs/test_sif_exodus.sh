#!/bin/sh
# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# sif-exodus from the outside: the command line, the configuration file and
# one run end to end. What it computes is test_finders' business; this checks
# the program around it.
#
#   test_sif_exodus.sh SIF_EXODUS DATA_DIR

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
grep -q '^sif-exodus ' out.txt || fail "--version: $(cat out.txt)"

# A build without HDF5 refuses HDF5 outputs, the template's among them; an
# .xfield has to be in the build's precision.
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

# A one-particle .xfield, which is all --check reads of it: the 64-byte
# header (magic, version 2, one particle, a box of 1000, no weights or
# velocities, the precision, no checksum, padding), then x, y, z = 0.5 in
# little-endian order.
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

run
expect_status 2 "no CONFIG"

run --bogus x.lua
expect_status 2 "unknown option"

run -D novalue x.lua
expect_status 2 "-D without ="

# --- the template, and what --check makes of it ---------------------------

run --template
expect_status 0 "--template"
cp out.txt template.lua
if [ "$HDF5" -eq 0 ]; then
  run --check template.lua
  expect_status 2 "an HDF5 output without HDF5"
  expect_error "built without HDF5" "an HDF5 output without HDF5"
  sed 's/voids\.h5/voids.txt/' template.lua >t.lua && mv t.lua template.lua
fi

run --check template.lua
expect_status 2 "template without its input"
expect_error "input.path: cannot open tracers.xfield" "template without its input"

make_xfield tracers.xfield
run --check template.lua
expect_status 0 "template --check"
cp out.txt resolved.lua

# The resolved configuration is a configuration, and resolves to itself.
run --check resolved.lua
expect_status 0 "resolved --check"
tail -n +2 resolved.lua >a.txt
tail -n +2 out.txt >b.txt
cmp -s a.txt b.txt || fail "resolving the resolved configuration changed it"

# --- mistakes -------------------------------------------------------------

cat >typos.lua <<'EOF'
input = "tracers.xfield"
gird = { n_cells = 64 }
finder = { radii = { 1, 2 }, treshold = -0.7 }
output = "voids.txt"
EOF
run --check typos.lua
expect_status 2 "typos"
expect_error "gird: is not a section (did you mean grid?)" "typos"
expect_error "finder.treshold: unknown key (did you mean threshold?)" "typos"
expect_error "finder.threshold: missing" "typos"

echo 'io.open("x", "w")' >sandbox.lua
run --check sandbox.lua
expect_status 2 "io in the sandbox"
[ ! -e x ] || fail "the sandbox let a file be created"

echo 'os.execute("touch y")' >sandbox.lua
run --check sandbox.lua
expect_status 2 "os.execute in the sandbox"
[ ! -e y ] || fail "the sandbox let a command run"

echo 'finder = { radii = ladder(10, 5, 0.05) }' >helper.lua
run --check helper.lua
expect_status 2 "a helper's argument check"
expect_error "helper.lua:1: bad argument #2 to 'ladder'" "a helper's argument check"

run -D input=x --check template.lua
expect_status 2 "-D naming a section"

# --- a run --------------------------------------------------------------

# The legacy binary GADGET fixture: 100 type-1 particles in a 10 Mpc/h box,
# found with -D the way a batch job would.
cat >run.lua <<EOF
input = {
  path = "$DATA/gadget/f1_legacy/snap_005",
  format = "gadget",
  gadget = { snapformat = 1, ptype = tonumber(ptype), length = "kpc" },
}
grid = { n_cells = 16 }
finder = { radii = { 2, 1.5 }, threshold = -0.7 }
output = "voids.txt"
run = { log_level = "warning" }
EOF
run -D ptype=1 run.lua
expect_status 0 "a run"
[ -s voids.txt ] || fail "a run: no catalogue written"

# Run again over the catalogue just written: said, and done.
cp voids.txt previous.txt
run -D ptype=1 --check run.lua
expect_status 0 "--check over an existing output"
expect_error "voids.txt exists, and will be replaced" "an existing output"
expect_error "memory: about" "the memory estimate"
expect_error "input: 100 unweighted tracers, box 10" "what the header says"

# A run that fails leaves an existing output as it was, and nothing beside
# it: the finder refuses a radius too large for the box only once the data
# is read, so the pre-flight check has to be the one catching it here.
sed 's/radii = { 2, 1.5 }/radii = { 9, 1.5 }/' run.lua >toobig.lua
run -D ptype=1 toobig.lua
expect_status 2 "a radius wider than the box"
expect_error "needs a search sphere wider than the box" "a radius wider than the box"
cmp -s voids.txt previous.txt || fail "a refused run changed the output"
ls voids.txt.tmp.* >/dev/null 2>&1 && fail "a refused run left a temporary"

run -D ptype=4 --check run.lua
expect_status 2 "a type the snapshot does not have"
expect_error "has no particles of type 4" "a type the snapshot does not have"

# --- any HDF5 file ----------------------------------------------------------

# The HDF5 GADGET fixture holds the particles of the binary one: read as a
# plain HDF5 file, through its datasets, it gives the same voids.
cat >h5.lua <<EOF
input = {
  path = "$DATA/gadget/hdf5_legacy/snap_005.hdf5",
  format = "hdf5",
  box_length = 10,
  hdf5 = {
    columns = {
      x = "ParticleType1/Coordinates[0]",
      y = "ParticleType1/Coordinates[1]",
      z = "ParticleType1/Coordinates[2]",
    },
    length_scale = 1e-3,
  },
}
grid = { n_cells = 16 }
finder = { radii = { 2, 1.5 }, threshold = -0.7 }
output = "h5_voids.txt"
run = { log_level = "warning" }
EOF
if [ "$HDF5" -eq 1 ]; then
  run h5.lua
  expect_status 0 "an HDF5 run"
  sed 's/voids.txt/gadget_voids.txt/' run.lua >gadget.lua
  run -D ptype=1 gadget.lua
  grep -v '^#' h5_voids.txt >a.txt
  grep -v '^#' gadget_voids.txt >b.txt
  [ -s a.txt ] && cmp -s a.txt b.txt ||
    fail "the snapshot read as plain HDF5 gave other voids than as GADGET"

  run --check h5.lua
  expect_status 0 "HDF5 --check"
  cp out.txt h5_resolved.lua
  run --check h5_resolved.lua
  tail -n +2 h5_resolved.lua >a.txt
  tail -n +2 out.txt >b.txt
  cmp -s a.txt b.txt || fail "resolving a resolved HDF5 configuration changed it"
fi

cat >h5_bad.lua <<EOF
input = { path = "$DATA/gadget/hdf5_legacy/snap_005.hdf5", format = "hdf5",
  box_length = 10 }
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check h5_bad.lua
expect_status 2 "an hdf5 input without datasets"
expect_error "input.hdf5: missing" "an hdf5 input without datasets"

# --- a FITS catalogue ------------------------------------------------------

# The FITS fixture: 100 rows, whole and split over two files. Its columns
# are no survey's, only numbers to read: POS is a vector column, and the
# weight an expression over two others.
cat >fits.lua <<EOF
input = {
  path = { "$DATA/fits/part_a.fits", "$DATA/fits/part_b.fits" },
  box_length = 400,
  fits = {
    columns = { x = "POS[1]", y = "RA", z = "POS[3]", w = "W1 * W2" },
    where = "Z < 0.5",
  },
}
grid = { n_cells = 16 }
finder = { radii = { 80, 60 }, threshold = -0.7 }
output = "fits_voids.txt"
run = { log_level = "warning" }
EOF
if [ "$FITS" -eq 1 ]; then
  run fits.lua
  expect_status 0 "a FITS run"
  [ -s fits_voids.txt ] || fail "a FITS run: no catalogue written"

  # The two halves are the whole catalogue: the same voids from either.
  sed "s|{ \"$DATA/fits/part_a.fits\", \"$DATA/fits/part_b.fits\" }|\"$DATA/fits/catalogue.fits\"|; s|fits_voids|whole_voids|" \
    fits.lua >whole.lua
  run whole.lua
  expect_status 0 "a FITS run on the whole file"
  # The voids, not the header: that records which files were read.
  grep -v '^#' fits_voids.txt >a.txt
  grep -v '^#' whole_voids.txt >b.txt
  cmp -s a.txt b.txt ||
    fail "the halves and the whole file gave different catalogues"
  grep -q '^#finder="exodus"' fits_voids.txt ||
    fail "an ASCII catalogue does not record the run's settings"

  run --check fits.lua
  expect_status 0 "FITS --check"
  cp out.txt fits_resolved.lua
  run --check fits_resolved.lua
  tail -n +2 fits_resolved.lua >a.txt
  tail -n +2 out.txt >b.txt
  cmp -s a.txt b.txt || fail "resolving a resolved FITS configuration changed it"

  # A FITS catalogue out, with the run's settings in its primary header.
  sed "s|fits_voids.txt|fits_voids.fits|" fits.lua >fits_out.lua
  run fits_out.lua
  expect_status 0 "a FITS output"
  grep -aq "VOIDS" fits_voids.fits || fail "a FITS output: no VOIDS table"
  grep -aq "HIERARCH SEARCH_FACTOR" fits_voids.fits ||
    fail "a FITS output: the settings are not in the header"
  run --check fits_out.lua
  grep -q 'format = "fits"' out.txt || fail "a FITS output: not resolved as fits"

  sed "s|part_b.fits|no_such_part.fits|" fits.lua >missing.lua
  run --check missing.lua
  expect_status 2 "a missing FITS file"
  expect_error "no_such_part.fits cannot be read as a FITS file" "a missing FITS file"
else
  run --check fits.lua
  expect_status 2 "FITS without cfitsio"
  expect_error "built without FITS support" "FITS without cfitsio"
fi

# Mistakes a FITS configuration can make, found whatever the build.
cat >fits_bad.lua <<EOF
input = {
  path = "$DATA/fits/catalogue.fits",
  fits = { columns = { x = "RA", y = "DEC", vx = "RA" } },
}
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check fits_bad.lua
expect_status 2 "FITS mistakes"
expect_error "input.fits.columns.vx: velocities are not read" "FITS mistakes"
expect_error "input.fits.columns.z: missing" "FITS mistakes"
expect_error "input.box_length: missing" "FITS mistakes"

cat >list_bad.lua <<EOF
input = {
  path = { "a", "b" },
  format = "gadget",
}
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check list_bad.lua
expect_status 2 "a list of files for GADGET"
expect_error "input.path: a list of files, which only the fits and hdf5 formats" \
  "a list of files for GADGET"

if [ "$failures" -ne 0 ]; then
  echo "$failures failure(s)"
  exit 1
fi
echo "sif-exodus: all checks passed"
