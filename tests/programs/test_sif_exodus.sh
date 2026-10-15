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
expect_status 2 "--template without a mode"
expect_error "--template takes the mode" "--template without a mode"

run --template survey
expect_status 0 "--template survey"
grep -q '^mode = "survey"' out.txt || fail "--template survey: not a survey"

run --template box
expect_status 0 "--template box"
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
mode = "box"
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

# A key an older sif-exodus took is told apart from a typo.
cat >units.lua <<'EOF'
mode = "box"
input = "tracers.xfield"
finder = { radii = { 1, 2 }, radii_units = "mps", threshold = -0.7 }
output = "voids.txt"
EOF
run --check units.lua
expect_status 2 "a removed key"
expect_error "finder.radii_units: removed: radii are in the positions' own" \
  "a removed key"
if grep -q "unknown key" err.txt; then
  fail "a removed key reported as unknown"
fi

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
mode = "box"
input = {
  path = "$DATA/gadget/f1_legacy/snap_005",
  format = "gadget",
  snapformat = 1,
  ptype = tonumber(ptype),
  length = "kpc",
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
mode = "box"
input = {
  path = "$DATA/gadget/hdf5_legacy/snap_005.hdf5",
  format = "hdf5",
  box_length = 10,
  columns = {
    x = "ParticleType1/Coordinates[0]",
    y = "ParticleType1/Coordinates[1]",
    z = "ParticleType1/Coordinates[2]",
  },
  length_scale = 1e-3,
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
mode = "box"
input = { path = "$DATA/gadget/hdf5_legacy/snap_005.hdf5", format = "hdf5",
  box_length = 10 }
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check h5_bad.lua
expect_status 2 "an hdf5 input without datasets"
expect_error "input.columns: missing" "an hdf5 input without datasets"

# --- a FITS catalogue ------------------------------------------------------

# The FITS fixture: 100 rows, whole and split over two files. Its columns
# are no survey's, only numbers to read: POS is a vector column, and the
# weight an expression over two others.
cat >fits.lua <<EOF
mode = "box"
input = {
  path = { "$DATA/fits/part_a.fits", "$DATA/fits/part_b.fits" },
  box_length = 400,
  columns = { x = "POS[1]", y = "RA", z = "POS[3]", w = "W1 * W2" },
  where = "Z < 0.5",
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
mode = "box"
input = {
  path = "$DATA/fits/catalogue.fits",
  columns = { x = "RA", y = "DEC", vx = "RA" },
  delimiter = ",",
}
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check fits_bad.lua
expect_status 2 "FITS mistakes"
expect_error "input.columns.vx: velocities are not read" "FITS mistakes"
expect_error "input.columns.z: missing" "FITS mistakes"
expect_error "input.delimiter: a setting of ascii files, and this is fits" \
  "FITS mistakes"
expect_error "input.box_length: missing" "FITS mistakes"

cat >list_bad.lua <<EOF
mode = "box"
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

# --- the mode --------------------------------------------------------------

cat >nomode.lua <<'EOF'
input = "tracers.xfield"
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check nomode.lua
expect_status 2 "no mode"
expect_error 'mode: missing: mode = "box"' "no mode"

# A key of the other mode is named as such.
cat >crossed.lua <<'EOF'
mode = "box"
input = { path = "tracers.xfield", random = "r.xfield" }
mesh = { n_cells_data = 16 }
finder = { radii = { 1 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check crossed.lua
expect_status 2 "survey keys in a box"
expect_error 'input.random: a setting of mode = "survey"' "survey keys in a box"
expect_error 'mesh.n_cells_data: a setting of mode = "survey"' "survey keys in a box"

# --- a survey ---------------------------------------------------------------

# A patch of sky, 40 by 30 degrees at redshift 0.1 to 0.2, uniformly filled:
# data and randoms alike, five times as many randoms. awk's generator makes
# it; the voids are whatever it gives, only their shape is checked.
sky_points() {
  awk -v n="$1" -v seed="$2" 'BEGIN {
    srand(seed); pi = atan2(0, -1); top = sin(30 * pi / 180)
    for (i = 0; i < n; i++) {
      s = top * rand()
      printf "%.6f %.6f %.6f\n", 150 + 40 * rand(),
        atan2(s, sqrt(1 - s * s)) * 180 / pi, 0.1 + 0.1 * rand()
    }
  }'
}
sky_points 3000 1 >sky_data.txt
sky_points 15000 2 >sky_randoms.txt

cat >survey.lua <<'EOF'
mode = "survey"
input = {
  cosmology = { omega_m = 0.31 },
  data = { path = "sky_data.txt", format = "ascii" },
  random = { path = "sky_randoms.txt", format = "ascii" },
}
finder = { radii = { 45, 35 }, threshold = -0.7 }
output = "survey_voids.txt"
run = { log_level = "warning" }
EOF
run survey.lua
expect_status 0 "a survey"
grep -q '^#mode="survey"' survey_voids.txt ||
  fail "a survey: the catalogue does not record its mode"
grep -q '^#ra dec z r footprint footprint_shell$' survey_voids.txt ||
  fail "a survey on the sky: the voids are not on the sky, with footprints"

run --check survey.lua
expect_status 0 "survey --check"
expect_error "data: 3000 unweighted tracers; randoms: 15000" "survey --check"
expect_error "memory: about" "survey --check"
grep -q 'coordinates = "sky"' out.txt || fail "a survey: sky is not the default"
cp out.txt survey_resolved.lua
run --check survey_resolved.lua
tail -n +2 survey_resolved.lua >a.txt
tail -n +2 out.txt >b.txt
cmp -s a.txt b.txt || fail "resolving a resolved survey configuration changed it"

# A survey in Cartesian coordinates: positions in, positions out, and no
# cosmology. The patch's numbers, read as positions, make a slab.
awk '{ print $1 - 150, 10 * $2, 1000 * $3 }' sky_data.txt >xyz_data.txt
awk '{ print $1 - 150, 10 * $2, 1000 * $3 }' sky_randoms.txt >xyz_randoms.txt
cat >cartesian.lua <<'EOF'
mode = "survey"
input = {
  coordinates = "cartesian",
  data = { path = "xyz_data.txt", format = "ascii", columns = "x y z" },
  random = { path = "xyz_randoms.txt", format = "ascii", columns = "x y z" },
}
mesh = { n_cells_data = 16, n_cells_random = 20 }
finder = { radii = { 25, 20 }, threshold = -0.7 }
output = "cartesian_voids.txt"
run = { log_level = "warning" }
EOF
run cartesian.lua
expect_status 0 "a Cartesian survey"
grep -q '^#cx cy cz r footprint footprint_shell$' cartesian_voids.txt ||
  fail "a Cartesian survey: the voids are not in Cartesian coordinates"
grep -q '^#random_mesh_n_cells=20$' cartesian_voids.txt ||
  fail "a Cartesian survey: mesh.n_cells_random was not used"

# A mesh coarser than the search sphere breaks the padding the box has.
sed 's/n_cells_data = 16/n_cells_data = 4/' cartesian.lua >coarse.lua
run coarse.lua
expect_status 1 "a mesh too coarse for the survey box"
expect_error "mesh.n_cells_data = 4 makes cells of" "a mesh too coarse for the survey box"
run --check coarse.lua
expect_status 2 "--check: a mesh too coarse for the survey box"
expect_error "mesh.n_cells_data = 4 makes cells of" "a mesh too coarse for the survey box"

# The slab is no lightcone around the origin, so its volume is not
# measurable from the randoms, and they say so; given, it is taken as it is,
# and nothing is measured.
run --check cartesian.lua
expect_error "radial range differs across the sky" "a slab measured"
awk '/^mesh = /{ print "input.footprint_volume = 2.5e6" } { print }' \
  cartesian.lua >given.lua
run --check given.lua
expect_status 0 "a given footprint volume"
expect_error "footprint: 2.5e+06 (Mpc/h)^3 (as given)" "a given footprint volume"
if grep -q "radial range differs" err.txt; then
  fail "a given footprint volume was measured anyway"
fi
grep -q "footprint_volume = 2500000" out.txt ||
  fail "a given footprint volume is missing from the resolved configuration"

# The observer: Cartesian only, three finite numbers, and kept.
awk '/^mesh = /{ print "input.observer = { 1, 2, 3 }" } { print }' \
  cartesian.lua >observer.lua
run --check observer.lua
expect_status 0 "an observer"
grep -q "observer = { 1, 2, 3 }" out.txt ||
  fail "the observer is missing from the resolved configuration"
awk '/^mesh = /{ print "input.observer = { 1, 2 }" } { print }' \
  cartesian.lua >observer_bad.lua
run --check observer_bad.lua
expect_status 2 "an observer of two numbers"
expect_error "input.observer: must be three numbers" "an observer of two numbers"
awk '/^finder = /{ print "input.observer = { 1, 2, 3 }" } { print }' \
  survey.lua >observer_sky.lua
run --check observer_sky.lua
expect_status 2 "an observer on the sky"
expect_error "input.observer: not used: sky coordinates put the observer" \
  "an observer on the sky"
awk '/^mesh = /{ print "input.footprint_volume = -1" } { print }' \
  cartesian.lua >volume_bad.lua
run --check volume_bad.lua
expect_status 2 "a negative footprint volume"
expect_error "input.footprint_volume: must be positive, got -1" \
  "a negative footprint volume"

# A grid coarser than the smallest radius, which the survey box does not
# allow for: refused before the run, by the run and by --check alike.
awk '/^mesh = /{ print "grid = { n_cells = 8 }" } { print }' cartesian.lua \
  >coarse_grid.lua
run coarse_grid.lua
expect_status 1 "a grid too coarse for the survey box"
expect_error "grid.n_cells = 8: cells of" "a grid too coarse for the survey box"
run --check coarse_grid.lua
expect_status 2 "--check: a grid too coarse for the survey box"
expect_error "grid.n_cells = 8: cells of" "a grid too coarse for the survey box"

# What a survey configuration can get wrong.
cat >survey_bad.lua <<'EOF'
mode = "survey"
input = {
  path = "sky_data.txt",
  box_length = 100,
  data = { path = "sky_data.txt", format = "ascii", columns = "x y z" },
  random = { path = "r.xfield" },
}
mesh = { n_cells = 16 }
finder = { radii = { 40 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check survey_bad.lua
expect_status 2 "survey mistakes"
expect_error 'input.box_length: a setting of mode = "box"' "survey mistakes"
expect_error "input.path: a survey reads two inputs" "survey mistakes"
expect_error 'mesh.n_cells: a setting of mode = "box"' "survey mistakes"
expect_error "input.cosmology: missing" "survey mistakes"
expect_error 'input.coordinates is "sky": read ra dec z' "survey mistakes"
expect_error "input.random.format: xfield files hold positions" "survey mistakes"

cat >box_sky.lua <<'EOF'
mode = "box"
input = { path = "d.txt", format = "ascii", box_length = 10,
  columns = "ra dec z" }
finder = { radii = { 4 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check box_sky.lua
expect_status 2 "sky coordinates in a box"
expect_error "which only a survey does" "sky coordinates in a box"

# --- --check on a data file ------------------------------------------------

expect_out() {
  grep -qF -- "$1" out.txt || { fail "$2: no \"$1\" in"; cat out.txt; }
}

# ASCII: told from its contents, checked row by row, every problem listed.
printf '# x y z w\n1 2 3 1\n4 5 6 1\n7 8 9 1\n' >good.txt
run --check good.txt
expect_status 0 "--check an ASCII file"
expect_out "rows          3" "--check an ASCII file"
expect_error "good.txt: OK" "--check an ASCII file"

printf 'x y z\n1 2 3\n4 5\n7 8 abc\n1,2,3\n' >bad.txt
run --check bad.txt
expect_status 2 "--check a malformed ASCII file"
expect_error "bad.txt:1: no number in the row (a header? skip 1 line)" "a header"
expect_error "bad.txt:3: 2 columns, line 1 has 3" "a short row"
expect_error "bad.txt:4: column 3: 'abc' is not a number" "not a number"
expect_error "(comma-separated? pass the delimiter ',')" "the delimiter hint"
expect_out "problems      5" "the problem count"

run --check --skip-header 1 --columns "x y z" bad.txt
expect_status 2 "--check against a format"
expect_error "bad.txt:3: 2 columns, format 'x y z' needs 3" "--check against a format"
run --check --skip-header 1 --columns "x y z" --delimiter , bad.txt
expect_status 2 "--check with a delimiter"
expect_error "'1 2 3' is not a number" "--check with a delimiter"

run --check missing.txt
expect_status 2 "--check a missing file"
expect_error "missing.txt: No such file or directory" "--check a missing file"

# GADGET, binary and multi-file, by its base name.
run --check "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "--check a snapshot"
expect_out "files read    3 of 3" "--check a snapshot"
run --check --deep "$DATA/gadget/f1_legacy/snap_005"
expect_status 0 "--check --deep a snapshot"
expect_out "1 x" "--check --deep a snapshot"

mkdir -p cut
for i in 0 1 2; do cp "$DATA/gadget/f1_legacy/snap_005.$i" cut/; done
head -c 2000 "$DATA/gadget/f1_legacy/snap_005.1" >cut/snap_005.1
run --check cut/snap_005
expect_status 2 "--check a truncated snapshot"
expect_error "cut/snap_005.1: record" "--check a truncated snapshot"
expect_out "files read    2 of 3" "--check a truncated snapshot"

# .xfield: the header, and with --deep every value.
run --check tracers.xfield
expect_status 0 "--check an .xfield"
expect_out "particles     1" "--check an .xfield"
# Its header carries no checksum, which only reading every value finds.
run --check --deep tracers.xfield
expect_status 2 "--check --deep an .xfield"
expect_error "header says 00000000 (corrupt)" "--check --deep an .xfield"
head -c 70 tracers.xfield >short.xfield
run --check short.xfield
expect_status 2 "--check a truncated .xfield"
expect_error "(truncated)" "--check a truncated .xfield"

if [ "$HDF5" -eq 1 ]; then
  H5="$DATA/gadget/hdf5_legacy/snap_005.hdf5"
  run --check "$H5"
  expect_status 0 "--check an HDF5 snapshot"
  expect_out "type         in file" "an HDF5 snapshot is checked as one"
  run --check --format hdf5 "$H5"
  expect_status 0 "--check --format hdf5"
  expect_out "/ParticleType1/Coordinates" "--format hdf5 lists the tree"
  expect_out "GADGET snapshot" "--format hdf5 says what it is"
  run --check --deep --format hdf5 "$H5"
  expect_status 0 "--check --deep --format hdf5"
  head -c 5000 "$H5" >cut.hdf5
  run --check cut.hdf5
  expect_status 2 "--check a truncated HDF5 file"
  expect_error "(truncated)" "--check a truncated HDF5 file"
fi

if [ "$FITS" -eq 1 ]; then
  run --check --deep "$DATA/fits/catalogue.fits"
  expect_status 0 "--check --deep a FITS file"
  expect_out "GALAXIES" "--check a FITS file"
  expect_out "ZNULL" "--deep ranges the columns"
  head -c 10000 "$DATA/fits/catalogue.fits" >cut.fits
  run --check cut.fits
  expect_status 2 "--check a truncated FITS file"
  expect_error "(truncated)" "--check a truncated FITS file"
fi

run --check --format xfield good.txt
expect_status 2 "--format that is not what the file is"
expect_error "not an .xfield file" "--format that is not what the file is"

run --check --columns "x y z" tracers.xfield
expect_status 2 "--columns on a file that is not ASCII"
expect_error "go with an ASCII file" "--columns on a file that is not ASCII"
run --deep template.lua
expect_status 2 "--deep without --check"
run --check --format nonsense good.txt
expect_status 2 "an unknown --format"

# A configuration with --deep reads its input, every row of it.
cat >deep.lua <<'EOF'
mode = "box"
input = { path = "bad.txt", format = "ascii", columns = "x y z",
  skip_header = 1, box_length = 10 }
finder = { radii = { 2 }, threshold = -0.7 }
output = "voids.txt"
EOF
run --check deep.lua
expect_status 0 "a configuration without --deep does not read its input"
run --check --deep deep.lua
expect_status 2 "a configuration with --deep"
expect_error "bad.txt:3: 2 columns, format 'x y z' needs 3" "--deep, row 3"
expect_error "bad.txt:4: column 3 (z): 'abc' is not a number" "--deep, row 4"

# --- checkpoints and prior catalogues ----------------------------------------

# A box with three spheres emptied out of it, so that there are voids at
# several rungs: awk's generator again, rejecting what falls in a hole.
awk 'BEGIN {
  srand(7); n = 0
  while (n < 30000) {
    x = 100 * rand(); y = 100 * rand(); z = 100 * rand()
    if ((x-25)^2 + (y-25)^2 + (z-25)^2 < 225 ||
        (x-70)^2 + (y-40)^2 + (z-60)^2 < 144 ||
        (x-40)^2 + (y-75)^2 + (z-80)^2 < 81) continue
    printf "%.5f %.5f %.5f\n", x, y, z; n++
  }
}' >holes.txt
cat >holes.lua <<'EOF'
mode = "box"
input = { path = "holes.txt", format = "ascii", box_length = 100 }
grid = { n_cells = 32 }
finder = {
  radii = radii == "large" and { 15, 12 } or radii == "small" and { 10, 8 }
    or { 15, 12, 10, 8 },
  threshold = -0.7,
}
output = out or "holes_full.txt"
prior = from
run = { log_level = "warning" }
EOF
voids() { grep -v '^#' "$1"; }

run holes.lua
expect_status 0 "the reference run"
[ ! -e holes_full.txt.ckpt ] || fail "a completed run left its checkpoint"
[ "$(voids holes_full.txt | wc -l)" -ge 3 ] || fail "the reference run: too few voids"

# The large radii, then the small ones from there: the single run's voids.
run -D radii=large -D out=holes_large.txt holes.lua
expect_status 0 "the large radii"
run -D radii=small -D out=holes_ext.txt -D from=holes_large.txt holes.lua
expect_status 0 "the small radii, from the large"
voids holes_full.txt >a.txt
voids holes_ext.txt >b.txt
cmp -s a.txt b.txt || fail "large then small radii differ from one run"
grep -q '^#prior="holes_large.txt"$' holes_ext.txt &&
  grep -q '^#n_prior=' holes_ext.txt ||
  fail "an extended catalogue does not record its prior"

run -D from=missing.txt --check holes.lua
expect_status 2 "a prior that is not there"
expect_error "prior.path: cannot open missing.txt" "a prior that is not there"

# A catalogue that cannot be written keeps the checkpoint, which --check
# reads and the same command, given somewhere to write, finishes from.
mkdir blocked.txt
run -D out=blocked.txt holes.lua
expect_status 1 "an output that cannot be written"
expect_error "blocked.txt.ckpt: kept" "an output that cannot be written"
[ -e blocked.txt.ckpt ] || fail "a run that could not write lost its checkpoint"

run --check blocked.txt.ckpt
expect_status 0 "--check on a checkpoint"
grep -q '^progress      all 4 rungs done$' out.txt ||
  fail "--check on a checkpoint: $(cat out.txt)"
grep -q '^threshold     -0.7$' out.txt || fail "--check on a checkpoint: threshold"

run -D out=blocked.txt --check holes.lua
expect_status 0 "--check with a checkpoint to resume"
expect_error "blocked.txt.ckpt exists: the run resumes from it, with 4 of 4" \
  "--check with a checkpoint to resume"

sed 's/threshold = -0.7/threshold = -0.6/' holes.lua >holes6.lua
run -D out=blocked.txt holes6.lua
expect_status 2 "another run's checkpoint"
expect_error "is not this run's: threshold -0.7 (this run's: -0.6)" \
  "another run's checkpoint"

run -D out=holes_resumed.txt --checkpoint blocked.txt.ckpt holes.lua
expect_status 0 "resumed from the checkpoint"
voids holes_resumed.txt >b.txt
cmp -s a.txt b.txt || fail "the resumed run differs from the reference"
[ ! -e blocked.txt.ckpt ] || fail "a resumed run left its checkpoint"
rmdir blocked.txt

run -D out=nock.txt --no-checkpoint --check holes.lua
expect_status 0 "--no-checkpoint"
grep -q '^  checkpoint = false,$' out.txt || fail "--no-checkpoint: not resolved"
run --checkpoint a.ckpt --no-checkpoint holes.lua
expect_status 2 "--checkpoint with --no-checkpoint"
run --checkpoint a.ckpt --check holes.txt
expect_status 2 "--checkpoint with a data file"

# On the sky: a prior written on the sky is taken off it with the run's
# cosmology and goes back out as it came. Not compared with one run over the
# whole ladder, as the box is: a survey's padded box and grid are sized from
# the ladder's ends, so the two sessions are not searched on the same grid.
for part in large small; do
  case $part in
    large) radii="35, 30"; extra="" ;;
    small) radii="26"; extra='prior = "sky_large.txt"' ;;
  esac
  sed "s/radii = { 45, 35 }/radii = { $radii }/; s/survey_voids.txt/sky_$part.txt/" \
    survey.lua >sky_$part.lua
  echo "$extra" >>sky_$part.lua
  run sky_$part.lua
  expect_status 0 "a survey on the sky, $part radii"
done
n_large=$(voids sky_large.txt | wc -l)
[ "$n_large" -gt 0 ] || fail "a survey on the sky: no voids at the large radii"
voids sky_large.txt | cut -d' ' -f1-4 >a.txt
voids sky_small.txt | head -n "$n_large" | cut -d' ' -f1-4 >b.txt
cmp -s a.txt b.txt || fail "the prior's voids did not come back as they went in"
[ "$(voids sky_small.txt | wc -l)" -gt "$n_large" ] ||
  fail "a survey extended from the sky added no voids"

if [ "$failures" -ne 0 ]; then
  echo "$failures failure(s)"
  exit 1
fi
echo "sif-exodus: all checks passed"
