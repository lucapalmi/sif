# Running exodus

This page shows how to run the exodus void finder using **sif**, in
three ways that do the same thing: a Python script, a C program, and the
`sif-exodus` command, which runs the whole pipeline from a configuration
file.

## exodus for N-body simulations

:::::{tab-set}

::::{tab-item} Python
```python
import numpy as np
import pysif

# initialize the library
# log_level 2 (info, the default) prints one line for each rung of
# the finder; 0 (trace) prints everything the library does, and
# higher levels print out only warnings and errors
# in this case, we don't need further configuration
pysif.init(log_level=2)

# first, load the data into a pysif.Field structure
# in this example, we consider a Gadget snapshot, and read its
# dark matter particles (ptype=1)
# with this command, we only load the particle positions
# 'length=' is the unit the snapshot is written in: 'auto' (the
# default) reads it from an HDF5 snapshot, but a binary one does not
# record it, so it has to be named: 'kpc' for kpc/h (Gadget's
# default) or 'mpc' for Mpc/h. positions and box_length always come
# out in Mpc/h
# the argument 'fraction=' allows for subsampling ('seed=' makes it
# reproducible)
field, box_length = pysif.io.read_gadget(
    "path/to/snapshot", ptype=1, length="mpc"
)

# this instruction corrects for possible rounding errors
# in the particle positions, ensuring that they all lay inside the box
# (subsequent functions will fail if this is not the case)
field.wrap(box_length=box_length)

# now, we construct the grid and assign the particles with
# the CiC scheme
n_cells = 512                     # change at will
grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
grid.assign_cic(field=field)

# exodus requires that the grid is expressed in density contrast
grid.to_density_contrast()

# now we construct the chain mesh, which is used by exodus to
# query the particles positions more efficiently
#
# first, let's obtain the optimal number of cells
# this operations requires the maximum search radius,
# so we construct the array here. use the pysif.real datatype
# to ensure compatibility
# the smallest radius should span at least two grid cells: below
# that, exodus warns that its results are dominated by noise
search_radii = np.geomspace(2 * box_length / n_cells, 50, 100, dtype=pysif.real)

n_cells_mesh = pysif.finders.suggest_mesh_cells(
    n_particles=field.n_particles,
    box_length=box_length,
    max_radius=float(search_radii.max())
)

# now the chain mesh. to save memory, we make the chain mesh constructor
# destroy the field (freeing its memory), because in this case we don't
# need it anymore. also, we don't need the particle indices
mesh = pysif.ChainMesh(
    n_cells=n_cells_mesh,
    box_length=box_length,
    field=field,
    drop_indices=True,
    consume_field=True
)

# now we can finally run exodus
# we make it consume the grid because we don't need it anymore
threshold = -0.7
catalog = pysif.finders.exodus(
    grid=grid,
    mesh=mesh,
    radii=search_radii,
    threshold=threshold,
    consume_grid=True
)

# we finally save the catalog and finalize the library
pysif.io.write_catalog_ascii(filepath="path/to/voids.txt", catalog=catalog)

pysif.finalize()
```
::::

::::{tab-item} C
```c
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/catalog_io.h"
#include "sif/io/gadget_io.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/utils/align.h"
#include "sif/utils/array.h"

#include <math.h>

int main(void) {
  sif_field_t* field = NULL;
  sif_grid_t* grid = NULL;
  sif_real* search_radii = NULL;
  sif_chain_mesh_t* mesh = NULL;
  sif_catalog_t* catalog = NULL;
  int status = -1;

  // initialize the library
  // SIF_CONFIG_STANDARD logs at info level, which prints one line for
  // each rung of the finder; SIF_CONFIG_VERBOSE prints everything the
  // library does, SIF_CONFIG_QUIET only warnings and errors
  // in this case, we don't need further configuration
  if (sif_init(SIF_CONFIG_STANDARD) != SIF_OK)
    return 1;

  // first, load the data into a sif_field_t structure
  // in this example, we consider a Gadget snapshot, and read its
  // dark matter particles (SIF_GADGET_PTYPE_1)
  // with this call, we only load the particle positions: velocities
  // and masses are skipped
  // SIF_GADGET_LENGTH_KPC is the unit the snapshot is written in:
  // kpc/h (Gadget's default); SIF_GADGET_LENGTH_MPC for Mpc/h, or
  // SIF_GADGET_LENGTH_AUTO to read it from an HDF5 snapshot.
  // positions and box_length always come out in Mpc/h
  // the fraction argument (1.0 here) allows for subsampling, and the
  // seed after it makes it reproducible
  double box_length = 0.0;
  field = sif_field_read_gadget("path/to/snapshot", SIF_GADGET_FORMAT_AUTO,
    SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_SKIP, SIF_GADGET_MASS_SKIP,
    SIF_GADGET_LENGTH_KPC, 1.0, 0, &box_length);
  if (!field)
    goto done;

  // this call corrects for possible rounding errors
  // in the particle positions, ensuring that they all lay inside the box
  // (subsequent functions will fail if this is not the case)
  if (sif_field_wrap_periodic(field, (sif_real)box_length, NULL, NULL) !=
      SIF_OK)
    goto done;

  // now, we construct the grid and assign the particles with
  // the CiC scheme
  const uint32_t n_cells = 512; // change at will
  grid = sif_grid_alloc(n_cells, (sif_real)box_length);
  if (!grid || sif_grid_assign_cic(grid, field) != SIF_OK)
    goto done;

  // exodus requires that the grid is expressed in density contrast
  if (sif_grid_to_density_contrast(grid) != SIF_OK)
    goto done;

  // now we construct the chain mesh, which is used by exodus to
  // query the particles positions more efficiently
  //
  // first, let's obtain the optimal number of cells
  // this operations requires the maximum search radius,
  // so we construct the array here: 100 radii, evenly spaced in log
  // (numpy's geomspace), as sif_real to ensure compatibility
  // the smallest radius should span at least two grid cells: below
  // that, exodus warns that its results are dominated by noise
  const uint32_t n_radii = 100;
  const double r_min = 2.0 * box_length / n_cells;
  const double r_max = 50.0;
  search_radii = sif_array_logspace(
    (sif_real)log10(r_min), (sif_real)log10(r_max), n_radii, 10.0f);
  if (!search_radii)
    goto done;

  // 0 means that no mesh can hold a search sphere this large: the
  // largest radius is too big for the box
  const uint32_t n_cells_mesh = sif_finder_suggest_mesh_cells(
    field->n_particles, (sif_real)box_length, (sif_real)r_max);
  if (n_cells_mesh == 0)
    goto done;

  // now the chain mesh. to save memory, we make the chain mesh constructor
  // destroy the field (taking its memory over), because in this case we
  // don't need it anymore. also, we don't need the particle indices
  mesh = sif_chain_mesh_alloc_consume(
    n_cells_mesh, (sif_real)box_length, field, SIF_MESH_DROP_INDICES);

  // the field is empty now, but it still has to be released
  sif_field_free(field);
  field = NULL;
  if (!mesh)
    goto done;

  // now we can finally run exodus
  // we make it consume the grid because we don't need it anymore
  const sif_real threshold = -0.7f;
  const sif_real overlap_fraction = 0.0f;
  catalog = sif_finder_exodus(grid, mesh, search_radii, n_radii, threshold,
    overlap_fraction, SIF_FINDER_CONSUME_GRID);
  if (!catalog)
    goto done;

  // we finally save the catalog and finalize the library
  status = sif_catalog_write_ascii("path/to/voids.txt", catalog);

done:
  sif_catalog_free(catalog);
  sif_chain_mesh_free(mesh);
  sif_free_aligned(search_radii);
  sif_grid_free(grid);
  sif_field_free(field);
  sif_finalize();
  return status == SIF_OK ? 0 : 1;
}
```

To build it against an installed **sif** (`cmake --install`, see
{doc}`../installation`), a `CMakeLists.txt` next to it is enough:

```cmake
cmake_minimum_required(VERSION 3.15)
project(run_exodus LANGUAGES C)

find_package(sif REQUIRED)

add_executable(run_exodus run_exodus.c)
target_link_libraries(run_exodus PRIVATE sif::sif m)
```
::::

::::{tab-item} sif-exodus
The same pipeline, described in a configuration file:

```lua
-- exodus.lua
input = {
  path = "path/to/snapshot",
  format = "gadget",
  gadget = { ptype = 1, length = "mpc" },
}

-- the grid is left at its default: one cell per mean particle
-- separation (mps). the smallest radius should span at least two grid
-- cells, so in units of the mps the ladder starts at 2, and grows by
-- 3% a rung
finder = {
  radii = ladder(2, 15, 0.03),
  radii_units = "mps",
  threshold = -0.7,
}

output = "path/to/voids.txt"
```

Check it, then run it:

```bash
sif-exodus --check exodus.lua
```

```bash
sif-exodus exodus.lua
```

`--check` reads only the headers of the input, and says what the run will
be -- tracers, box, grid, mesh, radii in Mpc/h, and the memory it needs --
along with anything that would make it fail, so a mistake in the file
costs seconds rather than a queued job.

The file is a Lua script, so values can be computed, and `-D name=value`
passes a string to it from the command line: with `snap = snap or "010"`
at the top and `path = "path/to/snapdir_" .. snap .. "/snap_" .. snap`,
`sif-exodus -D snap=042 exodus.lua` runs the same file on another
snapshot.

`sif-exodus --template` prints a configuration with every setting and its
default:

:::{dropdown} The template
```{literalinclude} ../../bin/sif-exodus/template.lua
:language: lua
```
:::
::::

:::::

## exodus for surveys

:::::{tab-set}

::::{tab-item} Python

```python
import numpy as np
import pysif

# initialize the library, as for a simulation
pysif.init(log_level=2)

# first, load the galaxies and the randoms into two pysif.Field
# structures, from FITS tables holding right ascension, declination (in
# degrees) and redshift. each part of the field is a column, or an
# expression over columns: here the galaxies' weight is the product of
# two (weights are optional, for the data and the randoms alike). a
# catalogue split over several files -- the two galactic caps here --
# reads as one. naming them ra, dec and z (rather than x, y and z) says
# they are sky coordinates, not positions
#
# the redshift cut goes in where=, and applies the same way to both
# catalogues. the randoms are often many times the galaxies: fraction=
# keeps a random share of what passes the cut (seed= makes it
# reproducible). pysif.io.inspect_fits() lists a file's columns
cut = "Z > 0.43 && Z < 0.7"
data = pysif.io.read_fits(
    ["path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits"],
    ra="RA", dec="DEC", z="Z", w="WEIGHT_SYSTOT * WEIGHT_CP",
    where=cut,
)
randoms = pysif.io.read_fits(
    ["path/to/randoms_NGC.fits", "path/to/randoms_SGC.fits"],
    ra="RA", dec="DEC", z="Z",
    where=cut, fraction=0.5, seed=1,
)

# exodus works in comoving cartesian coordinates, so we convert both
# catalogues with the same cosmology: a w0waCDM background, flat unless
# omega_de is given. positions come out in Mpc/h, with the observer at
# the origin (the convention of pyrecon)
cosmology = dict(omega_m=0.31, w0=-1.0, wa=0.0)
data.convert_sky_coordinates(**cosmology)
randoms.convert_sky_coordinates(**cosmology)

# the radii. in a survey they also decide how much empty space the
# survey needs around it (see below), so we construct them first
search_radii = np.geomspace(10, 60, 60, dtype=pysif.real)

# nothing wraps around in a survey: exodus runs in a box of empty
# space around the footprint, wide enough that the smoothing and the
# sphere searches never reach its far side. survey_box works out that
# box from the randoms (which define the footprint) for this ladder
# and this grid, along with the offset that moves the survey into it
n_cells = 512                     # change at will
offset, box_length = pysif.finders.survey_box(
    randoms=randoms, radii=search_radii, n_cells=n_cells
)

# move both catalogues into the box, by the same offset
data.translate(offset)
randoms.translate(offset)

# now the grids, one for the galaxies and one for the randoms, with
# the same number of cells over the same box, both assigned with the
# CiC scheme. unlike for a simulation, they are NOT turned into density
# contrasts: exodus compares the two densities itself
data_grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
data_grid.assign_cic(field=data)

random_grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
random_grid.assign_cic(field=randoms)

# now the chain meshes, one for each catalogue. in a survey box the
# tracers only fill the footprint, so suggest_mesh_cells_survey sizes
# each mesh for the density there, reading the footprint off the
# random grid
max_radius = float(search_radii.max())
n_cells_data = pysif.finders.suggest_mesh_cells_survey(
    n_particles=data.n_particles, random_grid=random_grid,
    max_radius=max_radius
)
n_cells_randoms = pysif.finders.suggest_mesh_cells_survey(
    n_particles=randoms.n_particles, random_grid=random_grid,
    max_radius=max_radius
)

# as for a simulation, the meshes consume the fields to save memory,
# and we don't need the particle indices
data_mesh = pysif.ChainMesh(
    n_cells=n_cells_data, box_length=box_length, field=data,
    drop_indices=True, consume_field=True
)
random_mesh = pysif.ChainMesh(
    n_cells=n_cells_randoms, box_length=box_length, field=randoms,
    drop_indices=True, consume_field=True
)

# now we can run exodus. a sphere's density contrast is measured
# against the randoms, and every void found is kept, along with the
# fraction of its sphere (footprint) and of the shell out to twice its
# radius (footprint_shell) that lie inside the survey
threshold = -0.7
catalog = pysif.finders.exodus_survey(
    data_grid=data_grid,
    random_grid=random_grid,
    data_mesh=data_mesh,
    random_mesh=random_mesh,
    radii=search_radii,
    threshold=threshold,
    consume_grid=True
)

# move the voids back to the frame of the catalogues: comoving
# cartesian positions, in Mpc/h, with the observer at the origin
catalog.translate(-offset)

# and, to hand them on in the survey's own terms, back to the sky with
# the same cosmology: right ascension, declination and redshift. the
# radii stay comoving lengths, in Mpc/h
catalog.to_sky(**cosmology)

# we finally save the catalog, footprint columns included, as a FITS
# table (RA, DEC, Z, R, ...), note what it was made with in its
# header, and finalize the library. write_catalog_hdf5 and
# write_catalog_ascii write a sky catalogue as well
pysif.io.write_catalog_fits("path/to/voids.fits", catalog)
pysif.io.set_fits_key("path/to/voids.fits", "threshold", threshold)
pysif.io.set_fits_key("path/to/voids.fits", "omega_m", cosmology["omega_m"])

pysif.finalize()
```
::::

::::{tab-item} C

```c
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/fits_io.h"
#include "sif/model/cosmology.h"
#include "sif/structures/catalog.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"
#include "sif/structures/grid.h"
#include "sif/utils/align.h"
#include "sif/utils/array.h"

#include <math.h>

int main(void) {
  sif_field_t* data = NULL;
  sif_field_t* randoms = NULL;
  sif_real* search_radii = NULL;
  sif_grid_t* data_grid = NULL;
  sif_grid_t* random_grid = NULL;
  sif_chain_mesh_t* data_mesh = NULL;
  sif_chain_mesh_t* random_mesh = NULL;
  sif_catalog_t* catalog = NULL;
  int status = -1;

  // initialize the library, as for a simulation
  if (sif_init(SIF_CONFIG_STANDARD) != SIF_OK)
    return 1;

  // first, load the galaxies and the randoms into two sif_field_t
  // structures, from FITS tables holding right ascension, declination (in
  // degrees) and redshift. each part of the field is a column, or an
  // expression over columns: here the galaxies' weight is the product of
  // two (weights are optional, for the data and the randoms alike). a
  // catalogue split over several files -- the two galactic caps here --
  // reads as one. naming them ra, dec and z (rather than x, y and z) says
  // they are sky coordinates, not positions
  //
  // the redshift cut is the filter, and applies the same way to both
  // catalogues. the randoms are often many times the galaxies: the
  // fraction keeps a random share of what passes the cut (the seed after
  // it makes it reproducible). sif_fits_inspect() lists a file's columns
  const char* cut = "Z > 0.43 && Z < 0.7";
  const char* data_files[] = {
    "path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits"};
  const char* random_files[] = {
    "path/to/randoms_NGC.fits", "path/to/randoms_SGC.fits"};
  const sif_fits_columns_t data_columns = {
    .ra = "RA", .dec = "DEC", .z = "Z", .w = "WEIGHT_SYSTOT * WEIGHT_CP"};
  const sif_fits_columns_t random_columns = {
    .ra = "RA", .dec = "DEC", .z = "Z"};
  data = sif_field_read_fits(data_files, 2, NULL, &data_columns, cut, 1.0, 0);
  randoms =
    sif_field_read_fits(random_files, 2, NULL, &random_columns, cut, 0.5, 1);
  if (!data || !randoms)
    goto done;

  // exodus works in comoving cartesian coordinates, so we convert both
  // catalogues with the same cosmology: a w0waCDM background, whose
  // curvature is 1 - omega_m - omega_de - omega_r. positions come out in
  // Mpc/h, with the observer at the origin (the convention of pyrecon)
  const sif_cosmology_t cosmology = {
    .omega_m = 0.31, .omega_de = 0.69, .omega_r = 0.0, .w0 = -1.0, .wa = 0.0};
  if (sif_field_convert_sky_coordinates(data, &cosmology) != SIF_OK ||
      sif_field_convert_sky_coordinates(randoms, &cosmology) != SIF_OK)
    goto done;

  // the radii: 60 of them, evenly spaced in log (numpy's geomspace). in
  // a survey they also decide how much empty space the survey needs
  // around it (see below), so we construct them first
  const uint32_t n_radii = 60;
  const double r_min = 10.0, r_max = 60.0;
  search_radii = sif_array_logspace(
    (sif_real)log10(r_min), (sif_real)log10(r_max), n_radii, 10.0f);
  if (!search_radii)
    goto done;

  // nothing wraps around in a survey: exodus runs in a box of empty
  // space around the footprint, wide enough that the smoothing and the
  // sphere searches never reach its far side. this call works out that
  // box from the randoms (which define the footprint) for this ladder
  // and this grid, along with the offset that moves the survey into it.
  // SIF_DEFAULT is the search factor the finder will run with
  const uint32_t n_cells = 512; // change at will
  sif_real offset[3], box_length;
  if (sif_finder_exodus_survey_box(randoms, search_radii, n_radii, n_cells,
        SIF_DEFAULT, offset, &box_length) != SIF_OK)
    goto done;

  // move both catalogues into the box, by the same offset
  if (sif_field_translate(data, offset) != SIF_OK ||
      sif_field_translate(randoms, offset) != SIF_OK)
    goto done;

  // now the grids, one for the galaxies and one for the randoms, with
  // the same number of cells over the same box, both assigned with the
  // CiC scheme. unlike for a simulation, they are NOT turned into density
  // contrasts: exodus compares the two densities itself
  data_grid = sif_grid_alloc(n_cells, box_length);
  random_grid = sif_grid_alloc(n_cells, box_length);
  if (!data_grid || !random_grid ||
      sif_grid_assign_cic(data_grid, data) != SIF_OK ||
      sif_grid_assign_cic(random_grid, randoms) != SIF_OK)
    goto done;

  // now the chain meshes, one for each catalogue. in a survey box the
  // tracers only fill the footprint, so each mesh is sized for the
  // density there, reading the footprint off the random grid
  const uint32_t n_cells_data = sif_finder_suggest_mesh_cells_survey(
    data->n_particles, random_grid, (sif_real)r_max);
  const uint32_t n_cells_randoms = sif_finder_suggest_mesh_cells_survey(
    randoms->n_particles, random_grid, (sif_real)r_max);
  if (n_cells_data == 0 || n_cells_randoms == 0)
    goto done;

  // as for a simulation, the meshes consume the fields to save memory,
  // and we don't need the particle indices
  data_mesh = sif_chain_mesh_alloc_consume(
    n_cells_data, box_length, data, SIF_MESH_DROP_INDICES);
  random_mesh = sif_chain_mesh_alloc_consume(
    n_cells_randoms, box_length, randoms, SIF_MESH_DROP_INDICES);

  // the fields are empty now, but they still have to be released
  sif_field_free(data);
  sif_field_free(randoms);
  data = randoms = NULL;
  if (!data_mesh || !random_mesh)
    goto done;

  // now we can run exodus. a sphere's density contrast is measured
  // against the randoms, and every void found is kept, along with the
  // fraction of its sphere (footprint) and of the shell out to twice its
  // radius (footprint_shell) that lie inside the survey
  const sif_real threshold = -0.7f;
  const sif_real overlap_fraction = 0.0f;
  catalog = sif_finder_exodus_survey(data_grid, random_grid, data_mesh,
    random_mesh, search_radii, n_radii, threshold, overlap_fraction,
    SIF_FINDER_CONSUME_GRID);
  if (!catalog)
    goto done;

  // move the voids back to the frame of the catalogues: comoving
  // cartesian positions, in Mpc/h, with the observer at the origin
  const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
  if (sif_catalog_translate(catalog, back) != SIF_OK)
    goto done;

  // and, to hand them on in the survey's own terms, back to the sky with
  // the same cosmology: right ascension, declination and redshift. the
  // radii stay comoving lengths, in Mpc/h
  if (sif_catalog_to_sky(catalog, &cosmology) != SIF_OK)
    goto done;

  // we finally save the catalog, footprint columns included, as a FITS
  // table (RA, DEC, Z, R, ...), note what it was made with in its
  // primary header, and finalize the library. sif_catalog_write_hdf5()
  // and sif_catalog_write_ascii() write a sky catalogue as well
  status = sif_catalog_write_fits("path/to/voids.fits", catalog);
  if (status == SIF_OK)
    status = sif_fits_set_key_real("path/to/voids.fits", NULL, "threshold",
      (double)threshold);
  if (status == SIF_OK)
    status = sif_fits_set_key_real(
      "path/to/voids.fits", NULL, "omega_m", cosmology.omega_m);

done:
  sif_catalog_free(catalog);
  sif_chain_mesh_free(random_mesh);
  sif_chain_mesh_free(data_mesh);
  sif_grid_free(random_grid);
  sif_grid_free(data_grid);
  sif_free_aligned(search_radii);
  sif_field_free(randoms);
  sif_field_free(data);
  sif_finalize();
  return status == SIF_OK ? 0 : 1;
}
```

It builds with the same `CMakeLists.txt` as the simulation example.
::::

:::::
