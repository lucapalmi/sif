# Running exodus

How to run the exodus void finder

This page contains practical examples that show how to run the exodus void finder, 
both for simulations and surveys. The same two pipelines are shown in the three languages
of exodus: Python, C, and the Lua configuration file of the **sif-exodus** program. 

## exodus for N-body simulations

:::::{tab-set}

::::{tab-item} Python
```python
import numpy as np
import pysif

# initialise the library
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
# record it, so it has to be named: 'kpc' for kpc/h or 'mpc' for Mpc/h.
# positions and box_length always come out in Mpc/h
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
n_cells = 512                            # change at will
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
catalogue = pysif.finders.exodus(
    grid=grid,
    mesh=mesh,
    radii=search_radii,
    threshold=threshold,
    consume_grid=True
)

# we finally save the catalogue and finalise the library
pysif.io.write_catalogue_ascii(filepath="path/to/voids.txt", catalogue=catalogue)

pysif.finalise()
```
::::

::::{tab-item} C
```c
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/catalogue_io.h"
#include "sif/io/gadget_io.h"
#include "sif/structures/catalogue.h"
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
  sif_catalogue_t* catalogue = NULL;
  int status = -1;

  // initialise the library
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
  catalogue = sif_finder_exodus(grid, mesh, search_radii, n_radii, threshold,
    overlap_fraction, SIF_FINDER_CONSUME_GRID);
  if (!catalogue)
    goto done;

  // we finally save the catalogue and finalise the library
  status = sif_catalogue_write_ascii("path/to/voids.txt", catalogue);

done:
  sif_catalogue_free(catalogue);
  sif_chain_mesh_free(mesh);
  sif_free_aligned(search_radii);
  sif_grid_free(grid);
  sif_field_free(field);
  sif_finalise();
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

```lua
-- exodus.lua
mode = "box"

input = {
  path = "path/to/snapshot",
  format = "gadget",
  ptype = 1,
  length = "mpc",
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

::::

:::::

## exodus for surveys

:::::{tab-set}

::::{tab-item} Python

```python
import numpy as np
import pysif

# initialise the library, as for a simulation
pysif.init(log_level=2)

# first, load the galaxies and the randoms into two pysif.Field
# structures, from FITS tables holding right ascension, declination (in
# degrees) and redshift. Weights are optional and can be computed
# on the fly from multiple columns (simple expressions only).
# Catalogues that split over several files can be loaded as one.
# The named arguments ra, dec and z (rather than x, y and z) automatically
# set the sky coordinates.
#
# A cut (e.g. redshifts) can be applied directly when reading. The arguments
# 'fraction=' and 'seed=' control the subsampling fraction and subsampling seed
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
# the origin
cosmology = dict(omega_m=0.31, w0=-1.0, wa=0.0)
data.convert_sky_coordinates(**cosmology)
randoms.convert_sky_coordinates(**cosmology)

# now the search radii. in a survey they also decide how much empty space the
# survey needs around it. We use the real datatype to ensure compatibility
search_radii = np.geomspace(10, 60, 60, dtype=pysif.real)

# we now compute the offset and the box length that pad the survey volume
# in the box, to ensure the boundary conditions do not apply.
# They depend on the survey footprint, the search radii and the number of
# cells in the grid
n_cells = 512                     # change at will
offset, box_length = pysif.finders.survey_box(
    randoms=randoms, radii=search_radii, n_cells=n_cells
)

# we now apply the offset
data.translate(offset)
randoms.translate(offset)

# now the grids, one for the galaxies and one for the randoms, with
# the same number of cells. Unlike for a simulation,
# they are NOT turned into density contrast
data_grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
data_grid.assign_cic(field=data)

random_grid = pysif.Grid(n_cells=n_cells, box_length=box_length)
random_grid.assign_cic(field=randoms)

# now the chain meshes, one for each catalogue. The optimal number of
# cells is computed from the random footprint.
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

# now we can run exodus
threshold = -0.7
catalogue = pysif.finders.exodus_survey(
    data_grid=data_grid,
    random_grid=random_grid,
    data_mesh=data_mesh,
    random_mesh=random_mesh,
    radii=search_radii,
    threshold=threshold,
    consume_grid=True
)

# move the voids back to the frame of the catalogues
catalogue.translate(-offset)

# optionally, we can translate the voids positions in sky coordinates
catalogue.to_sky(**cosmology)

# we finally save the catalogue, in FITS format.
# we also save some metadata, which every format keeps with it
catalogue.set_metadata("threshold", threshold)
catalogue.set_metadata("omega_m", cosmology["omega_m"])
pysif.io.write_catalogue_fits("path/to/voids.fits", catalogue)

pysif.finalise()
```
::::

::::{tab-item} C

```c
#include "sif/core/system.h"
#include "sif/finder/exodus_finder.h"
#include "sif/io/fits_io.h"
#include "sif/model/cosmology.h"
#include "sif/structures/catalogue.h"
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
  sif_catalogue_t* catalogue = NULL;
  int status = -1;

  // initialise the library, as for a simulation
  if (sif_init(SIF_CONFIG_STANDARD) != SIF_OK)
    return 1;

  // first, load the galaxies and the randoms into two sif_field_t
  // structures, from FITS tables holding right ascension, declination (in
  // degrees) and redshift. Weights are optional and can be computed
  // on the fly from multiple columns (simple expressions only).
  // Catalogues that split over several files can be loaded as one.
  // The named columns ra, dec and z (rather than x, y and z) automatically
  // set the sky coordinates.
  //
  // A cut (e.g. redshifts) can be applied directly when reading. The
  // arguments after the cut are the subsampling fraction and the
  // subsampling seed
  const char* cut = "Z > 0.43 && Z < 0.7";
  const char* data_files[] = {
    "path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits"};
  const char* random_files[] = {
    "path/to/randoms_NGC.fits", "path/to/randoms_SGC.fits"};
  const sif_field_columns_t data_columns = {
    .ra = "RA", .dec = "DEC", .z = "Z", .w = "WEIGHT_SYSTOT * WEIGHT_CP"};
  const sif_field_columns_t random_columns = {
    .ra = "RA", .dec = "DEC", .z = "Z"};
  data = sif_field_read_fits(data_files, 2, NULL, &data_columns, cut, 1.0, 0);
  randoms =
    sif_field_read_fits(random_files, 2, NULL, &random_columns, cut, 0.5, 1);
  if (!data || !randoms)
    goto done;

  // exodus works in comoving cartesian coordinates, so we convert both
  // catalogues with the same cosmology: a w0waCDM background, whose
  // curvature is 1 - omega_m - omega_de - omega_r. positions come out in
  // Mpc/h, with the observer at the origin
  const sif_cosmology_t cosmology = {
    .omega_m = 0.31, .omega_de = 0.69, .omega_r = 0.0, .w0 = -1.0, .wa = 0.0};
  if (sif_field_convert_sky_coordinates(data, &cosmology) != SIF_OK ||
      sif_field_convert_sky_coordinates(randoms, &cosmology) != SIF_OK)
    goto done;

  // now the search radii: 60 of them, evenly spaced in log (numpy's
  // geomspace). in a survey they also decide how much empty space the
  // survey needs around it. we use sif_real to ensure compatibility
  const uint32_t n_radii = 60;
  const double r_min = 10.0, r_max = 60.0;
  search_radii = sif_array_logspace(
    (sif_real)log10(r_min), (sif_real)log10(r_max), n_radii, 10.0f);
  if (!search_radii)
    goto done;

  // we now compute the offset and the box length that pad the survey
  // volume in the box, to ensure the boundary conditions do not apply.
  // they depend on the survey footprint (defined by the randoms), the
  // search radii and the number of cells in the grid.
  // SIF_DEFAULT is the search factor the finder will run with
  const uint32_t n_cells = 512; // change at will
  sif_real offset[3], box_length;
  if (sif_finder_exodus_survey_box(randoms, search_radii, n_radii, n_cells,
        SIF_DEFAULT, offset, &box_length) != SIF_OK)
    goto done;

  // we now apply the offset
  if (sif_field_translate(data, offset) != SIF_OK ||
      sif_field_translate(randoms, offset) != SIF_OK)
    goto done;

  // now the grids, one for the galaxies and one for the randoms, with
  // the same number of cells. unlike for a simulation, they are NOT
  // turned into density contrast
  data_grid = sif_grid_alloc(n_cells, box_length);
  random_grid = sif_grid_alloc(n_cells, box_length);
  if (!data_grid || !random_grid ||
      sif_grid_assign_cic(data_grid, data) != SIF_OK ||
      sif_grid_assign_cic(random_grid, randoms) != SIF_OK)
    goto done;

  // now the chain meshes, one for each catalogue. the optimal number of
  // cells is computed from the random footprint
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

  // now we can run exodus
  const sif_real threshold = -0.7f;
  const sif_real overlap_fraction = 0.0f;
  catalogue = sif_finder_exodus_survey(data_grid, random_grid, data_mesh,
    random_mesh, search_radii, n_radii, threshold, overlap_fraction,
    SIF_FINDER_CONSUME_GRID);
  if (!catalogue)
    goto done;

  // move the voids back to the frame of the catalogues
  const sif_real back[3] = {-offset[0], -offset[1], -offset[2]};
  if (sif_catalogue_translate(catalogue, back) != SIF_OK)
    goto done;

  // optionally, we can translate the voids positions in sky coordinates
  if (sif_catalogue_to_sky(catalogue, &cosmology) != SIF_OK)
    goto done;

  // we finally save the catalogue, in FITS format.
  // we also save some metadata, which every format keeps with it
  status = sif_catalogue_meta_real_set(catalogue, "threshold", threshold);
  if (status == SIF_OK)
    status = sif_catalogue_meta_real_set(catalogue, "omega_m", cosmology.omega_m);
  if (status == SIF_OK)
    status = sif_catalogue_write_fits("path/to/voids.fits", catalogue);

done:
  sif_catalogue_free(catalogue);
  sif_chain_mesh_free(random_mesh);
  sif_chain_mesh_free(data_mesh);
  sif_grid_free(random_grid);
  sif_grid_free(data_grid);
  sif_free_aligned(search_radii);
  sif_field_free(randoms);
  sif_field_free(data);
  sif_finalise();
  return status == SIF_OK ? 0 : 1;
}
```

It builds with the same `CMakeLists.txt` as the simulation example.
::::

::::{tab-item} sif-exodus

```lua
-- survey.lua
mode = "survey"

-- the galaxies and the randoms, each read as a box's input is: path,
-- format and the format's settings. the columns are sky coordinates,
-- which is what a survey reads unless coordinates = "cartesian" says
-- otherwise; the cosmology takes them to comoving Mpc/h, and the voids
-- back onto the sky. flat unless omega_de is given
local cut = "Z > 0.43 && Z < 0.7"
input = {
  cosmology = { omega_m = 0.31 },
  data = {
    path = { "path/to/galaxies_NGC.fits", "path/to/galaxies_SGC.fits" },
    format = "fits",
    columns = { ra = "RA", dec = "DEC", z = "Z", w = "WEIGHT_SYSTOT * WEIGHT_CP" },
    where = cut,
  },
  random = {
    path = { "path/to/randoms_NGC.fits", "path/to/randoms_SGC.fits" },
    format = "fits",
    columns = { ra = "RA", dec = "DEC", z = "Z" },
    where = cut,
    fraction = 0.5,
    seed = 1,
  },
}

-- the box around the survey, its padding and the offset into it are
-- worked out from the randoms; the grid spans that box
grid = { n_cells = 512 }

finder = {
  radii = geomspace(10, 60, 60),
  threshold = -0.7,
}

-- the run's settings, the cosmology among them, go in the metadata
output = "path/to/voids.fits"
```

Check it, then run it:

```bash
sif-exodus --check exodus.lua
```

```bash
sif-exodus exodus.lua
```

::::

:::::
