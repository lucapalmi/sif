# Void profiles

How to compute the radial profiles of a void catalogue.

This page contains practical examples of how to compute the density and
velocity profiles of a generic void catalogue, in both C and Python.

:::::{tab-set}

::::{tab-item} Python
```python
import numpy as np
import pysif

# initialise the library
pysif.init(log_level=2)

# first, load the voids. this can be any catalogue produced by any finder,
# it's just a list of centres and radii
catalogue = pysif.io.read_catalogue_ascii("path/to/voids.txt")

# now the tracers the profiles are measured from. these have to be the
# whole sample the voids were found in.
# load the velocities to also obtain the velocity profiles
field, box_length = pysif.io.read_gadget(
    "path/to/snapshot", ptype=1, velocities="peculiar", length="mpc"
)

# correct for float rounding errors
field.wrap(box_length=box_length)

# now the chain mesh, which is used to query the tracers efficiently
# we use the dedicated function to get the optimal number of cells
n_cells_mesh = pysif.measure.profiles_suggest_mesh_cells(field.n_particles)
mesh = pysif.ChainMesh(
    n_cells=n_cells_mesh,
    box_length=box_length,
    field=field,
    drop_indices=True,
    consume_field=True
)

# now we can compute the profiles
# 'ext=' is the outer edge of the profile in units of each void's own
# radius, split in 'n_bins' bins. by default a bin holds the density
# contrast enclosed within its outer edge (cumulative); with
# 'differential=True' it holds the contrast of its own shell instead.
# velocity profiles are only available if the mesh carries velocities
profiles = pysif.measure.profiles(
    catalogue=catalogue,
    mesh=mesh,
    n_bins=30,
    ext=3.0,
    compute_velocity=True,
    differential=True
)

# we finally save the profiles, together with the catalogue they were
# measured from, and finalise the library
pysif.io.write_profiles_ascii("path/to/profiles.txt", profiles, catalogue)

pysif.finalise()
```
::::

::::{tab-item} C
```c
#include "sif/core/system.h"
#include "sif/io/catalogue_io.h"
#include "sif/io/gadget_io.h"
#include "sif/io/profiles_io.h"
#include "sif/measure/profiles.h"
#include "sif/structures/catalogue.h"
#include "sif/structures/chain_mesh.h"
#include "sif/structures/field.h"

int main(void) {
  sif_catalogue_t* catalogue = NULL;
  sif_field_t* field = NULL;
  sif_chain_mesh_t* mesh = NULL;
  sif_density_profiles_t* dens = NULL;
  sif_velocity_profiles_t* vel = NULL;
  int status = -1;

  // initialise the library
  if (sif_init(SIF_CONFIG_STANDARD) != SIF_OK)
    return 1;

  // first, load the voids. this can be any catalogue produced by any finder,
  // it's just a list of centres and radii
  // the NULL takes the columns from the file's header
  catalogue = sif_catalogue_read_ascii("path/to/voids.txt", NULL);
  if (!catalogue)
    goto done;

  // now the tracers the profiles are measured from. these have to be the
  // whole sample the voids were found in.
  // load the velocities to also obtain the velocity profiles
  double box_length = 0.0;
  field = sif_field_read_gadget("path/to/snapshot", SIF_GADGET_FORMAT_AUTO,
    SIF_GADGET_PTYPE_1, SIF_GADGET_VELOCITY_PECULIAR, SIF_GADGET_MASS_SKIP,
    SIF_GADGET_LENGTH_MPC, 1.0, 0, &box_length);
  if (!field)
    goto done;

  // correct for float rounding errors
  if (sif_field_wrap_periodic(field, (sif_real)box_length, NULL, NULL) !=
      SIF_OK)
    goto done;

  // now the chain mesh, which is used to query the tracers efficiently
  // we use the dedicated function to get the optimal number of cells
  const uint32_t n_cells_mesh =
    sif_profiles_suggest_mesh_cells(field->n_particles);
  mesh = sif_chain_mesh_alloc_consume(
    n_cells_mesh, (sif_real)box_length, field, SIF_MESH_DROP_INDICES);

  // the field is empty now, but it still has to be released
  sif_field_free(field);
  field = NULL;
  if (!mesh)
    goto done;

  // now we can compute the profiles
  // 'ext' is the outer edge of the profile in units of each void's own
  // radius, split in 'n_bins' bins. by default a bin holds the density
  // contrast enclosed within its outer edge (SIF_PROFILES_CUMULATIVE); with
  // SIF_PROFILES_DIFFERENTIAL it holds the contrast of its own shell instead.
  // velocity profiles are only available if the mesh carries velocities;
  // passing NULL instead of &vel skips them
  const sif_real ext = 3.0f;
  const uint32_t n_bins = 30;
  const sif_option opt = SIF_PBC_PERIODIC | SIF_PROFILES_DIFFERENTIAL |
                         SIF_PROFILES_VELOCITY_NUMBER;
  if (sif_profiles(catalogue, mesh, ext, n_bins, opt, &dens, &vel) != SIF_OK)
    goto done;

  // we finally save the profiles, together with the catalogue they were
  // measured from, and finalise the library
  status = sif_profiles_write_ascii("path/to/profiles.txt", dens, vel, catalogue);

done:
  sif_velocity_profiles_free(vel);
  sif_density_profiles_free(dens);
  sif_chain_mesh_free(mesh);
  sif_field_free(field);
  sif_catalogue_free(catalogue);
  sif_finalise();
  return status == SIF_OK ? 0 : 1;
}
```

It builds with the same `CMakeLists.txt` as the {doc}`exodus example
<run_exodus>`.
::::

:::::
