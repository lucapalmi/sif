# Running **exodus**

This page shows how to run the **exodus** void finder using **sif**

## **exodus** for N-body simulations

:::::{tab-set}

::::{tab-item} Python

```python
import numpy as np
import pysif

# initialize the library
# log_level 0 means trace, which prints out detailed information 
# for each rung. higher levels print out only the essential information
# in this case, we don't need further configuration
pysif.init(log_level=0)

# first, load the data into a pysif.Field structure
# in this example, we consider a Gadget snapshot
# with this command, we only load the particle positions 
# the argument 'fraction=' allows for subsampling
field, box_length = pysif.io.read_gadget("path/to/snapshot", ptype=1)

# this instructions corrects for possible rounding errors 
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
search_radii = np.geomspace(1, 50, 100), dtype=pysif.real)

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
pysif.io.write_catalog(filepath="path/to/file.txt", catalog=catalog)

pysif.finalize()

```
::::

::::{tab-item} C

ciao

::::

:::::

## **exodus** for surveys

:::::{tab-set}

::::{tab-item} Python

ciao

::::

::::{tab-item} C

ciao

::::

:::::
