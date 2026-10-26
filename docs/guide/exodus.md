# The exodus finder

An efficient, spherical void-finding algorithm that works on both surveys and
simulations.

## Adopted definition of void

A sphere with centre $\mathbf{q}$ and radius $R$ is a void if:

1. $\mathbf{q}$ is not inside a larger void (the overlap fraction sets how much two voids
   may share);
2. $\mathbf{q}$ is a minimum of the density field smoothed at $R$;
3. $R$ is the largest scale at which the contrast inside the sphere of radius $R$
   about $\mathbf{q}$ crosses the threshold.

## Algorithm description

exodus walks a ladder of radii, largest first. At each rung it smooths the
grid with a top-hat of that radius, takes candidates below the threshold,
grows each candidate's radius on the tracers until the enclosed contrast
crosses the threshold (point 3), and accepts it if it does not overlap a void
already found (point 1).

## The variants

exodus comes in two variants, which are simply two configurations of the 
algorithm that follow two different definitions of void:

| Variant | Candidates | Centre | Overlap |
|---|---|---|---|
| `"excursion-set"` (default) | every cell below the threshold | its grid node | 0 | 
| `"excursion-peak"` | local minima of the rung's field, smoothed by 0.15 R | refined between the nodes | 1 |

With `"excursion-set"`, each cell below threshold is a candidate, and voids 
follow a strict, non-overlapping rule. This configuration follows the same
definition adopted by the finder of [Pylians3](https://pylians3.readthedocs.io/en/master/index.html)

`"excursion-peak"` follows a slightly different definition of void, where
a cell below threshold is a candidate only if it's a strict minimum of the 
field. Also, voids can overlap up to their centres.

:::::{tab-set}

::::{tab-item} sif-exodus
```lua
finder = {
  radii = ladder(10, 40, 0.05),
  threshold = -0.7,
  variant = "excursion-peak",
}
```
::::

::::{tab-item} Python
```python
voids = pysif.finders.exodus(grid, mesh, radii, -0.7, variant="excursion-peak")
```
::::

::::{tab-item} C
```c
sif_catalogue_t* voids = sif_finder_exodus(grid, mesh, radii, n_radii, -0.7f,
  SIF_EXODUS_EXCURSION_PEAK, NULL);
```
::::

:::::

### Beyond the variants

A variant is just a preset of the finder's settings: the candidates (`seeds`,
`"cells"` or `"minima"`), the overlap fraction, the search factor (1.5) and the
window deconvolution (`"full"`). Any other combination can be run with the raw
finder, which takes each setting separately. In **sif-exodus**,
`finder.exodus_params` overrides the settings of the variant, or sets all of
them on its own, starting from the default variant. One of the two is required.

:::::{tab-set}

::::{tab-item} sif-exodus
```lua
finder = {
  radii = ladder(10, 40, 0.05),
  threshold = -0.7,
  variant = "excursion-peak",
  exodus_params = { overlap_fraction = 0.5 },
}
```
::::

::::{tab-item} Python
```python
voids = pysif.finders.exodus_raw(grid, mesh, radii, -0.7, overlap_fraction=0.5,
                                 seeds="minima")
```
::::

::::{tab-item} C
```c
sif_catalogue_t* voids = sif_finder_exodus_raw(grid, mesh, radii, n_radii,
  -0.7f, 0.5f, SIF_FINDER_SEEDS_MINIMA, NULL);
```
::::

:::::

## Choosing the ladder and the grid

exodus walks the ladder from the largest radius down, and a void is accepted
only if its centre is not inside a larger one. This can only be checked
against voids already found, so the first rung of the ladder should find no
void at all: start the ladder above the largest void you expect, and make sure
that the first rung (at least) comes out empty. Otherwise, the voids found
there were never compared with the larger ones the ladder does not reach.

The void size function slightly depends on the grid resolution. Given a grid with 
cell size $\Delta x$:

- `"excursion-peak"` is stable for scales $R > 6\,\Delta x$
- `"excursion-set"` is stable for scales $R > 10\,\Delta x$

Finally, the algorithm causes a "loss of voids" at the scales close to the smallest 
radius. Run the finder slightly below the minimum scale you want to consider, 
indicatively by a factor 1.5x.
