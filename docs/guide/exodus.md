# The exodus finder

An efficient, spherical void-finding algorithm that works on both surveys and
simulations.

## What a void is

A void is a centre q and a radius R such that

1. q is not inside a larger void (the overlap fraction sets how much two voids
   may share);
2. q is a minimum of the density field smoothed at R;
3. R is the largest scale at which the contrast inside the sphere of radius R
   about q crosses the threshold.

Write R(x) for the radius point 3 gives at any point x. Then R(x) is largest
exactly where x is a minimum of the field smoothed at R(x): a void centre is a
local maximum of R(x), a property of the tracers alone.

## How it searches

exodus walks a ladder of radii, largest first. At each rung it smooths the
grid with a top-hat of that radius, takes the cells below the threshold as
candidates, grows each candidate's radius on the tracers until the enclosed
contrast crosses the threshold (point 3), and accepts it if it does not
overlap a void already found (point 1). Where a candidate is centred is the
`centres` option, and decides how closely point 2 is followed:

| `centres` | Candidates | Centre | Depends on the grid |
|---|---|---|---|
| `"legacy"` (default) | every cell below the threshold | its grid node | yes |
| `"minima"` | local minima of the rung's field, smoothed by 0.15 R | refined between the nodes | slightly |
| `"optimised"` | the same minima | moved to the largest R(x) nearby | no |

`"legacy"` is the finder of 0.4, kept to reproduce its catalogues: where a
minimum is refused, the cell next to it can be accepted instead, a void on the
slope of a larger one. `"minima"` centres every void on a minimum of its rung's
field. `"optimised"` then moves it to where the tracers give the largest radius,
so the grid only seeds the search; it costs about 1.3 to 6 times `"minima"`,
more for voids that are large in units of the tracer separation. The overlap
fraction, the search factor and the window options apply to all three alike.
`"minima"` and `"optimised"` are for periodic boxes only for now, and
`"optimised"` for unweighted tracers.

## Choosing the ladder and the grid

With `centres="optimised"` the void size function does not depend on the grid,
with two conditions, both from the bottom of the ladder:

- **Start the ladder below the smallest radius you report**, by about 1.5. A
  void just above the bottom rung can only be found from the last rung or two,
  and a grid resolves those least well.
- **Make the grid fine enough for the bottom rung**: a cell of at most about a
  third to a half of the smallest radius on the ladder.

In a 1876 Mpc/h box with 1500^3 tracers, on grids from 600^3 to 2000^3, the
VSF above 1.5 times the bottom rung then agrees within 1-2% at every grid,
where `"legacy"` differs by 15-30%.

:::{note}
The overlap fraction is independent of `centres`: an overlap fraction of 0
keeps voids disjoint, and 1 allows any void whose centre lies outside every
larger one.
:::
