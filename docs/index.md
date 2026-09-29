# sif

**sif** is a high performance library to study cosmic voids.

The library is written in C, but it also features a Python interface called **pysif**.

These are some of its many features:

- **exodus:** a spherical void finder for N-body simulations and surveys; fast and reliable
- **measuring:** functions to measure the size function, velocity and density profiles for any void catalogue
- **modelling:** functions to compute the [SvdW](https://arxiv.org/abs/astro-ph/0311260), [Vdn](https://arxiv.org/abs/1304.6087) and [excursion-peak](https://arxiv.org/abs/2401.14451) models for void abundances


```{toctree}
:hidden:
:maxdepth: 2

installation
examples/index
concepts/index
python/index
api/index
```

## Start here

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Installing the library
:::

:::{grid-item-card} Running **exodus**
:link: examples/run_exodus
:link-type: doc

Template files to run **exodus**
:::

:::{grid-item-card} Void profiles
:link: examples/void_profiles
:link-type: doc

Template files to compute void profiles
:::

:::{grid-item-card} How **exodus** works
:link: concepts/exodus
:link-type: doc

Description of the **exodus** algorithm
:::

:::{grid-item-card} Python API
:link: python/index
:link-type: doc

Python API of the library
:::

:::{grid-item-card} C API
:link: api/index
:link-type: doc

C API of the library
:::

::::
