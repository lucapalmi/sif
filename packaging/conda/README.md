# The conda package

`recipe.yaml` builds pysif as a conda package for conda-forge. Unlike the
wheel, which carries its own FFTW, HDF5, cfitsio and OpenMP runtime, the
package takes each from conda-forge as a dependency, so a conda environment
holds one OpenMP runtime, shared with numpy, scipy and everything else.

## Building it here

With [rattler-build](https://rattler.build) and conda-forge's variant
configuration (`conda-forge-pinning`) in an environment of their own:

```
conda create -n rattler -c conda-forge rattler-build conda-forge-pinning
conda activate rattler
rattler-build build -r packaging/conda/recipe.yaml \
  -m $CONDA_PREFIX/conda_build_config.yaml -m packaging/conda/local.yaml
```

This builds the checkout as it is, for one Python (`local.yaml`), and runs
the package's tests: the imports, the wheel's smoke test and the binding
tests. The package lands in `output/`; install it into a fresh environment with

```
conda create -n try -c ./output -c conda-forge pysif
```

## On conda-forge

conda-forge builds from the release's sdist on PyPI, never from a branch, so
a version reaches conda after it reaches PyPI.

The first time, through
[staged-recipes](https://github.com/conda-forge/staged-recipes):

1. Release the version to PyPI as usual.
2. Fork staged-recipes and add `recipes/pysif/recipe.yaml`: this file, with
   `source` replaced by the sdist,

   ```yaml
   source:
     url: https://pypi.org/packages/source/p/pysif/pysif-${{ version }}.tar.gz
     sha256: <sha256 of that file>
   ```

   (`pip download pysif==<version> --no-deps --no-binary :all:`, then
   `shasum -a 256`).
3. Open the pull request. Its CI builds and tests every platform and Python;
   a reviewer from conda-forge then reads it.
4. Once it is merged, conda-forge makes `conda-forge/pysif-feedstock`, with
   the maintainers listed in the recipe, and `conda install -c conda-forge
   pysif` works a few hours later.

After that, each PyPI release brings a pull request to the feedstock from
conda-forge's bot with the new version and checksum, to merge when its CI
passes. Changes to how pysif is built -- a new dependency, a new option --
are made to the feedstock's recipe by hand, and mirrored here so the two
stay the same apart from `source`.

The version in `recipe.yaml` follows CMakeLists.txt; a release whose two
disagree stops at the check in `.github/workflows/release.yml`.
