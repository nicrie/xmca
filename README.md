# xMCA | Maximum Covariance Analysis in Python

[![version](https://img.shields.io/pypi/v/xmca?color=f2cc8f&label=PyPI)](https://pypi.org/project/xmca/)
![CircleCI](https://img.shields.io/circleci/build/github/nicrie/xmca?color=81b29a)
[![Documentation Status](https://img.shields.io/readthedocs/xmca/latest?color=81b29a)](https://pyxmca.readthedocs.io/en/latest/?badge=latest)
![Maintenance](https://img.shields.io/maintenance/yes/2021?color=81b29a)
[![downloads](https://img.shields.io/pypi/dm/xmca?color=f2cc8f)](https://pypi.org/project/xmca/)

The aim of this package is to provide a flexible tool for the climate science community to perform Maximum Covariance Analysis (**MCA**) in a simple and consistent way. Given the huge popularity of [`xarray`][xarray] in the climate science community, the `xmca` package supports `xarray.DataArray` as well as `numpy.ndarray` as input formats.

## What is MCA?
MCA maximises the temporal covariance between two different
data fields and is closely related to Principal Component Analysis (**PCA**) / Empirical
Orthogonal Function analysis (**EOF analysis**). While EOF analysis maximises the variance within a single data
field, MCA allows to extract the dominant co-varying patterns between two different data
fields. When the two input fields are the same, MCA reduces to standard EOF analysis.

For the mathematical understanding please have a look at e.g. the [lecture material][mca-material] from C. Bretherton.


## Core Features
##### Pre-processing
- Normalisation
- Spatial weighting to correct for latitude bias
##### EOF analysis (PCA)
- [standard EOF][pca] analysis
- [rotated EOF][rotated-pca] analysis
	- Orthogonal [Varimax][varimax] rotation
	- Oblique [Promax][promax] rotation
- [Complex EOF][complex-pca] analysis (also known as Hilbert EOF analysis)
	- Optimised [Theta model][theta] extension
  - *New in v0.2.1:* Exponential extension
##### MCA
-  [standard MCA][mca]
- [rotated MCA][rotated-mca]
	- Orthogonal [Varimax][varimax] rotation
	- Oblique [Promax][promax] rotation
- Complex MCA (paper submitted, arXiv preprint)
	- Optimised [Theta model][theta] extension
  - *New in v0.2.1:* Exponential extension

##### Results
- eigenvalues / singular values
- explained variance
- EOFs (spatial patterns)
- PCs (temporal evolution)
- Heterogeneous/Homogeneous patterns
- If rotated: rotation and PC correlation matrix
- If complex: spatial amplitude/phase

##### Convenience
- plotting function
- saving/loading performed analyses

I'm currently working on a more detailed [documentation page](https://pyxmca.readthedocs.io/en/latest/index.html). Please have a look there for the entire API reference.


## Installation
Installation is simply performed via
```
pip install xmca
```

##### Known Issues
Actually `pip` should take care of installing the correct dependencies. However, the dependencies of `cartopy` itself are not installed via `pip` which is
why the setup may fail in some cases. If so, please
[install][cartopy] `cartopy` first before installing `xmca`. If you are using a `conda` environment, this can be achieved by
```
conda install cartopy
```


##### Testing
After cloning the repository
```
python -m unittest discover -v -s tests/
```



## Quickstart
Import the module for `xarray` via
```py
from xmca.xarray import xMCA
```
Create some dummy data, which should be of type `xr.DataArray`.
```py
import numpy as np
import xarray as xr

n_time = 300                 # number of time steps
lat1, lon1 = 20, 30     # number of latitudes/longitudes of field A
lat2, lon2 = 15, 10     # number of latitudes/longitudes of field B
A = xr.DataArray(np.random.randn(n_time, lat1, lon1)) # dummy field A
B = xr.DataArray(np.random.randn(n_time, lat1, lon2)) # dummy field B
```

### Principal Component Analysis
```py
pca = xMCA(A)                       # PCA on field A
pca.solve(complexfify=False)        # True for complex PCA
#pca.rotate(10)                     # optional; Varimax rotated solution
                                    # using 10 first EOFs
eigenvalues = pca.singular_values() # singular vales = eigenvalues for PCA
pcs         = pca.pcs()             # Principal component scores (PCs)
eofs        = pca.eofs()            # spatial patterns (EOFs)

```

### Maximum Covariance Analysis
```py
mca = xMCA(A,B)                     # MCA of field A and B
mca.solve(complexfify=False)        # True for complex MCA
#mca.rotate(10)                     # optional; Varimax rotated solution
                                    # using 10 first EOFs
eigenvalues = mca.singular_values() # singular vales
pcs = mca.pcs()                     # expansion coefficient (PCs)
eofs = mca.eofs()                   # spatial patterns (EOFs)

```
### Save/load an analysis
```py
mca.save_analysis()                 # this will save the data and a respective
                                    # info file. The files will be stored in a
                                    # special directory
mca2 = xMCA()                       # create a new, empty instance
mca2.load_analysis('./mca/left_right/mca_c0_r00_p00.info') # analysis can be
                                    # loaded via specifying the path to the
                                    # info file created earlier
mca2.plot(mode=1)
```
### Plot your results
The package provides a method to visually inspect the individual modes, e.g. for mode 2.

*Note: The following plots use real data (ERA5 SST & precipitation) instead of the toy data shown at the beginning of the tutorial. Apart from that the figures show exactly what is produced by calling the convenience plotting method.*
```py
mca2.set_field_names('SST', 'Precipitation')  # add variable names, optional
mca2.plot(mode=2)
```
![example-plot1](figs/example-plot1.png "Result of default plot method after performing complex rotated MCA on SST and precipitation showing mode 2")

You may want to modify the plot for some better optics:
```py
import cartopy.crs as ccrs  # for different map projections

# map projections for "left" and "right" field
projections = {
    'left': ccrs.EqualEarth(central_longitude=200),
    'right': ccrs.EqualEarth(central_longitude=160)
}

plot_kwargs = {
    "figsize"     : (8, 5),
    "threshold"   : 0.25,       # mask out values < 0.25 max-normalised amplitude
    "orientation" : 'vertical',
    'cmap_eof'    : 'viridis',  # colormap amplitude
    'cmap_phase'  : 'twilight', # colormap phase
    "phase_shift" : 2.2,        # apply phase shift to PCs
    "projection"  : projections,
}
mca2.plot(mode=2, **plot_kwargs)
```

![example-plot2](figs/example-plot2.png "Result of plot method with improved optics after performing complex rotated MCA on SST and precipitation showing mode 2.")

You can save the plot to your local disk as a `.png` file via
```py
save_kwargs={'dpi':200, 'transparent':True}
mca2.save_plot(mode=2, plot_kwargs=plot_kwargs, save_kwargs=save_kwargs)
```

## Credits
Kudos to the developers and contributors of the following Github projects which I initially used myself but had to expand to my needs for my own research:
* [ajdawson/eofs](https://github.com/ajdawson/eofs)
* [Yefee/xMCA](https://github.com/Yefee/xMCA)

[xarray]: http://xarray.pydata.org/en/stable/

[cartopy]: https://scitools.org.uk/cartopy/docs/latest/installing.html

[pca]: https://en.wikipedia.org/wiki/Empirical_orthogonal_functions

[mca]: ftp://eos.atmos.washington.edu/pub/breth/papers/1992/SVD-theory.pdf

[mca-material]: https://atmos.washington.edu/~breth/classes/AS552/lect/lect22.pdf

[rotated-pca]: https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/empirical-orthogonal-function-eof-analysis-and-rotated-eof-analysis

[rotated-mca]: https://journals.ametsoc.org/jcli/article/8/11/2631/35764/Orthogonal-Rotation-of-Spatial-Patterns-Derived

[varimax]: https://en.wikipedia.org/wiki/Varimax_rotation

[promax]: https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-8317.1964.tb00244.x

[complex-pca]: https://journals.ametsoc.org/doi/abs/10.1175/1520-0450(1984)023%3C1660%3ACPCATA%3E2.0.CO%3B2

[theta]: https://linkinghub.elsevier.com/retrieve/pii/S0169207016300243
