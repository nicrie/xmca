Quickstart
==========
Import the module for `xarray` via

.. code-block:: python

   from xmca.xarray import xMCA


Create some dummy data, which should be of type ``xarrar.DataArray``.

.. code-block:: python

   import numpy as np
   import xarray as xr

   n_time = 300
   n_lat1, n_lon1 = 20, 30
   n_lat2, n_lon2 = 15, 10
   A = xr.DataArray(np.random.randn(t, n_lat1, n_lon1))   # dummy field A
   B = xr.DataArray(np.random.randn(t, n_lat2, n_lon2))   # dummy field B


Principal Component Analysis
----------------------------

.. code-block:: python

   pca = xMCA(A)                        # PCA on field A
   pca.solve(complexfify=False)         # True for complex PCA
   pca.rotate(10)                       # optional; Varimax rotated solution
	                                      # using 10 first EOFs
   eigenvalues = pca.singular_values()  # singular vales = eigenvalues for PCA
   pcs         = pca.pcs()              # Principal component scores (PCs)
   eofs        = pca.eofs()             # spatial patterns (EOFs)



Maximum Covariance Analysis
---------------------------

.. code-block:: python

   mca = xMCA(A,B)                     # MCA of field A and B
   mca.solve(complexfify=False)        # True for complex MCA
   #mca.rotate(10)                     # optional; Varimax rotated solution
                                       # using 10 first EOFs
   eigenvalues = mca.singular_values() # singular vales
   pcs = mca.pcs()                     # expansion coefficient (PCs)
   eofs = mca.eofs()                   # spatial patterns (EOFs)



Save/load an analysis
---------------------

.. code-block:: python

   mca.save_analysis()                 # this will save the data and a respective
                                       # info file. The files will be stored in a
                                       # special directory
   info = './mca/left_right/mca_c0_r00_p00.info'
   mca2 = xMCA()                       # create a new, empty instance
   mca2.load_analysis(info)            # analysis can be
                                       # loaded via specifying the path to the
                                       # info file created earlier



Plot your results
-----------------
The package provides a method to visually inspect the individual modes, e.g. for mode 2.

*Note: The following plots use real data (ERA5 SST & precipitation) instead of the toy data shown at the beginning of the tutorial. Apart from that the figures show exactly what is produced by calling the convenience plotting method.*

.. code-block:: python

   mca2.set_field_names('SST', 'Precipitation')  # add variable names, optional
   mca2.plot(mode=2)

.. image:: ../../figs/example-plot1.png

You may want to modify the plot for some better optics:

.. code-block:: python

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


.. image:: ../../figs/example-plot2.png

You can save the plot to your local disk as a `.png` file via

.. code-block:: python

   save_kwargs={'dpi':200, 'transparent':True}
   mca2.save_plot(mode=2, plot_kwargs=plot_kwargs, save_kwargs=save_kwargs)
