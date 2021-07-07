.. xmca documentation master file, created by
   sphinx-quickstart on Thu May  6 13:42:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Maximum Covariance Analysis in Python
===========================================
The aim of this package is to provide a flexible tool for the climate
science community to perform Maximum Covariance Analysis (**MCA**) in a
simple and consistent way. Given the huge popularity of
`xarray <http://xarray.pydata.org/en/stable/>`__ in the climate
science community, the ``xmca`` package supports ``xarray.DataArray`` as
well as ``numpy.ndarray`` as input formats.

.. figure:: ../../figs/example-plot2.png
   :alt: Mode 2 of complex rotated Maximum Covariance Analysis showing the shared dynamics of SST and continental precipitation associated to ENSO between 1980 and 2020.

Mode 2 of complex rotated Maximum Covariance Analysis showing the shared 
dynamics of SST and continental precipitation associated to ENSO between
1980 and 2020.

What is MCA?
------------

MCA maximises the temporal covariance between two different data fields
and is closely related to Principal Component Analysis (**PCA**) /
Empirical Orthogonal Function analysis (**EOF analysis**). While EOF
analysis maximises the variance within a single data field, MCA allows
to extract the dominant co-varying patterns between two different data
fields. When the two input fields are the same, MCA reduces to standard
EOF analysis.

For the mathematical understanding please have a look at e.g. the
`lecture
material <https://atmos.washington.edu/~breth/classes/AS552/lect/lect22.pdf>`__
from C. Bretherton.



Documentation
-------------

.. toctree::
   installation
   quickstart
   api



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
