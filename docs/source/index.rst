.. xmca documentation master file, created by
   sphinx-quickstart on Thu May  6 13:42:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Maximum Covariance Analysis in Python
===========================================
The aim of this package is to provide a flexible tool for the climate
science community to perform Maximum Covariance Analysis (**MCA**) in a
simple and consistent way. Given the huge popularity of
```xarray`` <http://xarray.pydata.org/en/stable/>`__ in the climate
science community, the ``xmca`` package supports ``xarray.DataArray`` as
well as ``numpy.ndarray`` as input formats.

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

Core Features
-------------

Pre-processing
''''''''''''''

-  Normalisation
-  Spatial weighting to correct for latitude bias
   (PCA)

EOF analysis
''''''''''''

-  `standard
   EOF <https://en.wikipedia.org/wiki/Empirical_orthogonal_functions>`__
   analysis
-  `rotated
   EOF <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/empirical-orthogonal-function-eof-analysis-and-rotated-eof-analysis>`__
   analysis

   -  Orthogonal
      `Varimax <https://en.wikipedia.org/wiki/Varimax_rotation>`__
      rotation
   -  Oblique
      `Promax <https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-8317.1964.tb00244.x>`__
      rotation

-  `Complex
   EOF <https://journals.ametsoc.org/doi/abs/10.1175/1520-0450(1984)023%3C1660%3ACPCATA%3E2.0.CO%3B2>`__
   analysis (also known as Hilbert EOF analysis)

   -  Optimised `Theta
      model <https://linkinghub.elsevier.com/retrieve/pii/S0169207016300243>`__
      extension

-  *New in v0.2.1:* Exponential extension

MCA
'''

-  `standard
   MCA <ftp://eos.atmos.washington.edu/pub/breth/papers/1992/SVD-theory.pdf>`__
-  `rotated
   MCA <https://journals.ametsoc.org/jcli/article/8/11/2631/35764/Orthogonal-Rotation-of-Spatial-Patterns-Derived>`__

   -  Orthogonal
      `Varimax <https://en.wikipedia.org/wiki/Varimax_rotation>`__
      rotation
   -  Oblique
      `Promax <https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-8317.1964.tb00244.x>`__
      rotation

-  Complex MCA (paper submitted, arXiv preprint)

   -  Optimised `Theta
      model <https://linkinghub.elsevier.com/retrieve/pii/S0169207016300243>`__
      extension

-  *New in v0.2.1:* Exponential extension

Results
'''''''

-  eigenvalues / singular values
-  explained variance
-  EOFs (spatial patterns)
-  PCs (temporal evolution)
-  Heterogeneous/Homogeneous patterns
-  If rotated: rotation and PC correlation matrix
-  If complex: spatial amplitude/phase

Convenience
'''''''''''

-  plotting function
-  saving/loading performed analyses

I'm currently working on a more detailed `documentation
page <https://pyxmca.readthedocs.io/en/latest/index.html>`__. Please
have a look there for the entire API reference.

Documentation
-------------
.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
