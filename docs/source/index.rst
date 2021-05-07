.. xmca documentation master file, created by
   sphinx-quickstart on Thu May  6 13:42:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xmca: Maximum Covariance Analysis in Python
===========================================
Maximum Covariance Analysis (MCA) maximises the temporal covariance between two different
data fields and is closely related to Principal Component Analysis (PCA) / Empirical
Orthogonal Function (EOF) analysis, which maximises the variance within a single data
field. MCA allows to extract the dominant co-varying patterns between two different data
fields.

The module ``xmca`` works with ``numpy.ndarray`` and ``xarray.DataArray`` as input fields.

Core Features
-------------
- Standard PCA/`MCA`_
- PCA/MCA with rotation_
	- Orthogonal Varimax_ rotation
	- Oblique Promax_ rotation
- `Complex PCA`_/MCA (also known as Hilbert EOF analysis)
	- Optimised `Theta model`_ extension
- normalization of input data
- latitude correction to compensate for stretched areas in higher latitutes



.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   quickstart
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`MCA`: ftp://eos.atmos.washington.edu/pub/breth/papers/1992/SVD-theory.pdf

.. _rotation: https://journals.ametsoc.org/jcli/article/8/11/2631/35764/Orthogonal-Rotation-of-Spatial-Patterns-Derived

.. _Varimax: https://en.wikipedia.org/wiki/Varimax_rotation

.. _Promax: https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-8317.1964.tb00244.x

.. _`Complex PCA`: https://journals.ametsoc.org/doi/abs/10.1175/1520-0450(1984)023%3C1660%3ACPCATA%3E2.0.CO%3B2

.. _`Theta model`: https://linkinghub.elsevier.com/retrieve/pii/S0169207016300243


