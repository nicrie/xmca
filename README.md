# Maximum Covariance Analysis in Python
Maximum Covariance Analysis (MCA) maximises the temporal covariance between two different 
data fields and is closely related to Principal Component Analysis (PCA) / Empirical 
Orthogonal Function (EOF) analysis, which maximises the variance within a single data 
field. MCA allows to extract the dominant co-varying patterns between two different data 
fields.


The module `xmca` works with `numpy.ndarray` and `xarray.DataArray` as input fields.

## Core Features
- Standard MCA/PCA
- maximise covariance instead of correlation ==> Maximum Covariance Analysis (MCA)
- apply latitute correction to data fields to compensate for stretched areas in higher latitutes
- apply rotation of singular vectors
  - Orthogonal Varimax rotation
  - Oblique Promax rotation
- complexify data via Hilbert transform to inspect amplitude and phase information
