# Canonical Correlation Analysis
Canonical correlation analysis (CCA) seeks to find the patterns in two data fields with maximum amount of cross-correlation.

If both data fields are equal, CCA basically reduces to Principal Component Analysis (PCA), in climate science often called **EOF analysis**.

This packages works with `np.ndarray` and `xarray.DataArray` as input fields.

## Features
- maximise covariance instead of correlation ==> Maximum Covariance Analysis (MCA)
- apply latitute correction to data fields to compensate for stretched areas in higher latitutes
- apply rotation of singular vectors
  - Orthogonal Varimax rotation
  - Oblique Promax rotation
- complexify data via Hilbert transform to inspect amplitude and phase information
