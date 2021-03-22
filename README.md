# Maximum Covariance Analysis in Python
Maximum Covariance Analysis (MCA) maximises the temporal covariance between two different 
data fields and is closely related to Principal Component Analysis (PCA) / Empirical 
Orthogonal Function (EOF) analysis, which maximises the variance within a single data 
field. MCA allows to extract the dominant co-varying patterns between two different data 
fields.


The module `xmca` works with `numpy.ndarray` and `xarray.DataArray` as input fields.

## Installation 
```
pip install xmca
```

## Testing
After cloning the repository
```
python -m unittest discover -v -s tests/
```

## Core Features
- Standard PCA/MCA
- Rotated PCA/MCA
	- Orthogonal Varimax rotation
	- Oblique Promax rotation
- Complex PCA/MCA (also known as Hilbert EOF analysis)
	- Optimised Theta model extension
- normalization of input data
- latitude correction to compensate for stretched areas in higher latitutes

