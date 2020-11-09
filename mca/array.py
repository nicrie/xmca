#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex rotated maximum covariance analysis of two numpy arrays.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import textwrap
from scipy.signal import hilbert
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm

from tools.rotation import promax

# =============================================================================
# MCA
# =============================================================================
class MCA(object):
    """Perform maximum covariance analysis (MCA) for two `np.ndarray` data fields.

    MCA is principal component analysis (PCA) generalized
    for two input fields (left, right). If both data fields are the same,
    it is equivalent to PCA.

    Parameters
    ----------
    left : ndarray
        Left input data. First dimension needs to be time.
    right : ndarray, optional
        Right input data. First dimension needs to be time.
        If none is provided, automatically, right field is assumed to be
        the same as left field. In this case, MCA reducdes to normal PCA.
        The default is None.
    normalize : boolean, optional
        Input data is normalized to unit variance. The default is True.


    Examples
    --------
    Let `data1` and `data2` be some geophysical fields (e.g. SST and pressure).
    To perform PCA use:

    >>> pca = MCA(data1)
    >>> pca.solve()
    >>> pcs,_ = pca.pcs()

    To perform MCA use:

    >>> mca = MCA(data1, data2)
    >>> mca.solve()
    >>> pcsData1, pcsData2 = mca.pcs()
    """

    def __init__(self, left, right=None, normalize=True):
        """Load data fields and store information about data size/shape.

        Parameters
        ----------
        left : ndarray
            Left input data. First dimension needs to be time.
        right : ndarray, optional
            Right input data. First dimension needs to be time.
            If none is provided, automatically, right field is assumed to be
            the same as left field. In this case, MCA reducdes to normal PCA.
            The default is None.
        normalize : boolean, optional
            Input data is normalized to unit variance. The default is True.


        """
        self._left      = left.copy()
        self._right     = self._left if right is None else right.copy()
        self._useMCA    = not self._isSameArray(self._left, self._right)

        assert(self._isArray(self._left))
        assert(self._isArray(self._right))
        assert(self._hasSameTimeDimensions(self._left, self._right))


        self._observations          = self._left.shape[0]
        self._originalShapeLeft     = self._left.shape[1:]
        self._originalShapeRight 	= self._right.shape[1:]

        self._variablesLeft 		= np.product(self._originalShapeLeft)
        self._variablesRight 		= np.product(self._originalShapeRight)

        # create 2D matrix in order to perform PCA
        self._left 	    = self._left.reshape(self._observations,self._variablesLeft)
        self._right 	= self._right.reshape(self._observations,self._variablesRight)

        # check for NaN time steps
        assert(self._hasNoNanTimeSteps(self._left))
        assert(self._hasNoNanTimeSteps(self._right))

        # center input data to zero mean (remove mean)
        self._left 	    = self._centerArray(self._left)
        self._right 	= self._centerArray(self._right)

        # normalize input data to unit variance
        if (normalize):
            self._left  = self._normalizeArray(self._left)
            self._right = self._normalizeArray(self._right)

        # remove NaNs columns in data fields
        self._noNanDataLeft, self._noNanIndexLeft   = self._removeNanColumns(self._left)
        self._noNanDataRight, self._noNanIndexRight = self._removeNanColumns(self._right)

        assert(self._isNotEmpty(self._noNanDataLeft))
        assert(self._isNotEmpty(self._noNanDataRight))

        # meta information on rotation
        self._rotatedSolution   = False
        self._nRotations        = 0
        self._power             = 0


    def _removeNanColumns(self, array):
        noNanIndex = np.where(~(np.isnan(array[0])))[0]
        noNanData  = array[:,noNanIndex]
        return noNanData, noNanIndex


    def _isSameArray(self, arr1 ,arr2):
        if arr1.shape == arr2.shape:
            return ((np.isnan(arr1) & np.isnan(arr2)) | (arr1 == arr2)).all()
        else:
            return False


    def _isArray(self,data):
        if (isinstance(data,np.ndarray)):
            return True
        else:
            raise TypeError('Data needs to be np.ndarray.')


    def _hasSameTimeDimensions(self, left, right):
        if (left.shape[0] == right.shape[0]):
            return True
        else:
            raise ValueError('Both input fields need to have same time dimensions.')


    def _centerArray(self, array):
        """Remove the mean of an array along the first dimension."""
        return array - array.mean(axis=0)


    def _normalizeArray(self, array):
        """Normalize the array along the first dimension (divide by std)."""
        return array / array.std(axis=0)


    def _hasNoNanTimeSteps(self, data):
        """Check if data contains a nan time step.

        A nan time step is a time step for which the values of every station
        (column) in	the data is np.NaN. Nan time steps are problematic
        since centering of the data transforms all values of a station (column)
        to np.NaN if only one np.NaN is present.
        """
        if (np.isnan(data).all(axis=1).any()):
            raise ValueError(textwrap.fill(textwrap.dedent("""
            Gaps (np.NaN) in time series detected. Either remove or interpolate
            all NaN time steps in your data.""")))
        else:
            return True


    def _isNotEmpty(self, index):
        if (index.size > 0):
            return True
        else:
            raise ValueError('Input field is empty or contains NaN only.')


    def _thetaForecast(self, series, steps=None, seasonalPeriod=365):
        if steps is None:
            steps = len(series)

        model = ThetaModel(series, period=seasonalPeriod, deseasonalize=True, use_test=False).fit()
        forecast = model.forecast(steps=steps, theta=20)

        return forecast


    def _extendData(self, data, seasonalPeriod=365):

        extendedData = [self._thetaForecast(col, seasonalPeriod=seasonalPeriod) for col in tqdm(data.T)]
        extendedData = np.array(extendedData).T

        return extendedData


    def _complexifyData(self, data, extendSeries=False, seasonalPeriod=365):
        """Complexify data via Hilbert transform.

        Calculating Hilbert transform via scipy.signal.hilbert is done
        through Fast Fourier Transform. If the time series exhibits some
        non-periodic behaviour (e.g. a trend) the Hilbert transform
        produces extreme "legs" at the beginning/end of the time series.
        To encounter this issue, we can forecast/backcast the original time
        series via the Theta model before applying the Hilbert transform.
        Then, we only take the middle part of the Hilbert transform
        (corresponding to the original time series) which exhibits
        a dampened influence of the "legs".

        Parameters
        ----------
        data : ndarray
            Real input data which is to be transformed via Hilbert transform.
        extendSeries : boolean, optional
            If True, input time series are extended via forecast/backcast to
            3 * original length. This helps avoiding boundary effects of FFT.

        Returns
        -------
        ndarray
            Analytical signal of input data.

        """

        if extendSeries:
            forecast    = self._extendData(data, seasonalPeriod=seasonalPeriod)
            backcast    = self._extendData(data[::-1], seasonalPeriod=seasonalPeriod)[::-1]

            data = np.concatenate([backcast, data, forecast])

        # perform actual Hilbert transform of (extended) time series
        data = hilbert(data,axis=0)

        if extendSeries:
            # cut out the first and last third of Hilbert transform
            # which belong to the forecast/backcast
            data    = data[self._observations:(2*self._observations)]
            data = self._centerArray(data)

        return data


    def solve(self, useHilbert=False, extendSeries=False, seasonalPeriod=365):
        """Solve eigenvalue equation by performing SVD on covariance matrix.

        Parameters
        ----------
        useHilbert : boolean, optional
            Use Hilbert transform to complexify the input data fields
            in order to perform complex PCA/MCA. Default is false.
        extendSeries : boolean, optional
            If True, extend time series by fore/backcasting based on
            Theta model. New time series will have 3 * original length.
            Only used for complex time series (useHilbert=True).
            Default is False.
        """
        self._useHilbert = useHilbert
        # complexify input data via Hilbert transform
        if (self._useHilbert):
            self._noNanDataLeft = self._complexifyData(self._noNanDataLeft, extendSeries=extendSeries, seasonalPeriod=seasonalPeriod)
            # save computing time if left and right field are the same
            if self._useMCA:
                self._noNanDataRight = self._complexifyData(self._noNanDataRight, extendSeries=extendSeries, seasonalPeriod=seasonalPeriod)
            else:
                self._noNanDataRight = self._noNanDataLeft

        # create covariance matrix
        kernel = self._noNanDataLeft.conjugate().T @ self._noNanDataRight / self._observations

        # solve eigenvalue problem
        VLeft, eigenvalues, VTRight = np.linalg.svd(kernel, full_matrices=False)
        VRight = VTRight.conjugate().T

        S = np.sqrt(np.diag(eigenvalues) * self._observations)
        Si = np.diag(1./np.diag(S))

        self._eigenvalues = eigenvalues
        self._eigensum = eigenvalues.sum()

        # standardized EOF fields
        self._VLeft 	= VLeft
        self._VRight 	= VRight

        # loadings // EOF fields
        self._LLeft 	= VLeft @ S
        self._LRight 	= VRight @ S

        # get PC scores by projecting data fields on loadings
        self._ULeft 	= self._noNanDataLeft @ VLeft @ Si
        self._URight 	= self._noNanDataRight @ VRight @ Si


    def rotate(self, nRotations, power=1, tol=1e-5):
        """Perform Promax rotation on the first `n` EOFs.

        Promax rotation (Hendrickson & White 1964) is an oblique rotation which
        seeks to find `simple structures` in the EOFs. It transforms the EOFs
        via an orthogonal Varimax rotation (Kaiser 1958) followed by the Promax
        equation. If `power=1`, Promax reduces to Varimax rotation. In general,
        a Promax transformation breaks the orthogonality of EOFs and introduces
        some correlation between PCs.

        Parameters
        ----------
        nRotations : int
            Number of EOFs to rotate.
        power : int, optional
            Power of Promax rotation. The default is 1.
        tol : float, optional
            Tolerance of rotation process. The default is 1e-5.

        Raises
        ------
        ValueError
            If number of rotations are <2.

        Returns
        -------
        None.

        """
        if(nRotations < 2):
            raise ValueError('nRotations must be >=2')
        if(power<1):
            raise ValueError('Power must be >=1')

        # rotate loadings (Cheng and Dunkerton 1995)
        L = np.concatenate((self._LLeft[:,:nRotations], self._LRight[:,:nRotations]))
        Lr, R, Phi = promax(L, power, maxIter=1000, tol=tol)
        LLeft 	= Lr[:self._VLeft.shape[0],:]
        LRight 	= Lr[self._VLeft.shape[0]:,:]

        # calculate variance/reconstruct "eigenvalues"
        wLeft = np.linalg.norm(LLeft,axis=0)
        wRight = np.linalg.norm(LRight,axis=0)
        variance = wLeft * wRight / self._observations
        varIdx = np.argsort(variance)[::-1]

        # pull loadings from EOFs
        VLeft 	= LLeft / wLeft
        VRight 	= LRight / wRight

        # rotate PC scores
        # If rotation is orthogonal: R.T = R
        # If rotation is oblique (p>1): R^(-1).T = R
        if(power==1):
            ULeft 	= self._ULeft[:,:nRotations] @ R
            URight 	= self._URight[:,:nRotations] @ R
        else:
            ULeft 	= self._ULeft[:,:nRotations] @ np.linalg.pinv(R).conjugate().T
            URight 	= self._URight[:,:nRotations] @ np.linalg.pinv(R).conjugate().T


        # store rotated pcs, eofs and "eigenvalues"
        # and sort according to described variance
        self._eigenvalues 	= variance[varIdx]
        # Standardized EOFs
        self._VLeft 		= VLeft[:,varIdx]
        self._VRight 		= VRight[:,varIdx]
        # EOF loadings
        self._LLeft 		= LLeft[:,varIdx]
        self._LRight 		= LRight[:,varIdx]
        # Standardized PC scores
        self._ULeft 		= ULeft[:,varIdx]
        self._URight 		= URight[:,varIdx]

        # store rotation and correlation matrix of PCs + meta information
        self._rotationMatrix 		= R
        self._correlationMatrix 	= Phi[varIdx,varIdx]
        self._rotatedSolution 		= True
        self._nRotations            = nRotations
        self._power                 = power


    def rotationMatrix(self):
        """
        Return the rotation matrix.

        Returns
        -------
        ndarray
            Rotation matrix.
        """
        if (self._rotatedSolution):
            return self._rotationMatrix
        else:
            raise RuntimeError('Rotation matrix does not exist since EOFs were not rotated')


    def correlationMatrix(self):
        """
        Return the correlation matrix of rotated PCs.

        Returns
        -------
        ndarray
            Correlation matrix.

        """
        if (self._rotatedSolution):
            return self._correlationMatrix
        else:
            raise RuntimeError('Correlation matrix does not exist since EOFs were not rotated.')


    def eigenvalues(self,n=None):
        """Return the first `n` eigenvalues.

        Parameters
        ----------
        n : int, optional
            Number of eigenvalues to return. The default is 5.

        Returns
        -------
        values : ndarray
            Eigenvalues of PCA.
        error : ndarray
            Uncertainty of eigenvalues according to North's rule of thumb.

        """
        values = self._eigenvalues[:n]
        # error according to North's Rule of Thumb
        error = np.sqrt(2/self._observations) * values

        return values, error


    def explainedVariance(self, n=None):
        """Return the described variance of the first `n` PCs.

        Parameters
        ----------
        n : int, optioal
            Number of PCs to return. The default is None.

        Returns
        -------
        desVar : ndarray
            Described variance of each PC.
        desVarErr : ndarray
            Associated uncertainty according to North's `rule of thumb`.

        """
        values, error = self.eigenvalues(n)
        desVar 		= values / self._eigensum * 100
        desVarErr 	= error / self._eigensum * 100
        return desVar, desVarErr


    def pcs(self, n=None, scaling=0):
        """Return the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of PCs to be returned. The default is None.
        scaling : [0,1], optional
            If 1, scale PCs by square root of eigenvalues. If 0, return
            unscaled PCs. The default is 0.

        Returns
        -------
        pcsLeft : ndarray
            PCs associated with left input field.
        pcsRight : ndarray
            PCs associated with right input field.

        """
        pcsLeft 	= self._ULeft[:,:n]
        pcsRight 	= self._URight[:,:n]

        if (scaling==1):
            pcsLeft 	= pcsLeft * np.sqrt(self._eigenvalues[:n])
            pcsRight 	= pcsRight * np.sqrt(self._eigenvalues[:n])

        return pcsLeft, pcsRight


    def eofs(self, n=None, scaling=0):
        """Return the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to be returned. The default is None.
        scaling : [0,1], optional
            If 1, scale PCs by square root of eigenvalues. If 0, return
            unscaled PCs. The default is 0.

        Returns
        -------
        eofsLeft : ndarray
            EOFs associated with left input field.
        eofsRight : ndarray
            EOFs associated with right input field.

        """
        if n is None:
            n = self._eigenvalues.size

        # create data fields with original NaNs
        dtype = self._VLeft.dtype
        eofsLeft  	= np.zeros([self._variablesLeft, n],dtype=dtype) * np.nan
        eofsRight  	= np.zeros([self._variablesRight, n],dtype=dtype) * np.nan

        eofsLeft[self._noNanIndexLeft,:] = self._VLeft[:,:n]
        eofsRight[self._noNanIndexRight,:] = self._VRight[:,:n]

        # reshape data fields to have original input shape
        eofsLeft 	= eofsLeft.reshape(self._originalShapeLeft + (n,))
        eofsRight 	= eofsRight.reshape(self._originalShapeRight + (n,))

        if (scaling==1):
            eofsLeft 	= eofsLeft * np.sqrt(self._eigenvalues[:n])
            eofsRight 	= eofsRight * np.sqrt(self._eigenvalues[:n])

        return eofsLeft, eofsRight


    def spatialAmplitude(self, n=None):
        """Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to be returned. If None, return all fields. The default is None.

        Returns
        -------
        ndarray
            Amplitude fields of left input field.
        ndarray
            Amplitude fields of right input field.

        """
        eofsLeft, eofsRight = self.eofs(n)

        amplitudeLeft   = np.sqrt(eofsLeft * eofsLeft.conjugate())
        amplitudeRight  = np.sqrt(eofsRight * eofsRight.conjugate())

        # use the real part to force a real output
        return amplitudeLeft.real, amplitudeRight.real


    def spatialPhase(self, n=None):
        """Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        ndarray
            Fields of left input field.
        ndarray
            Fields of right input field.

        """
        eofsLeft, eofsRight = self.eofs(n)

        phaseLeft = np.arctan2(eofsLeft.imag,eofsLeft.real)
        phaseRight = np.arctan2(eofsRight.imag,eofsRight.real)

        # use the real part to force a real output
        return phaseLeft.real, phaseRight.real


    def temporalAmplitude(self, n=None):
        """Return the temporal amplitude time series for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude series to be returned. If None, return all series.
            The default is None.

        Returns
        -------
        ndarray
            Amplitude time series of left input field.
        ndarray
            Amplitude time series of right input field.

        """
        pcsLeft, pcsRight = self.pcs(n)

        amplitudeLeft   = np.sqrt(pcsLeft * pcsLeft.conjugate())
        amplitudeRight  = np.sqrt(pcsRight * pcsRight.conjugate())

        # use the real part to force a real output
        return amplitudeLeft.real, amplitudeRight.real


    def temporalPhase(self, n=None):
        """Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, return all series.
            The default is None.

        Returns
        -------
        ndarray
            Phase function of left input field.
        ndarray
            Phase function of right input field.

        """
        pcsLeft, pcsRight = self.pcs(n)

        phaseLeft = np.arctan2(pcsLeft.imag,pcsLeft.real)
        phaseRight = np.arctan2(pcsRight.imag,pcsRight.real)

        # use the real part to force a real output
        return phaseLeft.real, phaseRight.real


    def loadAnalysis(self, eofs=None, pcs=None, eigenvalues=None):
        # standardized fields // EOF fields + PCs
        if all(isinstance(var,list) for var in [eofs,pcs]):
            eofsLeft, eofsRight = [eofs[0], eofs[1]]
            pcsLeft, pcsRight   = [pcs[0], pcs[1]]
        else:
            eofsLeft, eofsRight = [eofs, eofs]
            pcsLeft, pcsRight   = [pcs, pcs]

        self._observations          = pcsLeft.shape[0]
        self._originalShapeLeft     = eofsLeft.shape[:-1]
        self._originalShapeRight 	= eofsRight.shape[:-1]
        nModes                      = eofsRight.shape[-1]

        self._variablesLeft 		= np.product(self._originalShapeLeft)
        self._variablesRight 		= np.product(self._originalShapeRight)

        eofsLeft    = eofsLeft.reshape(self._variablesLeft, nModes)
        eofsRight   = eofsRight.reshape(self._variablesRight, nModes)

        VLeftT, self._noNanIndexLeft   = self._removeNanColumns(eofsLeft.T)
        VRightT, self._noNanIndexRight = self._removeNanColumns(eofsRight.T)
        self._VLeft     = VLeftT.T
        self._VRight    = VRightT.T

        S   = np.sqrt(np.diag(eigenvalues) * self._observations)
        Si  = np.diag(1./np.diag(S))

        self._eigenvalues   = eigenvalues
        self._eigensum      = eigenvalues.sum()

        self._ULeft, self._URight = [pcsLeft, pcsRight]

        # loadings // EOF fields
        self._LLeft 	= self._VLeft @ S
        self._LRight 	= self._VRight @ S
