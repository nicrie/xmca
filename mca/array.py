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
        self.__left  = left.copy()
        self.__right = self.__getRightField(right)
        self.__useMCA = (self.__left is not self.__right)

        assert(self.__isArray(self.__left))
        assert(self.__isArray(self.__right))
        assert(self.__hasSameTimeDimensions(self.__left, self.__right))


        self.__observations 		= self.__left.shape[0]
        self.__originalShapeLeft 	= self.__left.shape[1:]
        self.__originalShapeRight 	= self.__right.shape[1:]

        self.__variablesLeft 		= np.product(self.__originalShapeLeft)
        self.__variablesRight 		= np.product(self.__originalShapeRight)

        # create 2D matrix in order to perform PCA
        self.__left 	= self.__left.reshape(self.__observations,self.__variablesLeft)
        self.__right 	= self.__right.reshape(self.__observations,self.__variablesRight)

        # check for NaN time steps
        assert(self.__hasNoNanTimeSteps(self.__left))
        assert(self.__hasNoNanTimeSteps(self.__right))

        # center input data to zero mean (remove mean)
        self.__left 	= self.__centerData(self.__left)
        self.__right 	= self.__centerData(self.__right)

        # normalize input data to unit variance
        if (normalize):
            self.__left  = self.__normalizeData(self.__left)
            self.__right = self.__normalizeData(self.__right)

        # remove NaNs in data fields
        self.__noNanIndexLeft 	= np.where(~(np.isnan(self.__left[0])))[0]
        self.__noNanIndexRight 	= np.where(~(np.isnan(self.__right[0])))[0]

        self.__noNanDataLeft 	= self.__left[:,self.__noNanIndexLeft]
        self.__noNanDataRight 	= self.__right[:,self.__noNanIndexRight]

        assert(self.__isNotEmpty(self.__noNanDataLeft))
        assert(self.__isNotEmpty(self.__noNanDataRight))

        self.__rotatedSolution = False

    def __getRightField(self,right):
        """Copy left field if no right field is provided.

        Basically, this defines whether MCA or PCA is performed.
        """
        if right is None:
            return self.__left
        else:
            return right.copy()


    def __isArray(self,data):
        if (isinstance(data,np.ndarray)):
            return True
        else:
            raise TypeError('Data needs to be np.ndarray.')


    def __hasSameTimeDimensions(self, left, right):
        if (left.shape[0] == right.shape[0]):
            return True
        else:
            raise ValueError('Both input fields need to have same time dimensions.')


    def __centerData(self, data, normalize=False):
        """Remove the mean of an array along the first dimension."""
        return data - data.mean(axis=0)


    def __normalizeData(self, data):
        """Normalize the array along the first dimension (divide by std)."""
        return data / data.std(axis=0)


    def __hasNoNanTimeSteps(self, data):
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


    def __isNotEmpty(self, index):
        if (index.size > 0):
            return True
        else:
            raise ValueError('Input field is empty or contains NaN only.')


    def thetaForecast(self, series, steps=None, seasonalPeriod=365):
        if steps is None:
            steps = len(series)

        model = ThetaModel(series, period=seasonalPeriod, deseasonalize=True, use_test=False).fit()
        forecast = model.forecast(steps=steps, theta=20)

        return forecast

    def extendData(self, data, seasonalPeriod=365):

        extendedData = [self.thetaForecast(col, seasonalPeriod=seasonalPeriod) for col in tqdm(data.T)]
        extendedData = np.array(extendedData).T

        return extendedData

    def complexifyData(self, data, extendSeries=False, seasonalPeriod=365):
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
            forecast    = self.extendData(data, seasonalPeriod=seasonalPeriod)
            backcast    = self.extendData(data[::-1], seasonalPeriod=seasonalPeriod)[::-1]

            data = np.concatenate([backcast, data, forecast])

        # perform actual Hilbert transform of (extended) time series
        data = hilbert(data,axis=0)

        if extendSeries:
            # cut out the first and last third of Hilbert transform
            # which belong to the forecast/backcast
            data    = data[self.__observations:(2*self.__observations)]
            data = self.__centerData(data)

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

        # complexify input data via Hilbert transfrom
        if (useHilbert):
            self.__noNanDataLeft = self.complexifyData(self.__noNanDataLeft, extendSeries=extendSeries, seasonalPeriod=seasonalPeriod)
            # save computing time if left and right field are the same
            if self.__useMCA:
                self.__noNanDataRight = self.complexifyData(self.__noNanDataRight, extendSeries=extendSeries, seasonalPeriod=seasonalPeriod)
            else:
                self.__noNanDataRight = self.__noNanDataLeft

        # create covariance matrix
        kernel = self.__noNanDataLeft.conjugate().T @ self.__noNanDataRight / self.__observations

        # solve eigenvalue problem
        VLeft, eigenvalues, VTRight = np.linalg.svd(kernel, full_matrices=False)
        VRight = VTRight.conjugate().T

        S = np.sqrt(np.diag(eigenvalues) * self.__observations)
        Si = np.diag(1./np.diag(S))

        self.__eigenvalues = eigenvalues
        self.__eigensum = eigenvalues.sum()

        # standardized EOF fields
        self.__VLeft 	= VLeft
        self.__VRight 	= VRight

        # loadings // EOF fields
        self.__LLeft 	= VLeft @ S
        self.__LRight 	= VRight @ S

        # get PC scores by projecting data fields on loadings
        self.__ULeft 	= self.__noNanDataLeft @ VLeft @ Si
        self.__URight 	= self.__noNanDataRight @ VRight @ Si


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
        L = np.concatenate((self.__LLeft[:,:nRotations], self.__LRight[:,:nRotations]))
        Lr, R, Phi = promax(L, power, maxIter=1000, tol=tol)
        LLeft 	= Lr[:self.__noNanIndexLeft.size,:]
        LRight 	= Lr[self.__noNanIndexLeft.size:,:]

        # calculate variance/reconstruct "eigenvalues"
        wLeft = np.linalg.norm(LLeft,axis=0)
        wRight = np.linalg.norm(LRight,axis=0)
        variance = wLeft * wRight / self.__observations
        varIdx = np.argsort(variance)[::-1]

        # pull loadings from EOFs
        VLeft 	= LLeft / wLeft
        VRight 	= LRight / wRight

        # rotate PC scores
        # If rotation is orthogonal: R.T = R
        # If rotation is oblique (p>1): R^(-1).T = R
        if(power==1):
            ULeft 	= self.__ULeft[:,:nRotations] @ R
            URight 	= self.__URight[:,:nRotations] @ R
        else:
            ULeft 	= self.__ULeft[:,:nRotations] @ np.linalg.pinv(R).conjugate().T
            URight 	= self.__URight[:,:nRotations] @ np.linalg.pinv(R).conjugate().T


        # store rotated pcs, eofs and "eigenvalues"
        # and sort according to described variance
        self.__eigenvalues 	= variance[varIdx]
        # Standardized EOFs
        self.__VLeft 		= VLeft[:,varIdx]
        self.__VRight 		= VRight[:,varIdx]
        # EOF loadings
        self.__LLeft 		= LLeft[:,varIdx]
        self.__LRight 		= LRight[:,varIdx]
        # Standardized PC scores
        self.__ULeft 		= ULeft[:,varIdx]
        self.__URight 		= URight[:,varIdx]

        # store rotation matrix and correlation matrix of PCs
        self.__rotationMatrix 		= R
        self.__correlationMatrix 	= Phi[varIdx,varIdx]
        self.__rotatedSolution 		= True


    def rotationMatrix(self):
        """
        Return the rotation matrix.

        Returns
        -------
        ndarray
            Rotation matrix.
        """
        if (self.__rotatedSolution):
            return self.__rotationMatrix
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
        if (self.__rotatedSolution):
            return self.__correlationMatrix
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
        values = self.__eigenvalues[:n]
        # error according to North's Rule of Thumb
        error = np.sqrt(2/self.__observations) * values

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
        desVar 		= values / self.__eigensum * 100
        desVarErr 	= error / self.__eigensum * 100
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
        pcsLeft 	= self.__ULeft[:,:n]
        pcsRight 	= self.__URight[:,:n]

        if (scaling==1):
            pcsLeft 	= pcsLeft * np.sqrt(self.__eigenvalues[:n])
            pcsRight 	= pcsRight * np.sqrt(self.__eigenvalues[:n])

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
            n = self.__eigenvalues.size

        # create data fields with original NaNs
        dtype = self.__VLeft.dtype
        eofsLeft  	= np.zeros([self.__variablesLeft, n],dtype=dtype) * np.nan
        eofsRight  	= np.zeros([self.__variablesRight, n],dtype=dtype) * np.nan

        eofsLeft[self.__noNanIndexLeft,:] = self.__VLeft[:,:n]
        eofsRight[self.__noNanIndexRight,:] = self.__VRight[:,:n]

        # reshape data fields to have original input shape
        eofsLeft 	= eofsLeft.reshape(self.__originalShapeLeft + (n,))
        eofsRight 	= eofsRight.reshape(self.__originalShapeRight + (n,))

        if (scaling==1):
            eofsLeft 	= eofsLeft * np.sqrt(self.__eigenvalues[:n])
            eofsRight 	= eofsRight * np.sqrt(self.__eigenvalues[:n])

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
