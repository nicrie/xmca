#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex rotated maximum covariance analysis of two numpy arrays.
"""
# =============================================================================
# Imports
# =============================================================================
import os

import numpy as np
import xarray as xr
from scipy.signal import hilbert
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm
from datetime import datetime

from tools.rotation import promax
from tools.array import is_arr, arrs_are_equal, remove_nan_cols, remove_mean
from tools.array import is_not_empty, check_time_dims, check_nan_rows
from tools.text import secure_str, boldify_str, wrap_str
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

    def __init__(self, left = None, right = None):
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

        """
        self._left      = np.array([]) if left is None else left.copy()
        self._right     = self._left if right is None else right.copy()

        is_arr(self._left)
        is_arr(self._right)
        check_time_dims(self._left, self._right)

        # center input data to zero mean (remove mean)
        self._left 	    = remove_mean(self._left)
        self._right 	= remove_mean(self._right)

        # store meta information
        self._analysis = {
            # data input
            'left_name'             : 'left',
            'right_name'            : 'right',
            'is_bivariate'          : False,
            # pre-processing
            'is_normalized'         : False,
            'is_coslat_corrected'   : False,
            'method'                : 'pca',
            # Complex solution
            'is_complex'            : False,
            'theta'                 : False,
            'theta_period'          : 365,
            #Rotated solution
            'is_rotated'            : False,
            'rotations'             : 0,
            'power'                 : 0,
            # Truncated solution
            'is_truncated'          : False,
            'is_truncated_at'       : 0,
            'eigen_dimension'    : 0,
            'eigensum'          : 0.0
            }

        self._analysis['is_bivariate']  = not arrs_are_equal(self._left, self._right)
        self._analysis['method']        = self._get_method_id()




    def _get_method_id(self):
        id = 'pca'
        if self._analysis['is_bivariate']:
            id = 'mca'
            if self._analysis['is_normalized']:
                id = 'cca'
        return id


    def _get_complex_id(self):
        id = int(self._analysis['is_complex'])
        return 'c{:}'.format(id)


    def _get_rotation_id(self):
        id = self._analysis['rotations']
        return 'r{:02}'.format(id)


    def _get_power_id(self):
        id = self._analysis['power']
        return 'p{:02}'.format(id)


    def _get_analysis_id(self):
        method      = self._get_method_id()
        hilbert     = self._get_complex_id()
        rotation    = self._get_rotation_id()
        power       = self._get_power_id()

        analysis    = '_'.join([method,hilbert,rotation,power])
        return analysis


    def _get_analysis_path(self, path=None):
        base_path   = path
        if base_path is None:
            base_path = os.getcwd()

        base_folder = 'pycca'

        left_var    = self._analysis['left_name']
        right_var   = self._analysis['right_name']

        analysis_folder     = left_var
        if self._analysis['is_bivariate']:
            analysis_folder = '_'.join([analysis_folder, right_var])
        analysis_folder = secure_str(analysis_folder)

        analysis_path   = os.path.join(base_path, base_folder, analysis_folder)

        if not os.path.exists(analysis_path):
            os.makedirs(analysis_path)

        return analysis_path


    def _theta_forecast(self, series, steps=None, period=365):
        if steps is None:
            steps = len(series)

        model = ThetaModel(series, period=period, deseasonalize=True, use_test=False).fit()
        forecast = model.forecast(steps=steps, theta=20)

        return forecast


    def _extend_data(self, data, period=365):

        extendedData = [self._theta_forecast(col, period=period) for col in tqdm(data.T)]
        extendedData = np.array(extendedData).T

        return extendedData


    def _complexify_data(self, data, theta=False, period=365):
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
        theta : boolean, optional
            If True, input time series are extended via forecast/backcast to
            3 * original length. This helps avoiding boundary effects of FFT.
        period : int, optional
            Period used to extend time series via Theta model. Using daily
            data a period of 365 represents seasonal cycle. The default is 365.

        Returns
        -------
        ndarray
            Analytical signal of input data.

        """

        if theta:
            forecast    = self._extend_data(data, period=period)
            backcast    = self._extend_data(data[::-1], period=period)[::-1]

            data = np.concatenate([backcast, data, forecast])
            self._analysis['theta_period'] = period

        # perform actual Hilbert transform of (extended) time series
        data = hilbert(data,axis=0)

        if theta:
            # cut out the first and last third of Hilbert transform
            # which belong to the forecast/backcast
            data = data[self._observations:(2*self._observations)]
            data = remove_mean(data)

        return data


    def apply_weights(self,left=None, right=None):
        """Apply weights to data sets.

        Supplied weights are applied via broadcasting.

        Parameters
        ----------
        left : ndarray
            Weights for left data set.
        right : ndarray
            Weights for right data set.


        """

        if left is None:
            left = 1

        if right is None:
            right = 1

        self._left  = self._left * left
        self._right = self._right * right


    def normalize(self):
        """Normalize the input data to unit variance."""

        if (self._analysis['is_normalized']):
            print('Data already normalized. Nothing was done.')
            return None

        else:
            self._left  = self._left / self._left.std(axis=0)
            self._right = self._right / self._right.std(axis=0)

            self._analysis['is_normalized'] = True
            self._analysis['is_coslat_corrected'] = False
            self._analysis['method'] = self._get_method_id()
            return None


    def solve(self, complexify=False, theta=False, period=365):
        """Solve eigenvalue equation by performing SVD on covariance matrix.

        Parameters
        ----------
        complexify : boolean, optional
            Use Hilbert transform to complexify the input data fields
            in order to perform complex PCA/MCA. Default is false.
        theta : boolean, optional
            If True, extend time series by fore/backcasting based on
            Theta model. New time series will have 3 * original length.
            Only used for complex time series (complexify=True).
            Default is False.
        period : int, optional
            Seasonal period used for Theta model. Default is 365, representing
            a yearly cycle for daily data.
        """

        self._observations                  = self._left.shape[0]
        self._left_original_spatial_shape   = self._left.shape[1:]
        self._right_original_spatial_shape 	= self._right.shape[1:]

        self._left_variables 		= np.product(self._left_original_spatial_shape)
        self._right_variables 		= np.product(self._right_original_spatial_shape)

        # create 2D matrix in order to perform PCA
        self._left 	    = self._left.reshape(self._observations,self._left_variables)
        self._right 	= self._right.reshape(self._observations,self._right_variables)

        # check for NaN time steps
        check_nan_rows(self._left)
        check_nan_rows(self._right)

        # remove NaNs columns in data fields
        self._left, self._left_no_nan_index   = remove_nan_cols(self._left)
        self._right, self._right_no_nan_index = remove_nan_cols(self._right)

        is_not_empty(self._left)
        is_not_empty(self._right)

        # complexify input data via Hilbert transform
        if (complexify == True):
            self._left = self._complexify_data(self._left, theta=theta, period=period)
            # save computing time if left and right field are the same
            if self._analysis['is_bivariate']:
                self._right = self._complexify_data(self._right, theta=theta, period=period)
            else:
                self._right = self._left

            self._analysis['is_complex'] = True

        # create covariance matrix
        kernel = self._left.conjugate().T @ self._right / self._observations

        # solve eigenvalue problem
        try:
            VLeft, eigenvalues, VTRight = np.linalg.svd(kernel, full_matrices=False)
        except LinAlgError:
            raise LinAlgError("SVD failed. NaN entries may be the problem.")

        VRight = VTRight.conjugate().T

        S = np.sqrt(np.diag(eigenvalues) * self._observations)
        Si = np.diag(1./np.diag(S))

        self._eigenvalues = eigenvalues
        self._analysis['eigensum'] = eigenvalues.sum()
        self._analysis['eigen_dimension'] = eigenvalues.size
        self._analysis['is_truncated_at'] = eigenvalues.size

        # standardized EOF fields
        self._VLeft 	= VLeft
        self._VRight 	= VRight

        # loadings // EOF fields
        self._LLeft 	= VLeft @ S
        self._LRight 	= VRight @ S

        # get PC scores by projecting data fields on loadings
        self._ULeft 	= self._left @ VLeft @ Si
        self._URight 	= self._right @ VRight @ Si


    def rotate(self, n_rot, power=1, tol=1e-5):
        """Perform Promax rotation on the first `n` EOFs.

        Promax rotation (Hendrickson & White 1964) is an oblique rotation which
        seeks to find `simple structures` in the EOFs. It transforms the EOFs
        via an orthogonal Varimax rotation (Kaiser 1958) followed by the Promax
        equation. If `power=1`, Promax reduces to Varimax rotation. In general,
        a Promax transformation breaks the orthogonality of EOFs and introduces
        some correlation between PCs.

        Parameters
        ----------
        n_rot : int
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
        if(n_rot < 2):
            raise ValueError('`n_rot` must be >=2')
        if(power<1):
            raise ValueError('`power` must be >=1')

        # rotate loadings (Cheng and Dunkerton 1995)
        L = np.concatenate((self._LLeft[:,:n_rot], self._LRight[:,:n_rot]))
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
            ULeft 	= self._ULeft[:,:n_rot] @ R
            URight 	= self._URight[:,:n_rot] @ R
        else:
            ULeft 	= self._ULeft[:,:n_rot] @ np.linalg.pinv(R).conjugate().T
            URight 	= self._URight[:,:n_rot] @ np.linalg.pinv(R).conjugate().T


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
        self._rotation_matrix 		= R
        self._correlation_matrix 	= Phi[varIdx,varIdx]
        self._analysis['is_rotated']            = True
        self._analysis['rotations']             = n_rot
        self._analysis['power']                 = power


    def rotation_matrix(self):
        """
        Return the rotation matrix.

        Returns
        -------
        ndarray
            Rotation matrix.
        """
        if (self._analysis['is_rotated']):
            return self._rotation_matrix
        else:
            print('Apply `.rotate()` first to retrieve the correlation matrix.')


    def correlation_matrix(self):
        """
        Return the correlation matrix of rotated PCs.

        Returns
        -------
        ndarray
            Correlation matrix.

        """
        if (self._analysis['is_rotated']):
            return self._correlation_matrix
        else:
            print('Apply `.rotate()` first to retrieve the correlation matrix.')


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


    def explained_variance(self, n=None):
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
        desVar 		= values / self._analysis['eigensum'] * 100
        desVarErr 	= error / self._analysis['eigensum'] * 100
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
        eofsLeft  	= np.zeros([self._left_variables, n],dtype=dtype) * np.nan
        eofsRight  	= np.zeros([self._right_variables, n],dtype=dtype) * np.nan

        eofsLeft[self._left_no_nan_index,:] = self._VLeft[:,:n]
        eofsRight[self._right_no_nan_index,:] = self._VRight[:,:n]

        # reshape data fields to have original input shape
        eofsLeft 	= eofsLeft.reshape(self._left_original_spatial_shape + (n,))
        eofsRight 	= eofsRight.reshape(self._right_original_spatial_shape + (n,))

        if (scaling==1):
            eofsLeft 	= eofsLeft * np.sqrt(self._eigenvalues[:n])
            eofsRight 	= eofsRight * np.sqrt(self._eigenvalues[:n])

        return eofsLeft, eofsRight


    def spatial_amplitude(self, n=None):
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


    def spatial_phase(self, n=None):
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


    def temporal_amplitude(self, n=None):
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


    def temporal_phase(self, n=None):
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


    def truncate(self, n):
        """Truncate solution.

        Parameters
        ----------
        n : int
            Cut off after mode `n`.


        """
        if (n < self._eigenvalues.size):
            self._eigenvalues = self._eigenvalues[:n]

            self._ULeft = self._ULeft[:,:n]
            self._VLeft = self._VLeft[:,:n]

            self._VRight = self._VRight[:,:n]
            self._URight = self._URight[:,:n]

            self._analysis['is_truncated'] = True
            self._analysis['is_truncated_at'] = n


    def _create_info_file(self, path):
        sep_line = '\n#' + '-' * 79
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_header = (
            'This file contains information neccessary to load stored analysis '
            'data from pyCCA module.')

        path_output   = os.path.join(path, self._get_analysis_id())

        file = open(path_output,"w+")
        file.write(wrap_str(file_header))
        file.write('\n# To load this analysis use:')
        file.write('\n# from mca.xarray import xMCA')
        file.write('\n# cca = xMCA()')
        file.write('\n# cca.load_analysis(PATH_TO_THIS_FILE)')
        file.write('\n')
        file.write(sep_line)
        file.write(sep_line)
        file.write('\n{:<20} : {:<57}'.format('created',now))
        file.close()


    def _save_info_to_file(self, path):
        sep_line = '\n#' + '-' * 79

        path_output   = os.path.join(path,self._get_analysis_id())

        file = open(path_output,"a")
        file.write(sep_line)
        for key, value in self._analysis.items():
            if key in ['is_bivariate','is_complex', 'is_rotated','is_truncated']:
                file.write(sep_line)
            file.write('\n{:<20} : {:<57}'.format(key, str(value)))
        file.close()


    def _save_paths_to_file(self, path, format):
        sep_line = '\n#' + '-' * 79

        var1 = secure_str(self._analysis['left_name'])
        var2 = secure_str(self._analysis['right_name'])
        eofs = '.'.join(['eofs',format])
        pcs = '.'.join(['pcs',format])
        eigenvalues = '.'.join(['eigenvalues',format])

        path_output   = os.path.join(path,self._get_analysis_id())
        path_eof1   = '_'.join([path_output, var1, eofs])
        path_eof2   = '_'.join([path_output, var2, eofs])
        path_pc1    = '_'.join([path_output, var1, pcs])
        path_pc2    = '_'.join([path_output, var2, pcs])
        path_eigen  = '_'.join([path_output, eigenvalues])

        file = open(path_output,"a")
        file.write(sep_line)
        file.write('\n{:<20} : {:<57}'.format('this file', path_output))
        file.write('\n{:<20} : {:<57}'.format('eigenvalues', path_eigen))
        file.write('\n{:<20} : {:<57}'.format('field1_eofs', path_eof1))
        file.write('\n{:<20} : {:<57}'.format('field1_pcs', path_pc1))
        file.write('\n{:<20} : {:<57}'.format('field2_eofs', path_eof2))
        file.write('\n{:<20} : {:<57}'.format('field2_pcs', path_pc2))
        file.write('\n')
        file.close()


    def _save_data(self, data_array, path, *args, **kwargs):
        raise NotImplementedError('only works for `xarray`')


    def _set_key(self, key, value):
        try:
            key_type = type(self._analysis[key])
        except KeyError:
            raise KeyError("Key `{}` not found in info file.".format(key))
        if key_type == bool:
            self._analysis[key] = (value == 'True')
        else:
            self._analysis[key] = key_type(value)


    def _set_info_from_file(self, path):

        info_file = open(path, 'r')
        lines = info_file.readlines()
        for line in lines:
            if (line[0] != '#'):
                key = line.split(':')[0]
                key = key.rstrip()
                if key in self._analysis.keys():
                    value = line.split(':')[1].strip()
                    self._set_key(key, value)
        info_file.close()


    def _get_locs_from_file(self, path):
        # locations for data files
        file_locations = {
            'field1_eofs'       : None,
            'field1_pcs'        : None,
            'field2_eofs'       : None,
            'field2_pcs'        : None,
            'eigenvalues'       : None
        }

        info_file = open(path, 'r')
        lines = info_file.readlines()
        for line in lines:
            if (line[0] != '#'):
                key = line.split(':')[0]
                key = key.rstrip()
                if key in file_locations.keys():
                    path = line.split(':')[1].strip()
                    file_locations[key] = path
        info_file.close()

        return file_locations



    def load_analysis(self, info_file, eofs=None, pcs=None, eigenvalues=None):
        self._set_info_from_file(info_file)

        # standardized fields // EOF fields + PCs
        if self._analysis['is_bivariate']:
            eofsLeft, eofsRight = [eofs[0], eofs[1]]
            pcsLeft, pcsRight   = [pcs[0], pcs[1]]
        else:
            eofsLeft, eofsRight = [eofs[0], eofs[0]]
            pcsLeft, pcsRight   = [pcs[0], pcs[0]]

        self._observations                      = pcsLeft.shape[0]
        self._left_original_spatial_shape       = eofsLeft.shape[:-1]
        self._right_original_spatial_shape 	    = eofsRight.shape[:-1]
        number_modes                            = eofsLeft.shape[-1]


        self._left_variables 		= np.product(self._left_original_spatial_shape)
        self._right_variables 		= np.product(self._right_original_spatial_shape)

        eofsLeft    = eofsLeft.reshape(self._left_variables, number_modes)
        eofsRight   = eofsRight.reshape(self._right_variables, number_modes)

        VLeftT, self._left_no_nan_index   = remove_nan_cols(eofsLeft.T)
        VRightT, self._right_no_nan_index = remove_nan_cols(eofsRight.T)
        self._VLeft     = VLeftT.T
        self._VRight    = VRightT.T

        S   = np.sqrt(np.diag(eigenvalues) * self._observations)
        Si  = np.diag(1./np.diag(S))

        self._eigenvalues   = eigenvalues

        self._ULeft, self._URight = [pcsLeft, pcsRight]

        # loadings // EOF fields
        self._LLeft 	= self._VLeft @ S
        self._LRight 	= self._VRight @ S
