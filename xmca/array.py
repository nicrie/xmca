#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex rotated maximum covariance analysis of two numpy arrays.
"""
# =============================================================================
# Imports
# =============================================================================
import cmath
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from statsmodels.tsa.forecasting.theta import ThetaModel
from tools.array import (arrs_are_equal, check_nan_rows, check_time_dims,
                         is_arr, is_not_empty, norm_to_1, remove_mean,
                         remove_nan_cols)
from tools.rotation import promax
from tools.text import boldify_str, secure_str, wrap_str
from tqdm import tqdm


# =============================================================================
# MCA
# =============================================================================
class MCA(object):
    """Perform Maximum Covariance Analysis (MCA) for two `numpy.ndarray`.

    MCA is a more general form of Principal Component Analysis (PCA)
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

    >>> pca = CCA(data1)
    >>> pca.solve()
    >>> pcs,_ = pca.pcs()

    To perform MCA use:

    >>> mca = CCA(data1, data2)
    >>> mca.solve()
    >>> pcs_data1, pcs_data2 = mca.pcs()

    """

    def __init__(self, *data):
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
        if len(data) > 2:
            raise ValueError("Too many fields. Pass 1 or 2 fields.")

        self._fields                = {} # input fields
        self._field_names           = {} # names of input fields
        self._field_means           = {} # mean of fields
        self._field_stds            = {} # standard deviation of fields
        self._fields_spatial_shape  = {} # spatial shapes of fields
        self._n_variables           = {} # number of variables of fields
        self._no_nan_index          = {} # index of variables containing data
        self._n_observations        = {} # number of observations/samples

        # set fields
        self._keys    = ['left', 'right']
        self._fields  = {key : field for key,field in zip(self._keys,data)}
        # set fields
        #self._fields = {'left'  : np.array([]) if left is None else left.copy()}
        #if right is not None:
        #    self._fields['right'] = right.copy()

        # set class variables
        for key, field in self._fields.items():
            is_arr(field)
            self._field_names[key]          = key
            self._field_means[key]          = field.mean(axis=0)
            self._field_stds[key]           = field.std(axis=0)
            self._fields_spatial_shape[key] = field.shape[1:]
            self._n_variables[key]          = np.product(field.shape[1:])
            self._n_observations[key]       = field.shape[0]

            # center input data to zero mean (remove mean)
            self._fields[key]       = remove_mean(field)

        # set meta information
        self._analysis = {
            'is_bivariate'          : True if len(self._fields) > 1 else False,
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
            'rank'                  : 0,
            'total_covariance'      : 0.0,
            'total_squared_covariance'      : 0.0
            }

        self._analysis['method']        = self._get_method_id()


    def set_field_names(self, left = None, right = None):
        if left is not None:
            self._field_names['left']   = left
        if right is not None:
            self._field_names['right']   = right


    def _get_method_id(self):
        id = 'pca'
        if self._analysis['is_bivariate']:
            id = 'mca'
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

        base_folder = 'xmca'

        analysis_folder = '_'.join(self._field_names.values())
        analysis_folder = secure_str(analysis_folder)

        analysis_path   = os.path.join(base_path, base_folder, analysis_folder)

        if not os.path.exists(analysis_path):
            os.makedirs(analysis_path)

        return analysis_path


    def _get_fields(self, original_scale=False):
        std     = self._field_stds
        mean    = self._field_means
        fields  = self._fields

        new_fields = {}
        for key, field in fields.items():
            new_fields[key] = field
            if original_scale:
                if self._analysis['is_normalized']:
                    new_fields[key] *= std[key]
                new_fields[key] += mean[key]

        return new_fields


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
        field_items = self._fields.items()

        weights = {'left' : left, 'right' : right}
        weights.update({k : 1 if w is None else w for k,w in weights.items()})
        self._fields.update({k : field * weight for k,field in field_items})


    def normalize(self):
        """Normalize the input data to unit variance."""


        keys        = self._fields.keys()
        fields      = self._fields.values()
        fields_std  = self._field_stds.values()

        for key,field,std in zip(keys,fields,fields_std):
            self._fields[key] = field / std

        self._analysis['is_normalized'] = True
        self._analysis['is_coslat_corrected'] = False
        self._analysis['method'] = self._get_method_id()
        return None


    def _theta_forecast(self, series):
        period = self._analysis['theta_period']
        steps = len(series)

        # replace last value of series by a mean value
        # to avoid some extreme cases where the foecast starts at a single
        # which may happen for very noisy data
        # series[0]   = series[::period].mean()
        # series[-1]  = series[::-period].mean()

        model = ThetaModel(series, period=period, deseasonalize=True, use_test=False).fit()
        forecast = model.forecast(steps=steps, theta=20)

        return forecast


    def _extend_data(self, data):
        extendedData = [self._theta_forecast(col) for col in tqdm(data.T)]
        extendedData = np.array(extendedData).T

        return extendedData


    def _complexify_data(self, data):
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
        n_observations = self._n_observations['left']

        if self._analysis['theta']:
            forecast    = self._extend_data(data)
            backcast    = self._extend_data(data[::-1])[::-1]

            data = np.concatenate([backcast, data, forecast])

        # perform actual Hilbert transform of (extended) time series
        data = hilbert(data,axis=0)

        if self._analysis['theta']:
            # cut out the first and last third of Hilbert transform
            # which belong to the forecast/backcast
            data = data[n_observations:(2*n_observations)]
            data = remove_mean(data)

        return data


    def solve(self, complexify=False, theta=False, period=365, save_memory=False):
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
        self._analysis['is_complex']    = complexify
        self._analysis['theta']         = theta
        self._analysis['theta_period']  = period

        n_observations  = self._n_observations
        n_variables     = self._n_variables

        field_2d = {}

        for key,field in self._fields.items():
            # create 2D matrix in order to perform PCA
            field = field.reshape(n_observations[key], n_variables[key])
            # check for NaN time steps
            check_nan_rows(field)
            # remove NaNs columns in data fields
            field, no_nan_index   = remove_nan_cols(field)
            is_not_empty(field)

            # complexify input data via Hilbert transform
            if self._analysis['is_complex']:
                field = self._complexify_data(field)

            field_2d[key]           = field
            # save index of real variables for recontruction of original field
            self._no_nan_index[key] = no_nan_index


        # create covariance matrix
        if self._analysis['is_bivariate']:
            kernel = field_2d['left'].conjugate().T @ field_2d['right']
        else:
            kernel = field_2d['left'].conjugate().T @ field_2d['left']
        kernel = kernel / n_observations['left']

        self._V = {} # singular vectors (EOFs)
        self._L = {} # "loaded" singular vectors
        self._U = {} # projections // (PCs)

        # perform singular value decomposition
        try:
            if save_memory:
                import scipy as sp
                VLeft, singular_values, VTRight = sp.sparse.linalg.svd(kernel, full_matrices=False)
            VLeft, singular_values, VTRight = np.linalg.svd(kernel, full_matrices=False)
        except LinAlgError:
            raise LinAlgError("SVD failed. NaN entries may be the problem.")

        self._V['left'] = VLeft
        if self._analysis['is_bivariate']:
            self._V['right'] = VTRight.conjugate().T
        # free up some space
        del(VLeft)
        del(VTRight)

        S = np.sqrt(np.diag(singular_values) * n_observations['left'])
        Si = np.diag(1./np.diag(S))

        self._singular_values = singular_values

        for key, V in self._V.items():
            # "loaded" EOFs
            self._L[key] = V @ S
            # get PC scores by projecting fields on loaded EOFs
            self._U[key] = field_2d[key] @ V @ Si

        self._analysis['total_covariance'] = singular_values.sum()
        self._analysis['total_squared_covariance'] = (singular_values**2).sum()
        self._analysis['rank'] = singular_values.size
        self._analysis['is_truncated_at'] = singular_values.size


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
            print('`n_rot` must be >=2. Solution not rotated.')
            return None
        if(power<1):
            raise ValueError('`power` must be >=1')

        n_observations = self._n_observations['left']

        truncated_L = {}
        for key, L in self._L.items():
            truncated_L[key] = L[:,:n_rot]

        # rotate loadings (Cheng and Dunkerton 1995)
        combined_L = np.concatenate(list(truncated_L.values()))
        combined_L, R, Phi = promax(combined_L, power, maxIter=1000, tol=tol)

        rot_V   = {} # rotated singular vectors (EOFs)
        rot_L   = {} # rotated "loading"
        rot_U   = {} # rotated projections (PCs)
        weights = {} # weights associated to "loadings"

        rot_L['left']   = combined_L[:truncated_L['left'].shape[0],:]
        if self._analysis['is_bivariate']:
            rot_L['right']  = combined_L[truncated_L['left'].shape[0]:,:]

        for key, L in rot_L.items():
            # calculate variance/reconstruct "singular_values"
            w = np.linalg.norm(L, axis=0)
            # pull loadings from EOFs
            rot_V[key] 	=  L / w
            weights[key] = w

        if self._analysis['is_bivariate']:
            variance = weights['left'] * weights['right'] / n_observations
        else:
            variance = weights['left'] * weights['left'] / n_observations
        var_idx = np.argsort(variance)[::-1]

        # rotate PC scores
        # If rotation is orthogonal: R.T = R
        # If rotation is oblique (p>1): R^(-1).T = R
        for key, pcs in self._U.items():
            if(power==1):
                rot_U[key] = pcs[:,:n_rot] @ R
            else:
                rot_U[key] = pcs[:,:n_rot] @ np.linalg.pinv(R).conjugate().T

        # store rotated pcs, eofs and "singular_values"
        # and sort according to described variance
        self._singular_values 	= variance[var_idx]
        for key in rot_L.keys():
            self._V[key] = rot_V[key][:,var_idx] # Standardized EOFs
            self._L[key] = rot_L[key][:,var_idx] # Loaded EOFs
            self._U[key] = rot_U[key][:,var_idx] # Standardized PC scores

        # store rotation and correlation matrix of PCs + meta information
        self._rotation_matrix           = R
        self._correlation_matrix        = Phi[var_idx,:][:,var_idx]
        self._analysis['is_rotated']    = True
        self._analysis['rotations']     = n_rot
        self._analysis['power']         = power


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


    def singular_values(self,n=None):
        """Return the first `n` singular_values.

        Parameters
        ----------
        n : int, optional
            Number of singular_values to return. The default is 5.

        Returns
        -------
        values : ndarray
            singular_values of PCA.
        error : ndarray
            Uncertainty of singular_values according to North's rule of thumb.

        """
        values = self._singular_values[:n]
        n_observations = self._n_observations['left']
        # error according to North's Rule of Thumb
        error = np.sqrt(2 / n_observations) * values

        return values

    def scf(self, n=None):
        """Return the SCF of the first `n` modes.

        The squared ovariance/correlation fraction (SCF) is a measure of
        importance of each mode. It is calculated as the
        squared singular values divided by the sum of squared singluar values.
        In contrast to CF, it is an invariant for CCA.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is None.

        Returns
        -------
        ndarray
            Fraction of described squared covariance/correlation of each mode.

        """
        values  = self.singular_values(n)
        scf = values**2 / self._analysis['total_squared_covariance'] * 100
        return scf


    def explained_variance(self, n=None):
        """Return the CF of the first `n` modes.

        The covariance/correlation fraction (CF) is a measure of
        importance of each mode. It is calculated as the
        singular values divided by the sum of singluar values.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is None.

        Returns
        -------
        ndarray
            Fraction of described covariance/correlation of each mode.

        """
        values  = self.singular_values(n)
        exp_var = values / self._analysis['total_covariance'] * 100
        return exp_var


    def pcs(self, n=None, scaling=None, phase_shift=0):
        """Return the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of PCs to be returned. The default is None.
        scaling : {None, 'eigen', 'max', 'std'}, optional
            Scale by singular_values ('eigen'), maximum value ('max') or
            standard deviation ('std'). The default is None.

        Returns
        -------
        pcs : dict[ndarray, ndarray]
            PCs associated to left and right input field.

        """
        n_obs       = self._n_observations['left']
        singular_values = self._singular_values

        if n is None:
            n = singular_values.size

        pcs = {}
        for key, U in self._U.items():
            pcs[key] = U[:,:n].copy()
            # apply phase shift
            if self._analysis['is_complex']:
                pcs[key] *= cmath.rect(1,phase_shift)
            # scale PCs by singular_values
            if scaling == 'eigen':
                pcs[key] *= np.sqrt(n_obs * singular_values[:n])
            # scale PCs by maximum value
            if scaling == 'max':
                pcs[key] /= np.nanmax(abs(pcs[key].real), axis=0)
            # scale PCs by standard deviation
            if scaling == 'std':
                pcs[key] /= np.nanstd(abs(pcs[key].real), axis=0)

        return pcs


    def eofs(self, n=None, scaling=None, phase_shift=0):
        """Return the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to be returned. The default is None.
        scaling : {None, 'eigen', 'max', 'std'}, optional
            Scale by singular_values ('eigen'), maximum value ('max') or
            standard deviation ('std'). The default is None.

        Returns
        -------
        eofs : dict[ndarray, ndarray]
            EOFs associated to left and right input field.

        """
        n_obs       = self._n_observations['left']
        n_var       = self._n_variables
        no_nan_idx  = self._no_nan_index
        field_shape = self._fields_spatial_shape
        singular_values = self._singular_values

        if n is None:
            n = self._singular_values.size

        eofs = {}
        for key, V in self._V.items():
            # create data fields with original NaNs
            dtype       = V.dtype
            eofs[key]   = np.zeros([n_var[key], n], dtype = dtype) * np.nan
            eofs[key][no_nan_idx[key],:] = V[:,:n]
            # reshape data fields to have original input shape
            eofs[key] 	= eofs[key].reshape(field_shape[key] + (n,))
            # scale EOFs with their singular_values
            if scaling == 'eigen':
                eofs[key] *= np.sqrt(n_obs * singular_values[:n])
            # scale EOFs by maximum value
            if scaling == 'max':
                eofs[key] /= np.nanmax(abs(eofs[key].real), axis=(0,1))
            # scale EOFs by standard deviation
            if scaling == 'std':
                eofs[key] /= np.nanstd(abs(eofs[key].real), axis=(0,1))
            # apply phase shift
            if self._analysis['is_complex']:
                eofs[key] *=  cmath.rect(1,phase_shift)

        return eofs


    def spatial_amplitude(self, n=None, scaling=None):
        """Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to be returned. If None, return all
            fields. The default is None.
        scaling : {None, 'max', 'std'}, optional
            Scale by maximum value ('max'). The default is None.
        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Spatial amplitude fields associated to left and right field.

        """
        eofs = self.eofs(n, scaling=None)

        amplitudes = {}
        for key, eof in eofs.items():
            amplitudes[key] = np.sqrt(eof * eof.conjugate()).real

            if scaling == 'max':
                amplitudes[key] /= np.nanmax(amplitudes[key], axis=(0,1))

        return amplitudes


    def spatial_phase(self, n=None, phase_shift=0):
        """Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Spatial phase fields associated to left and right field.

        """
        eofs = self.eofs(n, phase_shift=phase_shift)

        phases = {}
        for key, eof in eofs.items():
            phases[key] = np.arctan2(eof.imag, eof.real).real

        return phases


    def temporal_amplitude(self, n=None, scaling=None):
        """Return the temporal amplitude time series for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude series to be returned. If None, return all series.
            The default is None.
        scaling : {None, 'max'}, optional
            Scale by maximum value ('max'). The default is None.

        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Temporal ampliude series associated to left and right field.

        """
        pcs = self.pcs(n, scaling=None)

        amplitudes = {}
        for key, pc in pcs.items():
            amplitudes[key] = np.sqrt(pc * pc.conjugate()).real

            if scaling == 'max':
                amplitudes[key] /= np.nanmax(amplitudes[key], axis=0)

        return amplitudes


    def temporal_phase(self, n=None, phase_shift=0):
        """Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, return all series.
            The default is None.

        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Temporal phase function associated to left and right field.

        """
        pcs = self.pcs(n, phase_shift=phase_shift)

        phases = {}
        for key, pc in pcs.items():
            phases[key] = np.arctan2(pc.imag, pc.real).real

        return phases


    def plot(
        self, mode, threshold=0, phase_shift=0,
        cmap_eof=None, cmap_phase=None, figsize=(8.3,5.0)):
        """
        Plot results for `mode`.

        Parameters
        ----------
        mode : int, optional
            Mode to plot. The default is 1.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        cmap_eof : str or Colormap
            The colormap used to map the spatial patterns.
            The default is 'Blues'.
        cmap_phase : str or Colormap
            The colormap used to map the spatial phase function.
            The default is 'twilight'.

        Returns
        -------
        None.

        """
        pcs     = self.pcs(mode, scaling='max', phase_shift=phase_shift)
        eofs    = self.eofs(mode, scaling='max')
        phases  = self.spatial_phase(mode, phase_shift=phase_shift)
        var 	= self.explained_variance(mode)[mode-1]

        n_cols          = 2
        n_rows          = len(pcs)
        height_ratios   = [1] * n_rows

        # add additional row for colorbar
        n_rows += 1
        height_ratios.append(0.05)

        eof_title       = 'EOF'
        cmap_eof_range  = [-1, 0, 1]

        if self._analysis['is_complex']:
            n_cols          += 1
            eofs            = self.spatial_amplitude(mode, scaling='max')
            eof_title       = 'Amplitude'
            cmap_eof_range  = [0, 1]
            cmap_eof        = 'Blues' if cmap_eof is None else cmap_eof
            cmap_phase      = 'twilight' if cmap_phase is None else cmap_phase
        else:
            cmap_eof        = 'RdBu_r' if cmap_eof is None else cmap_eof

        for key in pcs.keys():
            pcs[key]    = pcs[key][:,mode-1].real
            eofs[key]   = eofs[key][:,:,mode-1]
            phases[key]  = phases[key][:,:,mode-1]

            # apply amplitude threshold
            eofs[key]  = np.where(abs(eofs[key]) >= threshold, eofs[key], np.nan)
            phases[key] = np.where(abs(eofs[key]) >= threshold, phases[key], np.nan)


        titles = {
        'pc' : 'PC {:d} ({:.1f} \%)'.format(mode,var),
        'eof': eof_title,
        'phase':'Phase',
        'var1' : self._field_names['left'],
        'var2' : self._field_names['right']
        }

        titles.update({k: v.replace('_',' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})


        # create figure environment
        fig = plt.figure(figsize=figsize, dpi=150)
        fig.subplots_adjust(hspace=0.1, wspace=.1, left=0.25)
        gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
        axes_pc = [fig.add_subplot(gs[i,0]) for i in range(n_rows-1)]
        axes_eof = [fig.add_subplot(gs[i,1]) for i in range(n_rows-1)]
        cbax_eof = fig.add_subplot(gs[-1,1])

        axes_space = axes_eof

        var_names = [titles['var1'], titles['var2']]

        # plot PCs
        for i, pc in enumerate(pcs.values()):
            axes_pc[i].plot(pc)
            axes_pc[i].set_ylim(-1.2,1.2)
            axes_pc[i].set_xlabel('')
            axes_pc[i].set_ylabel(var_names[i], fontweight='bold')
            axes_pc[i].set_title('')
            axes_pc[i].set_yticks([-1,0,1])
            axes_pc[i].spines['right'].set_visible(False)
            axes_pc[i].spines['top'].set_visible(False)

        axes_pc[0].xaxis.set_visible(False)
        axes_pc[0].set_title(titles['pc'], fontweight='bold')

        # plot EOFs
        for i, eof in enumerate(eofs.values()):
            cb_eof = axes_eof[i].imshow(
                eof, origin='lower',
                vmin=cmap_eof_range[0], vmax=cmap_eof_range[-1], cmap=cmap_eof)
            axes_eof[i].set_title('')

        plt.colorbar(cb_eof, cbax_eof, orientation='horizontal')
        cbax_eof.xaxis.set_ticks(cmap_eof_range)
        axes_eof[0].set_title(titles['eof'], fontweight='bold')

        # plot Phase function (if data is complex)
        if (self._analysis['is_complex']):
            axes_phase = [fig.add_subplot(gs[i,2]) for i in range(n_rows-1)]
            cbax_phase = fig.add_subplot(gs[-1,2])

            for i, phase in enumerate(phases.values()):
                cb_phase = axes_phase[i].imshow(
                    phase, origin='lower',
                    vmin=-np.pi, vmax=np.pi, cmap=cmap_phase)
                axes_phase[i].set_title('')

            plt.colorbar(cb_phase, cbax_phase, orientation='horizontal')
            cbax_phase.xaxis.set_ticks([-3.14,0,3.14])
            cbax_phase.set_xticklabels([r'-$\pi$','0',r'$\pi$'])

            for a in axes_phase:
                axes_space.append(a)

            axes_phase[0].set_title(titles['phase'], fontweight='bold')

        # add map features
        for a in axes_space:
            a.set_aspect('auto')
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)

        # if more than 1 row, remove xaxis
        if (len(pcs) == 2):
            axes_pc[0].xaxis.set_visible(False)
            axes_pc[0].spines['bottom'].set_visible(False)


    def save_plot(self, mode, path=None, dpi=96, **kwargs):
        if path is None:
            path = self._get_analysis_path()

        mode_id = ''.join(['mode',str(mode)])
        format = '.png'
        file_name = '_'.join([self._get_analysis_id(),mode_id])
        file_path = os.path.join(path, file_name)
        fig, axes= self.plot(mode=mode, **kwargs)
        fig.subplots_adjust(left=0.06)
        plt.savefig(file_path + format, dpi=dpi)


    def truncate(self, n):
        """Truncate solution.

        Parameters
        ----------
        n : int
            Cut off after mode `n`.


        """
        if (n < self._singular_values.size):
            self._singular_values = self._singular_values[:n]

            for key in self._U.keys():
                self._U[key] = self._U[key][:,:n]
                self._V[key] = self._V[key][:,:n]

            self._analysis['is_truncated'] = True
            self._analysis['is_truncated_at'] = n


    def _create_info_file(self, path):
        sep_line = '\n#' + '-' * 79
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_header = (
            'This file contains information neccessary to load stored analysis '
            'data from xmca module.')

        path_output   = os.path.join(path, self._get_analysis_id())
        path_output = '.'.join([path_output,'info'])

        file = open(path_output,"w+")
        file.write(wrap_str(file_header))
        file.write('\n# To load this analysis use:')
        file.write('\n# from xmca.xarray import xMCA')
        file.write('\n# mca = xMCA()')
        file.write('\n# mca.load_analysis(PATH_TO_THIS_FILE)')
        file.write('\n')
        file.write(sep_line)
        file.write(sep_line)
        file.write('\n{:<20} : {:<57}'.format('created',now))
        file.write(sep_line)
        for key, name in self._field_names.items():
            file.write('\n{:<20} : {:<57}'.format(key, str(name)))
        file.write(sep_line)
        for key, info in self._analysis.items():
            if key in ['is_bivariate','is_complex', 'is_rotated','is_truncated']:
                file.write(sep_line)
            file.write('\n{:<20} : {:<57}'.format(key, str(info)))
        file.close()


    def _get_file_names(self, format):
        base_name = self._get_analysis_id()

        fields  = {}
        eofs    = {}
        pcs     = {}
        for key, variable in self._field_names.items():
            variable    = secure_str(variable)
            field_name  = '_'.join([base_name, variable])
            eof_name    = '_'.join([base_name, variable, 'eofs'])
            pc_name     = '_'.join([base_name, variable, 'pcs'])

            fields[key] = '.'.join([field_name, format])
            eofs[key]   = '.'.join([eof_name, format])
            pcs[key]    = '.'.join([pc_name, format])

        singular_values = '_'.join([base_name, 'singular_values'])
        singular_values = '.'.join([singular_values, format])

        file_names = {
            'fields'    : fields,
            'eofs'      : eofs,
            'pcs'       : pcs,
            'singular'  : singular_values
        }
        return file_names


    def _save_data(self, data_array, path, *args, **kwargs):
        raise NotImplementedError('only works for `xarray`')


    def _set_analysis(self, key, value):
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
                if key in ['left', 'right']:
                    name = line.split(':')[1].strip()
                    self._field_names[key] = name
                if key in self._analysis.keys():
                    value = line.split(':')[1].strip()
                    self._set_analysis(key, value)
        info_file.close()


    def load_analysis(self, path, fields, eofs, pcs, singular_values):
        self._set_info_from_file(path)

        self._V                     = {}
        self._L                     = {}
        self._U                     = {}
        self._singular_values           = singular_values
        for key in eofs.keys():
            self._n_observations[key]       = pcs[key].shape[0]
            self._fields_spatial_shape[key] = eofs[key].shape[:-1]
            self._n_variables[key]          = np.product(eofs[key].shape[:-1])
            n_modes                         = eofs[key].shape[-1]
            S   = np.sqrt(np.diag(singular_values) * self._n_observations[key])
            Si  = np.diag(1./np.diag(S))

            eofs[key]    = eofs[key].reshape(self._n_variables[key], n_modes)
            VT, self._no_nan_index[key]   = remove_nan_cols(eofs[key].T)

            self._V[key]    = VT.T
            self._L[key] 	= self._V[key] @ S
            self._U[key]    = pcs[key]

            self._field_means[key]  = fields[key].mean(axis=0)
            self._field_stds[key]   = fields[key].std(axis=0)

            self._fields[key]       = fields[key] - fields[key].mean(axis=0)

        if self._analysis['is_normalized']:
            self.normalize()
