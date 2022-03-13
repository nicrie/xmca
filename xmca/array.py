#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports
# =============================================================================
import cmath
import os
import warnings
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import hilbert
from statsmodels.tsa.forecasting.theta import ThetaModel
from tqdm import tqdm

from xmca import __version__
from xmca.tools.array import (get_nan_cols, has_nan_time_steps, remove_mean,
                              remove_nan_cols, pearsonr, block_bootstrap)
from xmca.tools.rotation import promax
from xmca.tools.text import boldify_str, secure_str, wrap_str


# =============================================================================
# MCA
# =============================================================================
class MCA:
    '''Perform MCA on two ``numpy.ndarray``.

    MCA is a more general form of Principal Component Analysis (PCA)
    for two input fields (left, right). If both data fields are the same,
    it is equivalent to PCA.

    '''

    def __init__(self, *fields):
        '''Load data fields and store information about data size/shape.

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
        Let `left` and `right` be some geophysical fields (e.g. SST and SLP).
        To perform PCA on `left` use:

        >>> from xmca.array import MCA
        >>> pca = MCA(left)
        >>> pca.solve()
        >>> exp_var = pca.explained_variance()
        >>> pcs = pca.pcs()
        >>> eofs = pca.eofs()

        To perform MCA on `left` and `right` use:

        >>> mca = MCA(left, right)
        >>> mca.solve()
        >>> exp_var = mca.explained_variance()
        >>> pcs = mca.pcs()
        >>> eofs = mca.eofs()

        '''
        if len(fields) == 0:
            fields = np.array([])

        if len(fields) > 2:
            raise ValueError("Too many fields. Pass 1 or 2 fields.")

        if len(fields) == 2 and fields[0].shape[0] != fields[1].shape[0]:
            raise ValueError('''Time dimensions of given fields are different.
                Time series should have same time lengths.''')

        if not all(isinstance(f, np.ndarray) for f in fields):
            raise TypeError('''One or more fields are not `numpy.ndarray`.
            Please provide `numpy.ndarray` only.''')

        if any(has_nan_time_steps(f) for f in fields):
            raise ValueError('''One or more fields contain NaN time steps.
            Please remove these prior to analysis.''')

        # field meta information
        self._keys                  = ['left', 'right']
        self._fields                = {}  # input fields
        self._shape                 = {}  # input field shapes
        self._field_names           = {}  # names of input fields
        self._field_means           = {}  # mean of fields
        self._field_stds            = {}  # standard deviation of fields
        self._fields_spatial_shape  = {}  # spatial shapes of fields
        self._n_variables           = {}  # number of variables of fields
        self._no_nan_index          = {}  # index of variables containing data
        self._n_observations        = {}  # number of observations/samples

        # set fields
        if len(fields) == 1:
            self._keys.pop()
        fields = {k: f for k, f in zip(self._keys, fields)}

        self._set_field_meta(fields)
        fields = self._reshape_to_2d(fields)
        self._set_no_nan_idx(fields)
        fields = self._remove_nan_cols(fields)
        self._set_field_means(fields)
        self._set_field_stds(fields)

        self._fields  = self._center(fields)

        # set meta information
        self._analysis = {
            'version': __version__,
            'is_bivariate': len(self._fields) > 1,
            'is_normalized': False,
            'is_coslat_corrected': False,
            'method': 'pca',
            'is_complex': False,
            'extend': False,
            'theta_period': 365,
            'is_rotated': False,
            'n_rot': 0,
            'power': 0,
            'is_truncated': False,
            'is_truncated_at': 0,
            'rank': 0,
            'total_covariance': 0.0,
            'total_squared_covariance': 0.0,
        }


        self._analysis['method']        = self._get_method_id()

    def _get_slice(self, input):
        ''' Create a appropriate slice object.

        If input is an int, create slice(0, input). If input is a slice, create
        slice(input.start - 1, input.stop, input.step).

        '''

        if np.issubdtype(type(input), np.integer) or input is None:
            if input is None:
                input = self._analysis['rank']
            output = slice(0, input)
        elif isinstance(input, slice):
            try:
                new_start = max(0, input.start - 1)
            except TypeError:
                new_start = 0
            try:
                new_stop = min(input.stop, self._analysis['rank'])
            except TypeError:
                new_stop = self._analysis['rank']
            new_step = input.step
            output = slice(new_start, new_stop, new_step)
        else:
            dtype = type(input)
            msg = 'Invalid type {:}. Must be either int or slice.'.format(dtype)
            raise ValueError(msg)

        return output

    def set_field_names(self, left='left', right='right'):
        '''Set the name of the left and/or right field.

        Field names will be reflected when results are plotted or saved.

        Parameters
        ----------
        left : string
            Name of the `left` field. (the default is 'left').
        right : string
            Name of the `right` field. (the default is 'right').

        '''
        self._field_names['left']   = left
        self._field_names['right']  = right

    def _set_field_meta(self, data: Dict[str, np.ndarray]) -> None:
        for k, field in data.items():
            self._shape[k] = field.shape
            self._n_observations[k] = field.shape[0]
            self._fields_spatial_shape[k] = field.shape[1:]
            self._n_variables[k] = np.product(field.shape[1:])
            self._field_names[k] = k

    def _center(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        centered = {}
        for k, field in data.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centered[k] = field - field.mean(axis=0)

        return centered

    def _set_field_means(self, data: Dict[str, np.ndarray]) -> None:
        for k, field in data.items():
            self._field_means[k] = field.mean(axis=0)

    def _set_field_stds(self, data: Dict[str, np.ndarray]) -> None:
        for k, field in data.items():
            self._field_stds[k] = field.std(axis=0)

    def _set_no_nan_idx(self, data: Dict[str, np.ndarray]) -> None:

        for k, field in data.items():
            self._no_nan_index[k] = ~(get_nan_cols(field))

    def _remove_nan_cols(
            self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: remove_nan_cols(field) for k, field in data.items()}

    def _reshape_to_2d(
            self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data_2d = {}
        for k, field in data.items():
            n_obs = field.shape[0]
            n_vars = np.product(field.shape[1:])

            field = field.reshape(n_obs, n_vars)
            data_2d[k] = field

        return data_2d

    def _get_method_id(self):
        return 'mca' if self._analysis['is_bivariate'] else 'pca'

    def _get_analysis_path(self, path=None):
        if path is None:
            name_folder = '_'.join(self._field_names.values())
            name_folder = secure_str(name_folder)
            path = os.path.join(os.getcwd(), 'xmca', name_folder)
        elif not os.path.isabs(path):
            path = os.path.abspath(path)

        return path

    def _create_analysis_path(self, path):
        path = self._get_analysis_path(path)

        if not os.path.exists(path):
            os.makedirs(path)

    def _scale_X(self, data_dict):
        std     = self._field_stds
        mean    = self._field_means
        scaled = data_dict.copy()

        for k, field in scaled.items():
            field -= mean[k]
        if self._analysis['is_normalized']:
            field /= std[k]

        return scaled

    def _scale_X_inverse(self, data_dict):
        std     = self._field_stds
        mean    = self._field_means

        scaled = data_dict

        for k, field in scaled.items():
            if self._analysis['is_normalized']:
                field *= std[k]
            field += mean[k]

        return scaled

    def _get_X(self, original_scale=False, real=False):
        X  = {k : f.copy() for k, f in self._fields.items()}

        if real:
            X  = {k : x.real for k, x in X.items()}

        if original_scale:
            X = self._scale_X_inverse(X)

        return X

    def _get_fields(self, original_scale=False):
        n_obs  = self._n_observations['left']
        n_var     = self._n_variables
        fshape = self._fields_spatial_shape
        no_nan_idx = self._no_nan_index
        fields_2d = self._get_X(original_scale=original_scale)

        fields = {}
        for k, X in fields_2d.items():
            dtype       = X.dtype
            fields[k]   = np.zeros([n_obs, n_var[k]], dtype=dtype) * np.nan
            fields[k][:, no_nan_idx[k]] = X
            # reshape eofs to have original input shape
            fields[k]   = fields[k].reshape((n_obs,) + fshape[k])

        return fields

    def apply_weights(self, left=None, right=None):
        '''Apply weights to the left and/or right field.

        Parameters
        ----------
        left : ndarray
            Weights for left field.
        right : ndarray
            Weights for right field.

        Examples
        --------
        Let `left` and `right` be some input data (e.g. SST and precipitation).

        >>> left = np.random.randn(100, 30)
        >>> right = np.random.randn(100, 40)

        Call constructor, apply weights and then solve:

        >>> left_weights = np.random.randn(1, 30)  # some random weights
        >>> right_weights = np.random.randn(1, 40)
        >>> mca = MCA(left, right)
        >>> mca.apply_weights(lw, rw)
        >>> mca.solve()

        '''
        field_items = self._fields.items()

        weights = {'left' : left, 'right' : right}
        weights.update({k : 1 if w is None else w for k, w in weights.items()})
        self._fields.update({
            k : field * weights[k] for k, field in field_items
        })

    def normalize(self):
        '''Normalize each time series by its standard deviation.

        '''
        keys        = self._keys
        fields      = self._fields
        std  = self._field_stds

        for k in keys:
            self._fields[k] = fields[k] / std[k]

        self._analysis['is_normalized'] = True
        self._analysis['is_coslat_corrected'] = False
        self._analysis['method'] = self._get_method_id()
        return None

    def _theta_forecast(self, series):
        period = self._analysis['theta_period']
        steps = len(series)

        model = ThetaModel(
            series, period=period, deseasonalize=True, use_test=False
        ).fit()
        return model.forecast(steps=steps, theta=20)

    def _get_reg_coefs(self, x, y):
        assert(x.shape[0] == y.shape[0])
        N = x.shape[0]

        xmean = np.mean(x, axis=0)
        ymean = np.mean(y, axis=0)
        xstd  = np.mean(x, axis=0)

        # Compute covariance along time axis
        cov   = np.sum((x - xmean) * (y - ymean), axis=0) / N

        # Compute regression slope and intercept:
        slope     = cov / (xstd**2)
        intercept = ymean - xmean * slope
        return intercept, slope

    def _exp_forecast(self, field):
        N = field.shape[0]
        x = np.arange(N)
        x = np.repeat(x[:, np.newaxis], field.shape[1], axis=1)
        intercept, slope = self._get_reg_coefs(x, field)

        linear_end = slope * x[-1, :] + intercept
        series_end = field[-1, :]
        offset      = series_end - linear_end

        theta = self._analysis['theta_period']
        # start x at 1, because exp(0) would produce same value as the last
        # point of the original time series
        x_shift = x + 1
        exp_extension = offset * np.exp(-x_shift / theta)
        lin_extension = (slope * x) + linear_end

        return exp_extension + lin_extension

    def _extend(self, field):
        extend = self._analysis['extend']
        # Theta extension
        if extend == 'theta':
            extended = [self._theta_forecast(col) for col in tqdm(field.T)]
            extended = np.array(extended).T
        # Exponential extension
        elif extend == 'exp':
            extended = self._exp_forecast(field)
        else:
            error_message = '''{:} is not a valid extension. Choose either
            `exp` or `theta`.'''.format(extend)
            raise ValueError(error_message)

        return extended

    def _complexify(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        '''Complexify data via Hilbert transform.

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
        field : ndarray
            Real input field which is to be transformed via Hilbert transform.

        Returns
        -------
        ndarray
            Analytical signal of input field.

        '''
        n_observations = self._n_observations['left']

        for k in self._keys:
            fields[k] = fields[k].real
            if self._analysis['extend']:
                post    = self._extend(fields[k])
                pre     = self._extend(fields[k][::-1])[::-1]

                fields[k] = np.concatenate([pre, fields[k], post])

            # perform actual Hilbert transform of (extended) time series
            fields[k] = hilbert(fields[k], axis=0)

            if self._analysis['extend']:
                # cut out the first and last third of Hilbert transform
                # which belong to the forecast/backcast
                fields[k] = fields[k][n_observations:(2 * n_observations)]
                fields[k] = remove_mean(fields[k])

        return fields

    def _svd(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        Us = {}
        Ss = {}
        Vts = {}
        for k, f in fields.items():
            u, s, vt = np.linalg.svd(f, full_matrices=False)
            Us[k] = u
            Ss[k] = s
            Vts[k] = vt
        return Us, Ss, Vts

    def _get_min_mode(self, n=None, rotated=False):
        n_modes = [self._analysis['rank']]
        if n is not None:
            n_modes.append(n)

        if rotated:
            n_modes.append(self._analysis['n_rot'])

        return np.min(n_modes)

    def _get_max_mode(self, n=None, rotated=False):
        n_modes = [self._analysis['rank']]
        if n is not None:
            n_modes[0] = n

        if rotated:
            n_modes.append(self._analysis['n_rot'])

        return np.max(n_modes)

    def solve(self, complexify=False, extend=False, period=1):
        '''Call the solver to perform EOF analysis/MCA.

        Under the hood the method performs singular value decomposition on
        the covariance matrix.

        Parameters
        ----------
        complexify : boolean, optional
            Use Hilbert transform to complexify the input data fields
            in order to perform complex PCA/MCA. Default is false.
        extend : ['exp', 'theta', False], optional
            If specified, time series are extended by fore/backcasting based on
            either an exponential or a Theta model. Artificially extending
            the time series sometimes helps to reduce spectral leakage inherent
            to the Hilbert transform when time series are not stationary.
            Only used for complex time series i.e. when omplexify=True.
            Default is False.
        period : float, optional
            If Theta model, it represents the number of time steps for a
            season. If exponential model, it represents the number of time
            steps for the exponential to decrease to 1/e. If no extension is
            selected, this parameter has no effect. Default is 1.
        '''
        if any(np.isnan(field).all() for field in self._fields.values()):
            raise RuntimeError('''
            Fields are empty. Did you forget to load data?
            ''')

        self._analysis['is_complex']    = complexify
        self._analysis['extend']        = extend
        self._analysis['theta_period']  = period

        n_observations  = self._n_observations['left']
        dof = n_observations - 1

        # complexify input data via Hilbert transform
        if self._analysis['is_complex']:
            self._fields = self._complexify(self._fields)

        X = self._get_X()

        # perform PCA of input to speed up algorithm
        k, l, mt = self._svd(X)
        R = {key: k[key] * l[key] for key in self._keys}
        # create covariance matrix
        try:
            kernel = R['left'].conj().T @ R['right']
        except KeyError:
            try:
                kernel = R['left'].conj().T @ R['left']
            except KeyError as err:
                msg = (
                    'Error in creating kernel. Please report this issue on '
                    'GitHub'
                )
                raise KeyError(msg) from err
        kernel = kernel / dof

        # perform singular value decomposition
        try:
            VLeft, singular_values, VTRight = np.linalg.svd(
                kernel, full_matrices=False
            )
            VRight = VTRight.conj().T
            del(VTRight)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                '''SVD failed. NaN entries may be the problem.'''
            )

        V = {'left': VLeft, 'right': VRight}
        # singular vectors (EOFs) = V
        self._V = {key: mt[key].conj().T @ V[key] for key in self._keys}

        # free up space
        del(VLeft)
        del(VRight)

        self._singular_values = singular_values
        self._variance = singular_values
        self._var_idx = np.argsort(singular_values)[::-1]
        self._norm = {k: np.sqrt(singular_values) for k in self._keys}

        self._analysis['total_covariance'] = singular_values.sum()
        self._analysis['total_squared_covariance'] = (singular_values**2).sum()
        self._analysis['rank'] = len(singular_values)
        self._analysis['is_rotated']    = False
        self._analysis['n_rot'] = len(singular_values)
        self._analysis['power']         = 0
        self._rotation_matrix           = np.eye(len(singular_values))
        self._correlation_matrix        = np.eye(len(singular_values))
        self._analysis['is_truncated_at'] = len(singular_values)

    def _get_svals(self, n=None):
        modes = self._get_slice(n)
        try:
            return self._singular_values[modes]
        except AttributeError:
            raise RuntimeError(
                'Cannot retrieve singular values. '
                'Please call the method `solve` first.'
            )

    def _get_V(self, n=None, rotated=True):
        if rotated:
            max_mode = self._analysis['n_rot']
        else:
            max_mode = n.stop if isinstance(n, slice) else n
        keep_modes = self._get_slice(n)

        try:
            V = {k: v[:, :max_mode] for k, v in self._V.items()}
        except AttributeError:
            raise RuntimeError(
                'Cannot retrieve singular vectors. '
                'Please call the method `solve` first.'
            )

        for k in self._keys:
            if rotated:
                sqrt_svals = np.sqrt(self._get_svals(max_mode))
                norm    = self._get_norm(max_mode, sorted=False)
                R       = self.rotation_matrix()
                var_idx = self._var_idx
                V[k] = V[k] * sqrt_svals @ R / norm[k]
                # reorder according to variance
                V[k] = V[k][:, var_idx]

            V[k] = V[k][:, keep_modes]

        return V

    def _get_U(self, n=None, rotated=True):
        if rotated:
            max_mode = self._analysis['n_rot']
        else:
            max_mode = n.stop if isinstance(n, slice) else n
        keep_modes = self._get_slice(n)

        fields  = self._get_X()
        V = self._get_V(max_mode, rotated=False)
        sqrt_svals = np.sqrt(self._get_svals(max_mode))
        R       = self.rotation_matrix(inverse_transpose=True)
        var_idx = self._var_idx

        U = {}
        for k in self._keys:
            U[k] = fields[k] @ V[k] / sqrt_svals
            if rotated:
                U[k] = U[k] @ R
                # reorder according to variance
                U[k] = U[k][:, var_idx]
            U[k] = U[k][:, keep_modes]

        return U

    def _get_eofs(
            self, n=None,
            scaling='None', phase_shift=0, rotated=True):

        V = self._get_V(n, rotated=rotated)
        n_var       = self._n_variables
        no_nan_idx  = self._no_nan_index
        field_shape = self._fields_spatial_shape
        eofs = {}

        for k in self._keys:
            # create data fields with original NaNs
            dtype       = V[k].dtype
            n_modes = V[k].shape[1]
            eofs[k]   = np.zeros([n_var[k], n_modes], dtype=dtype) * np.nan
            eofs[k][no_nan_idx[k], :] = V[k]
            # reshape eofs to have original input shape
            eofs[k]   = eofs[k].reshape(field_shape[k] + (n_modes,))
            # apply phase shift
            if self._analysis['is_complex']:
                eofs[k] *= cmath.rect(1, phase_shift)
            # scaling
            if scaling == 'None':
                pass
            # by eigenvalues (field units)
            elif scaling == 'eigen':
                # sqrt_svals = np.sqrt(self._get_svals(n_max_mode))
                # sqrt_svals[:n_max_mode]
                n_max_mode = V['left'].shape[1]
                norm = self._get_norm(n_max_mode, sorted=True)
                eofs[k] *= norm[k]
            # by maximum value
            elif scaling == 'max':
                eofs[k] /= np.nanmax(abs(eofs[k].real), axis=(0, 1))
            # by standard deviation
            elif scaling == 'std':
                eofs[k] /= np.nanstd(eofs[k].real, axis=(0, 1))
            else:
                msg = (
                    'The scaling option {:} is not valid. Please choose one '
                    'of the following: None, eigen, std, max'
                )
                msg = msg.format(scaling)
                raise ValueError(msg)

        return eofs

    def _get_pcs(
            self, n=None, scaling='None', phase_shift=0, rotated=True):

        U = self._get_U(n, rotated=rotated)

        for k in self._keys:
            # apply phase shift
            if self._analysis['is_complex']:
                U[k] *= cmath.rect(1, phase_shift)
            # scaling
            if scaling == 'None':
                pass
            # by eigenvalues
            elif scaling == 'eigen':
                norm = self._get_norm(n, sorted=True)
                U[k] *= (norm[k])
            # by maximum value
            elif scaling == 'max':
                U[k] /= np.nanmax(abs(U[k].real), axis=0)
            # by standard deviation
            elif scaling == 'std':
                U[k] /= np.nanstd(U[k].real, axis=0)
            else:
                msg = (
                    'The scaling option {:} is not valid. Please choose one '
                    'of the following: None, eigen, std, max'
                )
                msg = msg.format(scaling)
                raise ValueError(msg)

        return U

    def _get_norm(self, n=None, sorted=True):
        modes = self._get_slice(n)
        try:
            norm = self._norm
        except AttributeError:
            raise RuntimeError(
                'Cannot retrieve field norms. '
                'Please call the method `solve` first.'
            )
        if sorted:
            idx = self._var_idx
            norm = {k: nrm[idx] for k, nrm in norm.items()}

        norm = {k: nrm[modes] for k, nrm in norm.items()}

        return norm

    def _get_variance(self, n=None, sorted=True):
        norm = self._get_norm(n=n, sorted=sorted)
        return (
            norm['left'] * norm['right']
            if self._analysis['is_bivariate']
            else norm['left'] ** 2
        )

    def rotate(self, n_rot, power=1, tol=1e-8):
        '''Perform Promax rotation on the first `n` EOFs.

        Promax rotation (Hendrickson & White 1964) is an oblique rotation which
        seeks to find `simple structures` in the EOFs. It transforms the EOFs
        via an orthogonal Varimax rotation (Kaiser 1958) followed by the Promax
        equation. If `power=1`, Promax reduces to Varimax rotation. In general,
        a Promax transformation breaks the orthogonality of EOFs and introduces
        some correlation between PCs.

        Parameters
        ----------
        n_rot : int
            Number of EOFs to rotate. For values below 2, nothing is done.
        power : int, optional
            Power of Promax rotation. The default is 1 (= Varimax).
        tol : float, optional
            Tolerance of rotation process. The default is 1e-5.

        Raises
        ------
        ValueError
            If number of rotations are <2.

        Returns
        -------
        None.

        '''
        if(n_rot < 2):
            raise ValueError('`n_rot` must be > 1')
        if(power < 1):
            raise ValueError('`power` must be >=1')

        singular_values = self._get_svals(n_rot)
        sqrt_svals = np.sqrt(singular_values)
        V = self._get_V(n_rot, rotated=False)
        n_vars_left = V['left'].shape[0]

        # rotate loadings (Cheng and Dunkerton 1995)
        L = np.concatenate(list(V.values()))
        L = L * sqrt_svals
        L_rot, R, Phi = promax(L, power, maxIter=1000, tol=tol)

        # calculate variance/reconstruct rotated "singular_values"
        norm = {'left': np.linalg.norm(L_rot[:n_vars_left, :], axis=0)}
        norm['right']   = np.linalg.norm(L_rot[n_vars_left:, :], axis=0)
        if not self._analysis['is_bivariate']:
            norm['right'] = norm['left']

        variance = norm['left'] * norm['right']
        var_idx = np.argsort(variance)[::-1]

        self._norm = norm
        self._variance = variance
        self._var_idx = var_idx

        # store rotation and correlation matrix of PCs + meta information
        self._rotation_matrix           = R
        self._correlation_matrix        = Phi
        self._analysis['is_rotated']    = True
        self._analysis['n_rot']     = n_rot
        self._analysis['power']         = power

    def rotation_matrix(self, inverse_transpose=False):
        '''Return the rotation matrix used for rotation.

        For non-rotated solutions the rotation matrix equals the unit matrix.

        Parameters
        ----------
        inverse_transpose : boolean
            If True, return the inverse transposed of the rotation matrix.
            For orthogonal rotations (Varimax) it is the same as the rotation
            matrix. The default is False.

        Returns
        -------
        ndarray
            Rotation matrix

        '''
        try:
            R = self._rotation_matrix
        except AttributeError:
            n = len(self.singular_values())
            R = np.eye(n)

        # only for oblique rotations
        # If rotation is orthogonal: R = R^(-1).T
        # If rotation is oblique (p>1): R != R^(-1).T
        if inverse_transpose and self._analysis['power'] > 1:
            R = np.linalg.pinv(R).conjugate().T

        return R

    def correlation_matrix(self):
        '''
        Return the correlation matrix of PCs.

        For non-rotated and Varimax-rotated solutions the correlation matrix
        is equivalent to the unit matrix.

        Returns
        -------
        ndarray
            Correlation matrix.

        '''
        try:
            var_idx = self._var_idx
            return self._correlation_matrix[var_idx, :][:, var_idx]
        except AttributeError:
            n = len(self.singular_values())
            return np.eye(n)

    def fields(self, original_scale=False):
        '''Return `left` (and `right`) input field.

        Parameters
        ----------
        original_scale : boolean, optional
            If True, decenter and denormalize (if normalized) the input fields
            to obtain the original unit scale. Default is False.

        Returns
        -------
        dict[ndarray, ndarray]
            Fields associated to left and right input field.

        '''
        return self._get_fields(original_scale)

    def singular_values(self, n=None):
        '''Return the first `n` singular_values.

        Parameters
        ----------
        n : int, optional
            Number of singular_values to return. The default is 5.

        Returns
        -------
        ndarray
            Singular values of the obtained solution.

        '''
        return self._get_svals(n)

    def norm(self, n=None, sorted=True):
        '''Return L2 norm of first `n` loaded singular vectors.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default will return all modes.

        Returns
        -------
        dict[str, ndarray]
            L2 norm associated to each mode and vector.

        '''
        return self._get_norm(n=n, sorted=sorted)

    def variance(self, n=None, sorted=True):
        '''Return variance of first `n` loaded singular vectors.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default will return all modes.

        Returns
        -------
        dict[str, ndarray]
            Variance of each mode and vector.

        '''
        return self._get_variance(n=n, sorted=sorted)

    def scf(self, n=None):
        '''Return the SCF of the first `n` modes.

        The squared covariance fraction (SCF) is a measure of
        importance of each mode. It is calculated as the
        squared singular values divided by the sum of all squared singluar
        values.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is all.

        Returns
        -------
        ndarray
            Fraction of described squared covariance of each mode.

        '''
        variance  = self._variance[self._var_idx][:n]
        return variance**2 / self._analysis['total_squared_covariance'] * 100

    def explained_variance(self, n=None):
        '''Return the CF of the first `n` modes.

        The covariance fraction (CF) is a measure of
        importance of each mode. It is calculated as the
        singular values divided by the sum of all singluar values.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is all.

        Returns
        -------
        ndarray
            Fraction of described covariance of each mode.

        '''
        variance  = self._get_variance(n=n, sorted=True)
        return variance / self._analysis['total_covariance'] * 100

    def pcs(self, n=None, scaling='None', phase_shift=0, rotated=True):
        '''Return the first `n` PCs.

        Depending on the model the PCs may be real or complex, rotated or
        unrotated. Depending on the rotation type (Varimax/Proxmax), the PCs
        may be correlated.


        Parameters
        ----------
        n : int, optional
            Number of PCs to be returned. By default return all.
        scaling : {'None', 'eigen', 'max', 'std'}, optional
            Scale PCs by square root of eigenvalues ('eigen'), maximum value
            ('max') or standard deviation ('std').
        phase_shift : float, optional
            If complex, apply a phase shift to the PCs. Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated PCs while
            False returns the unrotated (original) PCs. Default is True.

        Returns
        -------
        dict[ndarray, ndarray]
            PCs associated to left and right input field.

        '''
        return self._get_pcs(n, scaling, phase_shift, rotated)

    def eofs(self, n=None, scaling='None', phase_shift=0, rotated=True):
        '''Return the first `n` EOFs.

        Depending on the model the PCs may be real or complex, rotated or
        unrotated. Depending on the rotation type (Varimax/Proxmax), the PCs
        may be correlated.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to be returned. The default is None.
        scaling : {'None', 'eigen', 'max', 'std'}, optional
            Scale by square root of eigenvalues ('eigen'), maximum value
            ('max') or standard deviation ('std').
        phase_shift : float, optional
            If complex, apply a phase shift to the EOFs. Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated PCs while
            False returns the unrotated (original) EOFs. Default is True.

        Returns
        -------
        dict[ndarray, ndarray]
            EOFs associated to left and right input field.

        '''
        return self._get_eofs(n, scaling, phase_shift, rotated)

    def spatial_amplitude(self, n=None, scaling='None', rotated=True):
        '''Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to be returned. If None, return all
            fields. The default is None.
        scaling : {'None', 'max'}, optional
            Scale by maximum value ('max'). The default is None.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated spatial
            amplitudes while False returns the unrotated (original) spatial
            amplitudes. Default is True.

        Returns
        -------
        dict[ndarray, ndarray]
            Spatial amplitude fields associated to left and right field.

        '''
        eofs = self.eofs(n, scaling='None', rotated=rotated)

        amplitudes = {}
        for key, eof in eofs.items():
            amplitudes[key] = np.sqrt(eof * eof.conjugate()).real

            if scaling == 'max':
                amplitudes[key] /= np.nanmax(amplitudes[key], axis=(0, 1))

        return amplitudes

    def spatial_phase(self, n=None, phase_shift=0, rotated=True):
        '''Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If None, all fields are returned.
            The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the spatial phase. Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated spatial
            phases while False returns the unrotated (original) spatial
            phases. Default is True.

        Returns
        -------
        dict[ndarray, ndarray]
            Spatial phase fields associated to left and right field.

        '''
        eofs = self.eofs(n, phase_shift=phase_shift, rotated=rotated)

        return {key: np.arctan2(eof.imag, eof.real).real for key, eof in eofs.items()}

    def temporal_amplitude(self, n=None, scaling='None', rotated=True):
        '''Return the temporal amplitude time series for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude series to be returned. If None, return all
            series. The default is None.
        scaling : {'None', 'max'}, optional
            Scale by maximum value ('max'). The default is None.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated temporal
            amplitudes while False returns the unrotated (original) temporal
            amplitudes. Default is True.

        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Temporal ampliude series associated to left and right field.

        '''
        pcs = self.pcs(n, scaling='None', rotated=rotated)

        amplitudes = {}
        for key, pc in pcs.items():
            amplitudes[key] = np.sqrt(pc * pc.conjugate()).real

            if scaling == 'max':
                amplitudes[key] /= np.nanmax(amplitudes[key], axis=0)

        return amplitudes

    def temporal_phase(self, n=None, phase_shift=0, rotated=True):
        '''Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, return all series.
            The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the temporal phase.
            Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated temporal
            phases while False returns the unrotated (original) temporal
            phases. Default is True.

        Returns
        -------
        amplitudes : dict[ndarray, ndarray]
            Temporal phase function associated to left and right field.

        '''
        pcs = self.pcs(n, phase_shift=phase_shift, rotated=rotated)

        return {key: np.arctan2(pc.imag, pc.real).real for key, pc in pcs.items()}

    def homogeneous_patterns(self, n=None, phase_shift=0):
        pcs = self._get_pcs(n=n, phase_shift=phase_shift)
        Xraw = self._get_X(real=True)

        r = {}
        p = {}
        for key in self._keys:
            r[key], p[key] = pearsonr(Xraw[key], pcs[key].real)

        n_var       = self._n_variables
        no_nan_idx  = self._no_nan_index
        field_shape = self._fields_spatial_shape

        rvals = {}
        pvals = {}
        for k in self._keys:
            # create data fields with original NaNs
            dtype       = r[k].dtype
            n_modes = r[k].shape[1]
            rvals[k]   = np.zeros([n_var[k], n_modes], dtype=dtype) * np.nan
            rvals[k][no_nan_idx[k], :] = r[k]
            # reshape eofs to have original input shape
            rvals[k]   = rvals[k].reshape(field_shape[k] + (n_modes,))

        for k in self._keys:
            # create data fields with original NaNs
            dtype       = p[k].dtype
            n_modes = p[k].shape[1]
            pvals[k]   = np.zeros([n_var[k], n_modes], dtype=dtype) * np.nan
            pvals[k][no_nan_idx[k], :] = p[k]
            # reshape eofs to have original input shape
            pvals[k]   = pvals[k].reshape(field_shape[k] + (n_modes,))

        return rvals, pvals

    def heterogeneous_patterns(self, n=None, phase_shift=0):
        pcs = self._get_pcs(n=n, phase_shift=phase_shift)
        Xraw = self._get_X(real=True)

        r = {}
        p = {}
        reverse = dict(zip(self._keys, self._keys[::-1]))
        for key in self._keys:
            try:
                r[key], p[key] = pearsonr(Xraw[key], pcs[reverse[key]].real)
            except KeyError:
                err = 'Key not found. Two fields needed for heterogenous maps.'
                raise KeyError(err)

        n_var       = self._n_variables
        no_nan_idx  = self._no_nan_index
        field_shape = self._fields_spatial_shape

        rvals = {}
        pvals = {}
        for k in self._keys:
            # create data fields with original NaNs
            dtype       = r[k].dtype
            n_modes = r[k].shape[1]
            rvals[k]   = np.zeros([n_var[k], n_modes], dtype=dtype) * np.nan
            rvals[k][no_nan_idx[k], :] = r[k]
            # reshape eofs to have original input shape
            rvals[k]   = rvals[k].reshape(field_shape[k] + (n_modes,))

        for k in self._keys:
            # create data fields with original NaNs
            dtype       = p[k].dtype
            n_modes = p[k].shape[1]
            pvals[k]   = np.zeros([n_var[k], n_modes], dtype=dtype) * np.nan
            pvals[k][no_nan_idx[k], :] = p[k]
            # reshape eofs to have original input shape
            pvals[k]   = pvals[k].reshape(field_shape[k] + (n_modes,))

        return rvals, pvals

    def _reconstructed_X(self, mode=None, original_scale=True):
        V = self._get_V(n=mode, rotated=True)
        U = self._get_pcs(n=mode, scaling='eigen', rotated=True)

        Xrec = {}
        for loc in self._keys:
            Xrec[loc] = U[loc] @ V[loc].conj().T
            # ensure real part only for complex solution
            Xrec[loc] = Xrec[loc].real

        if original_scale:
            Xrec = self._scale_X_inverse(Xrec)

        return Xrec

    def _reconstructed_fields(self, mode=None, original_scale=True):
        Xrec = self._reconstructed_X(
            mode=mode, original_scale=original_scale
        )
        # reshape to input dimensions
        n_obs = self._n_observations['left']
        n_vars = self._n_variables
        spatial_shape = self._fields_spatial_shape
        no_nan_idx = self._no_nan_index
        for k in Xrec.keys():
            rec_fields = np.zeros((n_obs, n_vars[k])) * np.nan
            rec_fields[:, no_nan_idx[k]]  = Xrec[k]
            Xrec[k] = rec_fields.reshape((-1,) + spatial_shape[k])

        return Xrec

    def reconstructed_fields(self, mode=None, original_scale=True):
        return self._reconstructed_fields(
            mode=mode, original_scale=original_scale
        )

    def predict(
            self, left=None, right=None,
            n=None, scaling='None', phase_shift=0):
        '''Predict PCs of new data.

        left and right are projected on the left and right singular
        vectors. If rotation was performed, the predicted PCs will be rotated
        as well.

        Parameters
        ----------
        left : ndarray
            Description of parameter `left`.
        right : ndarray
            Description of parameter `right`.
        n : int
            Number of PC modes to return. If None, return all modes.
            The default is None.
        scaling : {'None', 'eigen', 'max', 'std'}, optional
            Scale PCs by square root of eigenvalues ('eigen'), maximum value
            ('max') or standard deviation ('std').
        phase_shift : float, optional
            If complex, apply a phase shift to the temporal phase.
            Default is 0.

        Returns
        -------
        dict[ndarray, ndarray]
            Predicted PCs associated to left and right input field.

        '''
        keys = self._keys
        data = [left, right]
        data_new = {k: d.copy() for k, d in zip(keys, data) if d is not None}

        shape = self._shape
        n_vars = self._n_variables
        no_nan_idx = self._no_nan_index

        V = self._get_V(rotated=False)
        fields_mean = self._field_means

        sqrt_svals = np.sqrt(self._get_svals())

        R = self.rotation_matrix(inverse_transpose=True)
        n_rot = R.shape[0]
        var_idx = self._var_idx

        if n is None:
            n = R.shape[0]

        pcs_new = {}
        for k, x_new in data_new.items():
            try:
                x_new = x_new.reshape(x_new.shape[0], n_vars[k])
                x_new = x_new[:, no_nan_idx[k]]
            except ValueError as err:
                if (len(x_new.shape) != len(shape[k])):
                    msg = (
                        'Error in {:} field. '
                        'Dimension of new data ({:}) and the original field '
                        '({:}) do not match. '
                        'Did you forget the time dimension?'
                    )

                    msg = msg.format(k, len(x_new.shape), len(shape[k]))
                elif x_new.shape[1:] != fields_mean[k].shape:
                    msg = (
                        'Error in {:} field. '
                        'Spatial dimensions of new data {:} and the original '
                        'field {:} do not match.'
                    )
                    msg = msg.format(k, x_new.shape[1:], shape[k][1:])
                else:
                    msg = (
                        'Dimension mismatch in {:} field.'
                    )
                    msg = msg.format(k)
                raise ValueError(msg) from err
            try:
                x_new = self._scale_X({k: x_new})[k]
                # x_new -= fields_mean[k]
            except ValueError as err:
                msg = (
                    'Error in {:} field. '
                    'Spatial dimensions of new data {:} and the original '
                    'field {:} do not match.'
                )
                msg = msg.format(k, x_new.shape[1:], fields_mean[k].shape)
                raise ValueError(msg) from err
            #
            # if self._analysis['is_normalized']:
            #     x_new /= fields_std[k]

            pcs = x_new @ V[k][:, :n_rot] / sqrt_svals[:n_rot]
            pcs = pcs @ R
            # reorder according to variance
            pcs = pcs[:, var_idx]
            # take first n PCs only
            pcs = pcs[:, :n]

            # apply phase shift
            if self._analysis['is_complex']:
                pcs *= cmath.rect(1, phase_shift)
            # apply scaling
            # by eigenvalues (field units)
            if scaling == 'None':
                pass
            elif scaling == 'eigen':
                norm = self._get_norm(n, sorted=True)
                pcs *= norm[k]
            # by maximum value
            elif scaling == 'max':
                original_pcs = self._get_pcs(n, 'None', phase_shift)
                pcs /= np.nanmax(abs(original_pcs[k].real), axis=0)
            # by standard deviation
            elif scaling == 'std':
                original_pcs = self._get_pcs(n, 'None', phase_shift)
                pcs /= np.nanstd(original_pcs[k].real, axis=0)
            else:
                msg = (
                    'The scaling option {:} is not valid. Please choose one '
                    'of the following: None, eigen, std, max'
                )
                msg = msg.format(scaling)
                raise ValueError(msg)

            pcs_new[k] = pcs

        return pcs_new

    def plot(
            self, mode, threshold=0, phase_shift=0,
            cmap_eof=None, cmap_phase=None, figsize=(8.3, 5.0)):
        '''
        Plot results for `mode`.

        Parameters
        ----------
        mode : int, optional
            Mode to plot. The default is 1.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        phase_shift : float, optional
            If complex, apply a phase shift to the shown results. Default is 0.
        cmap_eof : str or Colormap
            The colormap used to map the spatial patterns.
            The default is 'Blues'.
        cmap_phase : str or Colormap
            The colormap used to map the spatial phase function.
            The default is 'twilight'.
        figsize : tuple
            Figure size provided to plt.figure().

        '''
        pcs     = self.pcs(mode, scaling='max', phase_shift=phase_shift)
        eofs    = self.eofs(mode, scaling='max')
        phases  = self.spatial_phase(mode, phase_shift=phase_shift)
        var     = self.explained_variance(mode)[-1]

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
            pcs[key]    = pcs[key][:, -1].real
            eofs[key]   = eofs[key][:, :, -1]
            phases[key]  = phases[key][:, :, -1]

            # apply amplitude threshold
            eofs[key]  = np.where(
                abs(eofs[key]) >= threshold, eofs[key], np.nan
            )
            phases[key] = np.where(
                abs(eofs[key]) >= threshold, phases[key], np.nan
            )

        titles = {
            'pc'    : r'PC {:d} ({:.1f} \%)'.format(mode, var),
            'eof'   : eof_title,
            'phase' : 'Phase',
            'var1'  : self._field_names['left']
        }
        if 'right' in self._keys:
            titles['var2'] = self._field_names['right']

        titles.update({k: v.replace('_', ' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})

        # create figure environment
        fig = plt.figure(figsize=figsize, dpi=150)
        fig.subplots_adjust(hspace=0.1, wspace=.1, left=0.25)
        gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
        axes_pc = [fig.add_subplot(gs[i, 0]) for i in range(n_rows - 1)]
        axes_eof = [fig.add_subplot(gs[i, 1]) for i in range(n_rows - 1)]
        cbax_eof = fig.add_subplot(gs[-1, 1])

        axes_space = axes_eof

        var_names = [titles['var1']]
        if 'right' in self._keys:
            var_names.append(titles['var2'])

        # plot PCs
        for i, pc in enumerate(pcs.values()):
            axes_pc[i].plot(pc)
            axes_pc[i].set_ylim(-1.2, 1.2)
            axes_pc[i].set_xlabel('')
            axes_pc[i].set_ylabel(var_names[i], fontweight='bold')
            axes_pc[i].set_title('')
            axes_pc[i].set_yticks([-1, 0, 1])
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
            axes_phase = [fig.add_subplot(gs[i, 2]) for i in range(n_rows - 1)]
            cbax_phase = fig.add_subplot(gs[-1, 2])

            for i, phase in enumerate(phases.values()):
                cb_phase = axes_phase[i].imshow(
                    phase, origin='lower',
                    vmin=-np.pi, vmax=np.pi, cmap=cmap_phase)
                axes_phase[i].set_title('')

            plt.colorbar(cb_phase, cbax_phase, orientation='horizontal')
            cbax_phase.xaxis.set_ticks([-3.14, 0, 3.14])
            cbax_phase.set_xticklabels([r'-$\pi$', '0', r'$\pi$'])

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

    def save_plot(
            self, mode, path=None, plot_kwargs={}, save_kwargs={}):
        '''Create and save a plot to local disk.

        Parameters
        ----------
        mode : int
            Mode to plot.
        path : str
            Path where to save the plot. If none is provided, an automatic
            name will be generated based on the mode number.
        plot_kwargs : dict
            Additional parameters provided to `xmca.array.plot`.
        save_kwargs : dict
            Additional parameters provided to `matplotlib.pyplot.savefig`.

        '''
        output = 'mode{:}.png'.format(mode) if path is None else path
        fig, axes = self.plot(mode=mode, **plot_kwargs)
        fig.subplots_adjust(left=0.06)
        plt.savefig(output, **save_kwargs)

    def truncate(self, n):
        '''Truncate the solution to the first `n` modes.

        This may be helpful when the full model takes up to much space to be
        saved.

        Parameters
        ----------
        n : int
            Number of modes to be retained.

        '''
        n_rot = self._analysis['n_rot']
        is_rotated = self._analysis['is_rotated']
        if (is_rotated & (n < n_rot)):
            raise ValueError(
                'Cannot truncte rotated solution. Please ensure `n` > `n_rot`'
            )
        if (n < self._singular_values.size):
            self._singular_values = self._singular_values[:n]

            for key in self._keys:
                self._V[key] = self._V[key][:, :n]

            self._analysis['is_truncated'] = True
            self._analysis['is_truncated_at'] = n

    def _create_info_file(self, path):
        sep_line = '\n#' + '-' * 79
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_header = (
            'This file contains information neccessary to load stored analysis'
            'data from xmca module.')

        path_output = os.path.join(path, 'info.xmca')

        with open(path_output, 'w+') as file:
            file.write(wrap_str(file_header))
            file.write('\n# To load this analysis use:')
            file.write('\n# from xmca.xarray import xMCA')
            file.write('\n# mca = xMCA()')
            file.write('\n# mca.load_analysis(PATH_TO_THIS_FILE)')
            file.write('\n')
            file.write(sep_line)
            file.write(sep_line)
            file.write('\n{:<20} : {:<57}'.format('created', now))
            file.write(sep_line)
            for key, name in self._field_names.items():
                file.write('\n{:<20} : {:<57}'.format(key, str(name)))
            file.write(sep_line)
            for key, info in self._analysis.items():
                if key in [
                        'is_bivariate', 'is_complex', 'is_rotated', 'is_truncated'
                ]:
                    file.write(sep_line)
                file.write('\n{:<20} : {:<57}'.format(key, str(info)))

    def _get_file_names(self, format):

        fields  = {}
        eofs    = {}
        pcs     = {}
        norm    = {}
        for key, variable in self._field_names.items():
            variable    = secure_str(variable)
            field_name  = variable
            eof_name    = '_'.join([variable, 'eofs'])

            fields[key] = '.'.join([field_name, format])
            eofs[key]   = '.'.join([eof_name, format])

        singular_values = 'singular_values'
        singular_values = '.'.join([singular_values, format])

        return {
            'fields'    : fields,
            'eofs'      : eofs,
            'pcs'       : pcs,
            'singular'  : singular_values,
            'norm'      : norm
        }

    def _save_data(self, data_array, path, *args, **kwargs):
        raise NotImplementedError('only works for `xarray`')

    def _set_analysis(self, key, value):
        try:
            key_type = type(self._analysis[key])
        except KeyError:
            raise KeyError("Key `{}` not found in info file.".format(key))
        self._analysis[key] = (
            (value == 'True') if key_type == bool else key_type(value)
        )

    def _set_info_from_file(self, path):

        with open(path, 'r') as info_file:
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

    def rule_n(self, n_runs, n_modes=None):
        '''Apply *Rule N* by Overland and Preisendorfer, 1982.

        The aim of Rule N is to provide a rule of thumb for the significance of
        the obtained singular values via Monte Carlo simulations of
        uncorrelated Gaussian random variables. The obtained singular values
        are scaled such that their sum equals the sum of true singular value
        spectrum.

        Parameters
        ----------
        n_runs : int
            Number of Monte Carlo simulations.
        n_modes : int
            Number of singular values to return.

        Returns
        -------
        DataArray
            Singular values obtained by Rule N.

        References
        ----------
        * Overland, J.E., Preisendorfer, R.W., 1982. A significance test for
            principal components applied to a cyclone climatology. Mon. Weather
            Rev. 110, 1–4.

        '''
        m = self._n_observations
        n = self._n_variables
        complexify = self._analysis['is_complex']
        rotated = self._analysis['is_rotated']
        n_rot = self._analysis['n_rot']
        power = self._analysis['power']

        svals = []

        for _ in tqdm(range(n_runs)):
            data = {k: np.random.standard_normal([m[k], n[k]]) for k in self._keys}
            model = MCA(*list(data.values()))
            model.solve(complexify=complexify)
            if rotated:
                try:
                    model.rotate(n_rot, power)
                except RuntimeError:
                    continue
            svals.append(model._get_variance())
            del(model)

        svals = np.array(svals).T
        ref = self._get_variance()
        svals /= svals.sum(axis=0) / ref.sum()
        n_modes = self._get_slice(n_modes)
        return svals[n_modes]

    def rule_north(self, n=None):
        '''Uncertainties of singular values based on North's *rule of thumb*.

        In case of compex PCA/MCA, the rule of thumb includes another factor of
        spqrt(2) according to Horel 1984.

        Parameters
        ----------
        n : int, optional
            Number of modes to be returned. By default return all.

        Returns
        -------
        ndarray
            Uncertainties associated to singular values.

        References
        ----------
        North, G, T L. Bell, R Cahalan, and F J. Moeng. 1982.
        “Sampling Errors in the Estimation of Empirical Orthogonal Functions.”
        Monthly Weather Review 110.
        https://doi.org/10.1175/1520-0493(1982)110<0699:SEITEO>2.0.CO;2.

        Horel, JD. 1984. “Complex Principal Component Analysis: Theory and
        Examples.” Journal of Climate and Applied Meteorology 23 (12): 1660–73.
        https://doi.org/10.1175/1520-0450(1984)023<1660:CPCATA>2.0.CO;2.

        '''

        svals = self._get_svals(n)
        n_obs = self._n_observations['left']
        is_complex = self._analysis['is_complex']

        err = svals * np.sqrt(2. / n_obs)

        if is_complex:
            err *= np.sqrt(2)

        return err

    def bootstrapping(
            self, n_runs, n_modes=20, axis=0, on_left=True, on_right=False,
            block_size=1, replace=True, strategy='standard',
            disable_progress=False):
        '''Perform Monte Carlo bootstrapping on model.

        Monte Carlo simulations allow to assess the signifcance of the
        obtained singular values and hence modes by re-performing the analysis
        on synthetic sample data. Using bootstrapping the synthetic data is
        created by resampling the original data.

        Parameters
        ----------
        n_runs : int
            Number of Monte Carlo simulations.
        n_modes : int
            Number of modes to be returned. By default return all modes.
        axis : int
            Whether to resample along time (axis=0) or in space (axis=1).
            The default is 0.
        on_left : bool
            Whether to resample the left field. True by default.
        on_right : bool
            Whether to resample the right field. False by default.
        block_size : int
            Resamples blocks of data of size `block_size`. This is particular
            useful when there is strong autocorrelation (e.g. annual cycle)
            which would be destroyed under standard bootstrapping. This
            procedure is known as moving-block bootstrapping. By default block
            size is 1.
        replace : bool
            Whether to resample with (bootstrap) or without replacement
            (permutation). True by default (bootstrap).
        strategy : ['standard', 'iterative']
            Whether to perform standard or iterative permutation. Standard
            permutation typically is overly conservative since it estimates
            the entire singular value spectrum at once. Iterative approach is
            more realistic taking into account each singular value before
            estimating the next one. The iterative approach usually takes much
            more time. Consult Winkler et al. (2020) for more details on
            the iterative approach.
        disable_progress : bool
            Whether to disable progress bar or not. By default False.

        Returns
        -------
        np.ndarray
            2d-array containing the singular values for each Monte Carlo
            simulation.

        References
        ----------
        Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap.
        Chapman and Hall. 436 pp.

        Winkler, A. M., Renaud, O., Smith, S. M. & Nichols, T. E. Permutation
        inference for canonical correlation analysis. NeuroImage 220, 117065
        (2020).


        '''
        # get meta information from original analysis
        complexify = self._analysis['is_complex']
        extend = self._analysis['extend']
        period = self._analysis['theta_period']
        is_rotated = self._analysis['is_rotated']
        n_rot = self._analysis['n_rot']
        power = self._analysis['power']

        n_modes_max = self._get_min_mode(n_modes, rotated=True)

        var_surr = np.zeros([n_modes_max, n_runs])

        # check each individual mode; for standard approach the singular value
        # spectrum is estimated within one single iteration. For iterative
        # approach, estimate first mode, then remove the mode from the data
        # and rerun the analysis on the residual data for testing the second
        # mode. Repeat this up to n_modes.
        for mode in tqdm(range(n_modes), disable=disable_progress):
            if strategy == 'standard':
                # get original data
                X_surr = self._get_X(original_scale=False, real=True)
            elif strategy == 'iterative':
                # get resuidual data by removing first modes
                X_surr = self._get_X(original_scale=False, real=True)
                X_rec = self._reconstructed_X(mode=mode, original_scale=False)
                for k in X_surr.keys():
                    X_surr[k] -= X_rec[k]
            # perform n_runs on bootstrapped data
            for run in tqdm(range(n_runs), disable=disable_progress, leave=True):
                # resample data
                if on_left and not on_right:
                    X_surr['left'] = block_bootstrap(
                        X_surr['left'], axis=axis, block_size=block_size,
                        replace=replace
                    )
                elif on_right and not on_left:
                    try:
                        X_surr['right'] = block_bootstrap(
                            X_surr['right'], axis=axis, block_size=block_size,
                            replace=replace
                        )
                    except KeyError as err:
                        msg = (
                            'No bootstrapping possible. There is no right field. '
                            'Set `on_right=False`.'
                        )
                        raise ValueError(msg) from err
                elif on_left:
                    concat = np.concatenate(list(X_surr.values()), axis=1)
                    concat = block_bootstrap(
                        concat, axis=axis, block_size=block_size,
                        replace=replace
                    )
                    X_surr['left'] = concat[:, :X_surr['left'].shape[1]]
                    X_surr['right'] = concat[:, X_surr['left'].shape[1]:]
                # perform analysis
                model = MCA(*list(X_surr.values()))
                model.solve(
                    complexify=complexify, extend=extend, period=period
                )
                if is_rotated:
                    try:
                        model.rotate(n_rot, power)
                    except RuntimeError:
                        continue

                var = model._get_variance(n_modes_max - mode)
                var_surr[mode:, run] = var
                del(model)

            if strategy == 'standard':
                break

        return var_surr

    def load_analysis(
            self, path,
            fields=None, eofs=None, singular_values=None):
        '''Load a model.

        This method allows to load a models which was saved by
        `.save_analysis()`.

        Parameters
        ----------
        path : str
            Location of the `.info` file created by `.save_analysis()`.
        fields : ndarray
            The original input fields.
        eofs : ndarray
            The obtained EOFs.
        singular_values : ndarray
            The obtained singular values.

        '''
        self._set_info_from_file(path)

        self._keys = ['left', 'right'] if self._analysis['is_bivariate'] else ['left']
        self._set_field_meta(fields)
        fields = self._reshape_to_2d(fields)
        self._set_no_nan_idx(fields)
        fields = self._remove_nan_cols(fields)
        self._set_field_means(fields)
        self._set_field_stds(fields)

        self._fields  = self._center(fields)

        if self._analysis['is_normalized']:
            self.normalize()
        if self._analysis['is_complex']:
            self._fields = self._complexify(self._fields)

        self._V = {}
        self._norm = {}
        self._singular_values = singular_values
        self._variance = singular_values
        self._var_idx = np.argsort(singular_values)[::-1]

        for key in self._keys:
            self._norm[key] = np.sqrt(singular_values)

            n_modes                         = eofs[key].shape[-1]
            eofs[key]    = eofs[key].reshape(self._n_variables[key], n_modes)
            VT   = remove_nan_cols(eofs[key].T)
            self._V[key]    = VT.T

        if self._analysis['is_rotated']:
            n_rot = self._analysis['n_rot']
            power = self._analysis['power']
            self.rotate(n_rot, power)

    def summary(self):
        '''Return meta information of the performed analysis.

        '''
        analysis = self._analysis
        strings_only = {k: str(v) for k, v in analysis.items()}
        print(yaml.dump(
            strings_only,
            sort_keys=False,
            default_flow_style=False
        ))
