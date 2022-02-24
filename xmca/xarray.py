#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports
# =============================================================================
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xmca.array import MCA
from xmca.tools.text import boldify_str, secure_str
from xmca.tools.xarray import get_extent

# =============================================================================
# xMCA
# =============================================================================


class xMCA(MCA):
    '''Perform MCA on two ``xarray.DataArray``.

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
            The default is None.
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
        if len(fields) > 2:
            raise ValueError("Too many fields. Pass 1 or 2 fields.")

        if not all(isinstance(f, xr.DataArray) for f in fields):
            raise TypeError('''One or more fields are not `xarray.DataArray`.
            Please provide `xarray.DataArray` only.''')

        # set fields
        keys    = ['left', 'right']
        fields  = {keys[i] : field for i, field in enumerate(fields)}

        # store meta information of DataArrays
        self._field_dims    = {}  # dimensions of fields
        self._field_coords  = {}  # coordinates of fields

        for key, field in fields.items():
            self._field_dims[key]   = field.dims
            self._field_coords[key] = field.coords

        # constructor of base class for numpy.ndarray
        fields = {key : field.values for key, field in fields.items()}
        super().__init__(*fields.values())

    def _scale_X(self, data_dict):
        std     = self._field_stds
        mean    = self._field_means
        coords  = self._field_coords
        spatial_shape = self._fields_spatial_shape
        no_nan_idx = self._no_nan_index

        scaled = data_dict

        for k, field in scaled.items():
            scaled[k] -= mean[k]
        if self._analysis['is_normalized']:
            scaled[k] /= std[k]
        if self._analysis['is_coslat_corrected']:
            coslat = np.sqrt(np.cos(np.deg2rad(coords[k]['lat']))).values
            coslat = coslat.reshape(coslat.size, 1)

            weights = np.ones(spatial_shape[k]) * coslat
            weights = weights.flatten()[no_nan_idx[k]]

            scaled[k] *= weights
        return scaled

    def _scale_X_inverse(self, data_dict):
        std     = self._field_stds
        mean    = self._field_means
        coords  = self._field_coords
        spatial_shape = self._fields_spatial_shape
        no_nan_idx = self._no_nan_index

        scaled = data_dict

        for k, field in scaled.items():
            if self._analysis['is_coslat_corrected']:
                coslat = np.sqrt(np.cos(np.deg2rad(coords[k]['lat']))).values
                coslat = coslat.reshape(coslat.size, 1)

                weights = np.ones(spatial_shape[k]) * coslat
                weights = weights.flatten()[no_nan_idx[k]]

                field /= weights
            if self._analysis['is_normalized']:
                field *= std[k]
            field += mean[k]
            scaled[k] = field

        return scaled

    def apply_weights(self, **weights):
        fields = self.fields()
        n_obs = self._n_observations
        n_vars = self._n_variables
        no_nan_idx = self._no_nan_index

        for k, weight in weights.items():

            try:
                new_field  = (fields[k] * weight).data
            except KeyError as err:
                msg = (
                    'Key `{:}` not found. Please use `left` or `right`'
                )
                msg = msg.format(k)
                raise KeyError(msg) from err

            try:
                new_field = new_field.reshape(n_obs[k], n_vars[k])
                new_field = new_field[:, no_nan_idx[k]]
            except ValueError as err:
                msg = (
                    'Error for {:} weights. '
                    'Mismatch between dimensions of weights ({:}) '
                    'and original field ({:}).'
                )
                msg = msg.format(k, weight.shape, fields[k].shape)
                raise ValueError(msg) from err

            self._fields[k] = new_field

    def apply_coslat(self):
        '''Apply area correction to higher latitudes.

        '''

        coords  = self._field_coords
        weights = {}
        for key, coord in coords.items():
            # add small epsilon to assure correct handling of boundaries
            # e.g. 90.00001 degrees results in a negative value for sqrt
            epsilon = 1e-6
            weights[key] = np.sqrt(np.cos(np.deg2rad(coord['lat'])) + epsilon)

        self.apply_weights(**weights)
        self._analysis['is_coslat_corrected'] = True

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
        super().solve(complexify, extend, period)

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
            Number of EOFs to rotate.
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
        super().rotate(n_rot, power, tol)

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
        dims = self._field_dims
        coords = self._field_coords
        names = self._field_names
        fields = super().fields(original_scale)

        for k in self._keys:
            fields[k] = xr.DataArray(
                fields[k],
                dims=dims[k],
                coords=coords[k],
                name=names[k]
            )

        return fields

    def singular_values(self, n=None):
        '''Return first `n` singular values of the SVD.

        Parameters
        ----------
        n : int, optional
            Number of singular values to return. If None, all singular
            values are returned. The default is None.

        Returns
        -------
        DataArray
            Singular values of the SVD.

        '''

        attrs = {k: str(v) for k, v in self._analysis.items()}

        values = super().singular_values(n)

        slice = self._get_slice(n)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(values)]
        values = xr.DataArray(
            values,
            dims=['mode'],
            coords={'mode' : modes},
            name='singular values',
            attrs=attrs)

        return values

    def norm(self, n=None, sorted=True):
        '''Return L2 norm of first `n` loaded singular vectors.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. By default will return all modes.

        Returns
        -------
        dict[str, DataArray]
            L2 norm associated to each mode and vector.

        '''
        norms = super().norm(n=n, sorted=sorted)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names

        for k, data in norms.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:len(data)]
            norms[k] = xr.DataArray(
                data,
                dims=['mode'],
                coords={'mode' : modes},
                name=' '.join([field_names[k], 'norm']),
                attrs=attrs)

        return norms

    def variance(self, n=None, sorted=True):
        '''Return variance of first `n` loaded singular vectors.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. By default will return all modes.

        Returns
        -------
        dict[str, DataArray]
            Variance associated to each mode and vector.

        '''
        var = super().variance(n=n, sorted=sorted)

        slice = self._get_slice(n)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(var)]

        attrs = {k: str(v) for k, v in self._analysis.items()}

        var = xr.DataArray(
            var,
            dims=['mode'],
            coords={'mode' : modes},
            name='variance',
            attrs=attrs)

        return var

    def explained_variance(self, n=None):
        '''Return the CF of the first `n` modes.

        The covariance fraction (CF) is a measure of
        importance of each mode. It is calculated as the singular
        values divided by the sum of singular values.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is None.

        Returns
        -------
        DataArray
            Fraction of described covariance of each mode.

        '''
        expvar = super().explained_variance(n)

        slice = self._get_slice(n)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(expvar)]

        attrs = {k: str(v) for k, v in self._analysis.items()}

        expvar = xr.DataArray(
            expvar,
            dims=['mode'],
            coords={'mode' : modes},
            name='covariance fraction',
            attrs=attrs)

        return expvar

    def scf(self, n=None):
        '''Return the SCF of the first `n` modes.

        The squared covariance fraction (SCF) is a measure of
        importance of each mode. It is calculated as the squared singular
        values divided by the sum of squared singular values.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is None.

        Returns
        -------
        DataArray
            Fraction of described squared covariance of each mode.

        '''
        variance = super().scf(n)

        slice = self._get_slice(n)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(variance)]

        attrs = {k: str(v) for k, v in self._analysis.items()}

        variance = xr.DataArray(
            variance,
            dims=['mode'],
            coords={'mode' : modes},
            name='squared covariance fraction',
            attrs=attrs)

        return variance

    def pcs(self, n=None, scaling='None', phase_shift=0, rotated=True):
        '''Return first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of PCs to return. If none, then all PCs are returned.
        The default is None.
        scaling : {'None', 'eigen', 'max', 'std'}, optional
            Scale by square root of singular values ('eigen'), maximum value
            ('max') or standard deviation ('std'). The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the PCs. Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated PCs while
            False returns the unrotated (original) PCs. Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            PCs associated to left and right input field.

        '''
        pcs = super().pcs(n, scaling, phase_shift, rotated)

        attrs = {k: str(v) for k, v in self._analysis.items()}

        coords      = self._field_coords
        field_names = self._field_names
        for key, pc in pcs.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:pc.shape[-1]]
            pcs[key] = xr.DataArray(
                data=pc,
                dims=['time', 'mode'],
                coords={'time' : coords[key]['time'], 'mode' : modes},
                name=' '.join([field_names[key], 'pcs']),
                attrs=attrs)

        return pcs

    def eofs(self, n=None, scaling='None', phase_shift=0, rotated=True):
        '''Return the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to return If none, all EOFs are returned.
            The default is None.
        scaling : {None, 'eigen', 'max', 'std'}, optional
            Scale by square root of singular values ('eigen'), maximum value
            ('max') or standard deviation ('std'). The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the EOFs. Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated EOFs while
            False returns the unrotated (original) EOFs. Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            EOFs associated to left and right input field.
        '''
        eofs = super().eofs(n, scaling, phase_shift, rotated)

        attrs = {k: str(v) for k, v in self._analysis.items()}

        coords      = self._field_coords
        field_names = self._field_names

        for key, eof in eofs.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:eof.shape[-1]]
            eofs[key] = xr.DataArray(
                data=eof,
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name=' '.join([field_names[key], 'eofs']),
                attrs=attrs
            )

        return eofs

    def spatial_amplitude(self, n=None, scaling='None', rotated=True):
        '''Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to return. If none, all fields are
            returned. The default is None.
        scaling : {'None', 'max'}, optional
            Scale by maximum value ('max'). The default is None.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated spatial
            amplitudes while False returns the unrotated (original)
            spatial amplitudes. Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            Spatial amplitudes associated to left and right input field.
        '''
        amplitudes = super().spatial_amplitude(n, scaling, rotated)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        coords      = self._field_coords
        field_names = self._field_names

        for key, amp in amplitudes.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:amp.shape[-1]]
            amplitudes[key] = xr.DataArray(
                data=amp,
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon'   : coords[key]['lon'],
                    'lat'   : coords[key]['lat'],
                    'mode'  : modes
                },
                name=' '.join([field_names[key], 'spatial amplitude']),
                attrs=attrs
            )

        return amplitudes

    def spatial_phase(self, n=None, phase_shift=0, rotated=True):
        '''Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.
        scaling : {None, 'max', 'std'}, optional
            Scale by maximum value ('max') or
            standard deviation ('std'). The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the spatial phases.
            Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated spatial
            phases while False returns the unrotated (original) spatial phases.
            Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            Spatial phases associated to left and right input field.

        '''
        phases = super().spatial_phase(
            n, phase_shift=phase_shift, rotated=rotated
        )

        attrs = {k: str(v) for k, v in self._analysis.items()}
        coords      = self._field_coords
        field_names = self._field_names

        for key, pha in phases.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:pha.shape[-1]]
            phases[key] = xr.DataArray(
                data=pha,
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon'   : coords[key]['lon'],
                    'lat'   : coords[key]['lat'],
                    'mode'  : modes
                },
                name=' '.join([field_names[key], 'spatial phase']),
                attrs=attrs
            )

        return phases

    def temporal_amplitude(self, n=None, scaling='None', rotated=True):
        '''Return the temporal amplitude functions for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude functions to return. If none, all functions are
            returned. The default is None.
        scaling : {'None', 'max'}, optional
            Scale by maximum value ('max'). The default is None.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated temporal
            amplitudes while False returns the unrotated (original) temporal
            amplitudes. Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            PCs associated to left and right input field.
        '''
        amplitudes = super().temporal_amplitude(n, scaling, rotated)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        coords      = self._field_coords
        field_names = self._field_names

        for key, amp in amplitudes.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:amp.shape[-1]]
            amplitudes[key] = xr.DataArray(
                data=amp,
                dims=['time', 'mode'],
                coords={'time' : coords[key]['time'], 'mode' : modes},
                name=' '.join([field_names[key], 'temporal amplitude']),
                attrs=attrs
            )

        return amplitudes

    def temporal_phase(self, n=None, phase_shift=0, rotated=True):
        '''Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, all functions are
            returned. The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the temporal phases.
            Default is 0.
        rotated: boolean, optional
            When rotation was performed, True returns the rotated temporal
            phases while False returns the unrotated (original) temporal
            phases. Default is True.

        Returns
        -------
        dict[DataArray, DataArray]
            Temporal phases associated to left and right input field.
        '''
        phases = super().temporal_phase(
            n, phase_shift=phase_shift, rotated=rotated
        )

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names
        coords      = self._field_coords

        for key, pha in phases.items():
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))[:pha.shape[-1]]
            phases[key] = xr.DataArray(
                data=pha,
                dims=['time', 'mode'],
                coords={'time' : coords[key]['time'], 'mode' : modes},
                name=' '.join([field_names[key], 'temporal phase']),
                attrs=attrs
            )

        return phases

    def homogeneous_patterns(self, n=None, phase_shift=0):
        '''
        Return left and right homogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all
            patterns are returned. The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the homogeneous patterns.
            Default is 0.

        Returns
        -------
        dict[DataArray, DataArray]
            Homogeneous patterns associated to left and right input field.

        '''

        hom_pats, pvals = super().homogeneous_patterns(
            n=n, phase_shift=phase_shift
        )

        field_names = self._field_names
        coords      = self._field_coords
        attrs = {k: str(v) for k, v in self._analysis.items()}

        for key in self._keys:
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))
            modes = modes[:hom_pats[key].shape[-1]]

            hom_pats[key] = xr.DataArray(
                data=hom_pats[key],
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name=' '.join([field_names[key], 'homogeneous patterns']),
                attrs=attrs
            )

            pvals[key] = xr.DataArray(
                data=pvals[key],
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name=' '.join([field_names[key], 'pvalues homogeneous patterns']),
                attrs=attrs
            )

        return hom_pats, pvals

    def heterogeneous_patterns(self, n=None, phase_shift=0):
        '''
        Return left and right heterogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all
            patterns are returned. The default is None.
        phase_shift : float, optional
            If complex, apply a phase shift to the heterogeneous patterns.
            Default is 0.

        Returns
        -------
        dict[DataArray, DataArray]
            Heterogeneous patterns associated to left and right input field.

        '''
        het_pats, pvals = super().heterogeneous_patterns(
            n=n, phase_shift=phase_shift
        )

        field_names = self._field_names
        coords      = self._field_coords
        attrs = {k: str(v) for k, v in self._analysis.items()}

        for key in self._keys:
            slice = self._get_slice(n)
            modes = list(range(slice.start + 1, slice.stop + 1))
            modes = modes[:het_pats[key].shape[-1]]

            het_pats[key] = xr.DataArray(
                data=het_pats[key],
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name=' '.join([field_names[key], 'heterogeneous patterns']),
                attrs=attrs
            )

            pvals[key] = xr.DataArray(
                data=pvals[key],
                dims=['lat', 'lon', 'mode'],
                coords={
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name=' '.join([field_names[key], 'pvalues heterogeneous patterns']),
                attrs=attrs
            )

        return het_pats, pvals

    def reconstructed_fields(self, mode=slice(1, None), original_scale=True):
        '''Reconstruct original input fields based on specified `mode`s.

        Parameters
        ----------
        mode : int, slice
            Modes to be considered for reconstructing the original fields.
            The default is `slice(1, None)` which returns the original fields
            based on all modes.

        Returns
        -------
        dict[DataArray, DataArray]
            Left and right reconstructed fields.

        '''
        coords  = self._field_coords
        dims  = self._field_dims
        rec_fields = super().reconstructed_fields(
            mode=mode, original_scale=original_scale
        )

        for key in self._keys:
            rec_fields[key] = xr.DataArray(
                rec_fields[key],
                dims=dims[key],
                coords=coords[key],
                name='reconstructed_{:}_field'.format(key)
            )

        return rec_fields

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
        try:
            values = {k: d if d is None else d.values for k, d in zip(keys, data)}
        except AttributeError as err:
            msg = 'Please provide `xr.DataArray` to `left` and `right`'
            raise ValueError(msg) from err

        if self._analysis['is_bivariate']:
            pcs_new = super().predict(
                values['left'], values['right'], n, scaling, phase_shift
            )
        else:
            pcs_new = super().predict(
                values['left'], None, n, scaling, phase_shift
            )

        coords = {
            k: {
                'time' : d.coords['time'],
                'mode' : range(1, pcs_new[k].shape[1] + 1)
            } for k, d in zip(keys, data) if d is not None
        }
        dims = ('time', 'mode')
        for k, pc in pcs_new.items():
            pcs_new[k] = xr.DataArray(pcs_new[k], dims=dims, coords=coords[k])
        return pcs_new

    def _create_gridspec(
            self,
            figsize=None,
            orientation='horizontal',
            projection=None):
        is_bivariate    = self._analysis['is_bivariate']
        is_complex      = self._analysis['is_complex']

        n_rows = 2 if is_bivariate else 1
        n_cols = 3 if is_complex else 2
        height_ratios   = [1] * n_rows
        width_ratios    = [1] * n_cols
        # add additional row for colorbar
        n_rows += 1
        height_ratios.append(0.05)

        # horiontal layout
        if orientation == 'horizontal':
            grid = {
                'pc'   : {'left': [0, 0]},
                'eof'  : {'left': [0, 1]}
            }

            # position for phase
            if is_complex:
                grid['phase'] = {'left': [0, 2]}

            # positions for right field
            if is_bivariate:
                for k, panel in grid.items():
                    yx = panel.get('left')
                    grid[k]['right'] = [yx[0] + 1, yx[1]]

            # positions for colorbars
            for k, panel in grid.items():
                if k in ['eof', 'phase']:
                    row_cb = len(panel)
                    col_cb = panel.get('left')[1]
                    grid[k]['cb'] = [row_cb, col_cb]

        # vertical layout
        if orientation == 'vertical':
            grid = {
                'pc'   : {'left': [-1, 1]},
                'eof'  : {'left': [0, 1]}
            }

            # position for phase
            if is_complex:
                grid['phase'] = {'left': [1, 1]}

            # positions for right field
            if is_bivariate:
                for k, panel in grid.items():
                    yx = panel.get('left')
                    grid[k]['right'] = [yx[0], yx[1] + 1]

            # positions for colorbars
            for k, panel in grid.items():
                if k in ['eof', 'phase']:
                    row, col = panel.get('left')
                    grid[k]['cb'] = [row, col - 1]

            n_rows, n_cols = n_cols, n_rows
            height_ratios = n_rows * [1]
            width_ratios = n_cols * [1]
            width_ratios[0] = 0.05

        map_projs = {}
        for k1, data in grid.items():
            map_projs[k1] = {}
            for k2 in data.keys():
                map_projs[k1][k2] = None
                if k1 in ['eof', 'phase']:
                    if k2 in ['left', 'right']:
                        map_projs[k1][k2] = projection.get(k2, None)

        # create figure and axes
        fig = plt.figure(figsize=figsize, dpi=150)
        gs = fig.add_gridspec(
            n_rows, n_cols,
            height_ratios=height_ratios, width_ratios=width_ratios
        )
        axes = {}
        for key_data, data in grid.items():
            axes[key_data] = {}
            for key_pos, pos in data.items():
                row, col = pos
                axes[key_data][key_pos] = fig.add_subplot(
                    gs[row, col],
                    projection=map_projs[key_data][key_pos]
                )

        return fig, axes

    def plot(
            self, mode, threshold=0, phase_shift=0,
            cmap_eof=None, cmap_phase=None, figsize=(8.3, 5.0),
            resolution='110m', projection=None, orientation='horizontal',
            land=True):
        '''
        Plot results for `mode`.

        Parameters
        ----------
        mode : int, optional
            Mode to plot. The default is 1.
        threshold : [0,1], optional
            Threshold of max-normalised amplitude below which the fields are
            masked out. The default is 0.
        cmap_eof : str or Colormap
            The colormap used to map the spatial patterns.
            The default is `Blues`.
        cmap_phase : str or Colormap
            The colormap used to map the spatial phase function.
            The default is `twilight`.
        figsize : tuple
            Size of figure. Default is (8.3, 5.0).
        resolution : {None, '110m', '50m', '10m'}
            A named resolution to use from the Natural Earth dataset.
            Currently can be one of `110m`, `50m`, and `10m`. If None, no
            coastlines will be drawn. Default is `110m`.
        projection : cartopy.crs projection,
                     dict of {str : cartopy.crs projection}
            Projection can be either a valid cartopy projection (cartopy.crs)
            or a dictionary of different projections with keys 'left' and
            'right'. The default is cartopy.crs.PlateCaree.
        orientation : {'horizontal', 'vertical'}
            Orientation of the plot. Default is horizontal.
        land : boolean
            Turn coloring of land surface on/off.

        Returns
        -------
        matplotlib.figure.Figure :
            Figure instance.
        dict of matplotlib.axes._subplots.AxesSubplot :
            Dictionary of axes for `left` (and `right`) field containing
            `pcs`, `eofs` and, if complex, `phase`.

        '''
        complex     = self._analysis['is_complex']
        bivariate   = self._analysis['is_bivariate']

        # Get data
        var         = self.explained_variance(mode).sel(mode=mode).values
        pcs         = self.pcs(mode, scaling='max', phase_shift=phase_shift)
        eofs        = self.eofs(mode, scaling='max')
        phases      = self.spatial_phase(mode, phase_shift=phase_shift)
        if complex:
            eofs = self.spatial_amplitude(mode, scaling='max')

        # ticks
        ticks = {
            'pc'       : [-1, 0, 1],
            'eof'      : [0, 1] if complex else [-1, 0, 1],
            'phase'    : [-np.pi, 0, np.pi]
        }

        # tick labels
        tick_labels = {
            'phase' : [r'-$\pi$', '0', r'$\pi$']
        }

        # colormaps
        cmaps = {
            'eof'      : 'Blues' if complex else 'RdBu_r',
            'phase'    : 'twilight'
        }

        if cmap_eof is not None:
            cmaps['eof'] = cmap_eof
        if cmap_phase is not None:
            cmaps['phase'] = cmap_phase

        # titles
        titles = {
            'pc'        : 'PC',
            'eof'       : 'Amplitude' if complex else 'EOF',
            'phase'     : 'Phase',
            'mode'      : 'Mode {:d} ({:.1f} \%)'.format(mode, var)
        }

        for key, name in self._field_names.items():
            titles[key] = name

        titles.update({k: v.replace('_', ' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})

        # map projections and boundaries
        map = {
            # projections pf maps
            'projection' : {
                'left'  : ccrs.PlateCarree(),
                'right' : ccrs.PlateCarree()
            },
            # west, east, south, north limit of maps
            'boundaries' : {'left': None, 'right' : None}
        }
        if projection is not None:
            try:
                map['projection'].update(projection)
            except TypeError:
                map['projection'] = {
                    k: projection for k in map['projection'].keys()
                }
        data_projection  = ccrs.PlateCarree()

        # pre-process data and maps
        for key in pcs.keys():
            pcs[key] = pcs[key].sel(mode=mode).real
            eofs[key] = eofs[key].sel(mode=mode)
            phases[key] = phases[key].sel(mode=mode)

            # apply amplitude threshold
            eofs[key]   = eofs[key].where(abs(eofs[key]) >= threshold)
            phases[key] = phases[key].where(abs(eofs[key]) >= threshold)

            # map boundaries as [east, west, south, north]
            c_lon = map['projection'][key].proj4_params['lon_0']
            map['boundaries'][key] = get_extent(eofs[key], c_lon)

        # create figure panel
        fig, axes = self._create_gridspec(
            figsize=figsize,
            orientation=orientation,
            projection=map['projection']
        )

        # plot PCs, EOFs, and Phase
        for i, key in enumerate(pcs.keys()):
            # plot PCs
            pcs[key].plot(ax=axes['pc'][key])
            axes['pc'][key].set_ylim(-1.2, 1.2)
            axes['pc'][key].set_yticks([-1, 0, 1])
            axes['pc'][key].set_ylabel(titles[key], fontweight='bold')
            axes['pc'][key].set_xlabel('')
            axes['pc'][key].set_title('')
            axes['pc'][key].spines['right'].set_visible(False)
            axes['pc'][key].spines['top'].set_visible(False)

            # plot EOFs
            cb_eof = eofs[key].plot(
                ax=axes['eof'][key], transform=data_projection,
                vmin=ticks['eof'][0], vmax=ticks['eof'][-1], cmap=cmaps['eof'],
                add_colorbar=False)
            axes['eof'][key].set_extent(
                map['boundaries'][key],
                crs=data_projection
            )
            axes['eof'][key].set_title('')

            if resolution in ['110m', '50m', '10m']:
                axes['eof'][key].coastlines(lw=.4, resolution=resolution)
            if land:
                axes['eof'][key].add_feature(
                    cfeature.LAND,
                    color='#808080',
                    zorder=0
                )
            axes['eof'][key].set_aspect('auto')

            plt.colorbar(cb_eof, axes['eof']['cb'], orientation=orientation)
            if orientation == 'horizontal':
                axes['eof']['cb'].xaxis.set_ticks(ticks['eof'])
            elif orientation == 'vertical':
                axes['eof']['cb'].yaxis.set_ticks(ticks['eof'])

            # plot Phase function
            if complex:
                cb_phase = phases[key].plot(
                    ax=axes['phase'][key], transform=data_projection,
                    vmin=ticks['phase'][0], vmax=ticks['phase'][-1],
                    cmap=cmaps['phase'], add_colorbar=False)
                axes['phase'][key].set_extent(
                    map['boundaries'][key],
                    crs=data_projection
                )
                axes['phase'][key].set_title('')

                plt.colorbar(
                    cb_phase, axes['phase']['cb'],
                    orientation=orientation
                )
                if orientation == 'horizontal':
                    axes['phase']['cb'].xaxis.set_ticks(ticks['phase'])
                    axes['phase']['cb'].set_xticklabels(tick_labels['phase'])
                elif orientation == 'vertical':
                    axes['phase']['cb'].yaxis.set_ticks(ticks['phase'])
                    axes['phase']['cb'].set_yticklabels(tick_labels['phase'])

                if resolution in ['110m', '50m', '10m']:
                    axes['phase'][key].coastlines(lw=.4, resolution=resolution)
                if land:
                    axes['phase'][key].add_feature(
                        cfeature.LAND,
                        color='#808080',
                        zorder=0
                    )
                axes['phase'][key].set_aspect('auto')
                axes['phase']['left'].set_title(
                    titles['phase'],
                    fontweight='bold'
                )

        # special tweaking of axes according to orientation
        if orientation == 'horizontal':
            # titles
            axes['pc']['left'].set_title(titles['pc'], fontweight='bold')
            axes['eof']['left'].set_title(titles['eof'], fontweight='bold')

            if bivariate:
                axes['pc']['left'].xaxis.set_visible(False)
                axes['pc']['left'].spines['bottom'].set_visible(False)

        elif orientation == 'vertical':
            # titles
            axes['pc']['left'].set_ylabel(titles['pc'], fontweight='bold')
            axes['pc']['left'].set_title('')
            axes['eof']['left'].set_title(titles['left'], fontweight='bold')
            axes['eof']['cb'].set_ylabel(titles['eof'], fontweight='bold')
            axes['eof']['cb'].yaxis.set_label_position('left')
            axes['eof']['cb'].yaxis.set_ticks_position('left')
            if bivariate:
                axes['pc']['right'].yaxis.set_visible(False)
                axes['pc']['right'].spines['left'].set_visible(False)
                axes['eof']['right'].set_title(
                    titles['right'],
                    fontweight='bold'
                )

            if complex:
                axes['phase']['cb'].set_ylabel(
                    titles['phase'],
                    fontweight='bold'
                )
                axes['phase']['left'].set_title('')
                axes['phase']['cb'].yaxis.set_label_position('left')
                axes['phase']['cb'].yaxis.set_ticks_position('left')

        fig.subplots_adjust(wspace=.1)
        fig.suptitle(titles['mode'], horizontalalignment='left')

        return fig, axes

    def _save_data(self, data, path, engine='h5netcdf', *args, **kwargs):
        analysis_path   = path
        file_name        = secure_str('.'.join([data.name, 'nc']))

        output_path = os.path.join(analysis_path, file_name)

        invalid_netcdf = True
        if engine != 'h5netcdf':
            invalid_netcdf = False
        data.to_netcdf(
            path=output_path,
            engine=engine, invalid_netcdf=invalid_netcdf, *args, **kwargs
        )

    def save_analysis(self, path=None, engine='h5netcdf'):
        '''Save analysis including netcdf data and meta information.

        Parameters
        ----------
        path : str, optional
            Path to storage location. If none is provided, the analysis is
            saved into `./xmca/<left_name>(_<right_name>)/`.
        engine : str
            h5netcdf is needed for complex values. Otherwise standard netcdf
            works as well (the default is 'h5netcdf').

        '''
        analysis_path = self._get_analysis_path(path)
        self._create_analysis_path(analysis_path)
        self._create_info_file(analysis_path)

        fields      = self.fields(original_scale=True)
        eofs        = self.eofs(rotated=False)
        singular_values = self.singular_values()

        self._save_data(singular_values, analysis_path, engine)
        for key in self._keys:
            self._save_data(eofs[key], analysis_path, engine)
            # save storage and save only real part of fields
            # complex part can be cheaply reconstructed when loaded
            self._save_data(fields[key].real, analysis_path, engine)

    def load_analysis(self, path, engine='h5netcdf'):
        self._set_info_from_file(path)
        path_folder, _ = os.path.split(path)
        file_names = self._get_file_names(format='nc')

        path_eigen   = os.path.join(path_folder, file_names['singular'])
        singular_values = xr.open_dataarray(path_eigen, engine=engine).data

        fields  = {}
        eofs    = {}
        self._field_coords = {}

        for key in self._field_names.keys():
            path_fields   = os.path.join(
                path_folder, file_names['fields'][key]
            )
            path_eofs   = os.path.join(
                path_folder, file_names['eofs'][key]
            )

            eofs[key]   = xr.open_dataarray(path_eofs, engine=engine).data
            fields[key] = xr.open_dataarray(path_fields, engine=engine)
            self._field_coords[key] = fields[key].coords
            self._field_dims[key]   = fields[key].dims
            fields[key] = fields[key].data

        super().load_analysis(
            path=path,
            fields=fields,
            eofs=eofs,
            singular_values=singular_values)

        if self._analysis['is_coslat_corrected']:
            self.apply_coslat()

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
        uncertainties = super().rule_north(n=n)
        attrs = {k: str(v) for k, v in self._analysis.items()}
        slice = self._get_slice(n)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(uncertainties)]
        uncertainties = xr.DataArray(
            uncertainties,
            dims=['mode'],
            coords={'mode' : modes},
            attrs=attrs,
            name='singular values'
        )
        return uncertainties

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

        surr_svals = super().bootstrapping(
            n_runs=n_runs, n_modes=n_modes, axis=0,
            on_left=on_left, on_right=on_right, block_size=block_size,
            replace=replace, strategy=strategy,
            disable_progress=disable_progress
        )

        slice = self._get_slice(n_modes)
        modes = list(range(slice.start + 1, slice.stop + 1))[:len(surr_svals)]

        attrs = {k: str(v) for k, v in self._analysis.items()}
        surr_svals = xr.DataArray(
            surr_svals,
            dims=['mode', 'run'],
            coords={
                'mode' : modes,
                'run' : range(1, surr_svals.shape[1] + 1)
            },
            name='singular values',
            attrs=attrs
        )
        return surr_svals

    def summary(self):
        '''Return meta information of the performed analysis.

        '''
        super().summary()

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
        svals = super().rule_n(n_runs, n_modes)

        slice = self._get_slice(n_modes)
        modes = list(range(slice.start + 1, slice.stop + 1))[:svals.shape[0]]
        svals = xr.DataArray(
            svals,
            dims=['mode', 'run'],
            coords={
                'mode' : modes,
                'run' : np.arange(1, svals.shape[1] + 1)
            },
            name='singular values'
        )
        return svals
