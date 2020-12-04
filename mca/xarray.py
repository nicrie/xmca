#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complex rotated maximum covariance analysis of two xarray DataArrays.
"""

# =============================================================================
# Imports
# =============================================================================
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mca.array import MCA

# =============================================================================
# xMCA
# =============================================================================

class xMCA(MCA):
    """Perform maximum covariance analysis (MCA) for two `xr.DataArray` data fields.

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
        If True, input data is normalized to unit variance which translates to
        CCA. No normalization performs MCA. The default is True.


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

    def __init__(self, left, right=None, normalize=True, coslat=False):
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

        Returns
        -------
        None.

        """
        left  = left.copy()
        right = left if right is None else right.copy()

        assert(self._is_data_array(left))
        assert(self._is_data_array(right))

        left     = self._center_data_array(left)
        right    = self._center_data_array(right)

        self.normalize = normalize
        if self.normalize:
            left     = self._normalize_data_array(left)
            right    = self._normalize_data_array(right)


        if coslat:
            # coslat correction needs to happen AFTER normalization since
            # normalization mathematically removes the coslat correction effect
            left     = self._apply_coslat_correction(left)
            right    = self._apply_coslat_correction(right)
            # Deactivate normalization for array.MCA, otherwise
            # coslat correction may be overwritten
            self.normalize = False

        # constructor of base class for np.ndarray
        MCA.__init__(self, left.data, right.data, self.normalize)

        # store meta information of DataArrays
        self._timesteps	    = left.coords['time'].values
        self._left_lons 	= left.coords['lon'].values
        self._left_lats 	= left.coords['lat'].values
        self._right_lons 	= right.coords['lon'].values
        self._right_lats 	= right.coords['lat'].values

        # store meta information about analysis
        self._attrs = {
            'analysis'      : 'mca' if self._use_MCA else 'pca',
            'left_field'  : self._get_field_attr(left,'left_field','left_field'),
            'right_field' : self._get_field_attr(right,'right_field','right_field')
        }


    def set_field_names(self, left_field = None, right_field = None):
        if left_field is not None:
            self._attrs['left_field']     = left_field
        if right_field is not None:
            self._attrs['right_field']    = right_field


    def _is_data_array(self, data):
        """Check if data is of type `xr.DataArray`.

        Parameters
        ----------
        A : DataArray
            Input data.

        Returns
        -------
        bool
            Input data is of type `DataArray`.

        """
        return (isinstance(data,xr.DataArray))


    def _get_field_attr(self, data_array, attr, fallback='undefined'):
        try:
            return data_array.attrs[attr]
        except KeyError:
            return fallback


    def _center_data_array(self, data_array):
        return data_array - data_array.mean('time')


    def _normalize_data_array(self, data_array):
        return data_array / data_array.std('time')


    def _apply_coslat_correction(self, array):
        """Apply area correction to higher latitudes.

        """

        weights     = np.cos(np.deg2rad(array.lat))

        return array * weights



    def eigenvalues(self, n=None):
        """Return first `n` eigenvalues of the PCA.

        Parameters
        ----------
        n : int, optional
            Number of eigenvalues to return. If none, then all eigenvalues are returned.
            The default is None.

        Returns
        -------
        DataArray
            Eigenvalues of PCA.
        DataArray
            Uncertainty of eigenvalues according to North's rule of thumb.

        """
        # for n=Nonr, all eigenvalues are returned
        val, err = MCA.eigenvalues(self, n)

        # if n is not provided, take all eigenvalues
        if n is None:
            n = val.size

        modes = list(range(1,n+1))
        attrs = self._attrs
        attrs['name'] = 'eigenvalues'
        values = xr.DataArray(val,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= attrs['name'].replace('_',' '),
            attrs   = attrs)

        attrs['name'] = 'error_eigenvalues'
        error = xr.DataArray(err,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= attrs['name'].replace('_',' '),
            attrs   = attrs)

        return values, error


    def explained_variance(self, n=None):
        """Return the described variance of the first `n` PCs.

        Parameters
        ----------
        n : int, optioal
            Number of PCs to return. The default is None.

        Returns
        -------
        DataArray
            Described variance of each PC.
        DataArray
            Associated uncertainty according to North's `rule of thumb`.

        """
        variance, error 	= MCA.explained_variance(self, n)

        # if n is not provided, take all eigenvalues
        if n is None:
            n = variance.size

        modes = list(range(1,n+1))

        attrs = self._attrs
        attrs['name'] = 'explained_variance'
        variance = xr.DataArray(variance,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= attrs['name'].replace('_',' '),
            attrs   = attrs)

        attrs['name'] = 'error_explained_variance'
        error = xr.DataArray(error,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= attrs['name'].replace('_',' '),
            attrs   = attrs)

        return variance, error


    def pcs(self, n=None, scaling=0):
        """Return first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of PCs to return. If none, then all PCs are returned.
        The default is None.

        Returns
        -------
        DataArray
            PCs of left input field.
        DataArray
            PCs of right input field.

        """
        left_pcs, right_pcs = MCA.pcs(self, n, scaling=scaling)

        if n is None:
            n = left_pcs.shape[1]

        modes = list(range(1,n+1))

        attrs = self._attrs
        attrs['name'] = '_'.join([attrs['left_field'],'pcs'])
        left_pcs = xr.DataArray(
            data        = left_pcs,
            dims        = ['time','mode'],
            coords      = {'time' : self._timesteps, 'mode' : modes},
            name        = attrs['name'].replace('_',' '),
            attrs       = attrs)

        attrs['name'] = '_'.join([attrs['right_field'],'pcs'])
        right_pcs = xr.DataArray(
            data        = right_pcs,
            dims        = ['time','mode'],
            coords      = {'time' : self._timesteps, 'mode' : modes},
            name        = attrs['name'].replace('_',' '),
            attrs       = attrs)

        return left_pcs, right_pcs


    def eofs(self, n=None, scaling=0):
        """Return the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to return If none, all EOFs are returned.
            The default is None.

        Returns
        -------
        DataArray
            EOFs of left input field.
        DataArray
            EOFs of right input field.

        """
        left_eofs, right_eofs = MCA.eofs(self, n, scaling=scaling)

        if n is None:
            n = left_eofs.shape[-1]

        modes = list(range(1,n+1))

        attrs = self._attrs
        attrs['name'] = '_'.join([attrs['left_field'],'eofs'])
        left_eofs = xr.DataArray(
            data    = left_eofs,
            dims    = ['lat','lon','mode'],
            coords  = {
                'lon' : self._left_lons,
                'lat' : self._left_lats,
                'mode' : modes},
            name    = attrs['name'].replace('_',' '),
            attrs   = attrs)

        attrs['name'] = '_'.join([attrs['right_field'],'eofs'])
        right_eofs = xr.DataArray(
            data    = right_eofs,
            dims 	= ['lat','lon','mode'],
            coords  = {
                'lon' : self._right_lons,
                'lat' : self._right_lats,
                'mode' : modes},
            name    = attrs['name'].replace('_',' '),
            attrs   = attrs)

        return left_eofs, right_eofs


    def spatial_amplitude(self, n=None):
        """Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        left_eofs, right_eofs = self.eofs(n)

        left_amplitude   = np.sqrt(left_eofs * left_eofs.conjugate())
        right_amplitude  = np.sqrt(right_eofs * right_eofs.conjugate())

        attrs = self._attrs

        attrs['name'] = '_'.join([attrs['left_field'],'spatial_amplitude'])
        left_amplitude.name  = attrs['name'].replace('_',' ')
        left_amplitude.attrs = attrs

        attrs['name'] = '_'.join([attrs['right_field'],'spatial_amplitude'])
        right_amplitude.name = attrs['name'].replace('_',' ')
        right_amplitude.attrs = attrs


        # use the real part to force a real output
        return left_amplitude.real, right_amplitude.real


    def spatial_phase(self, n=None):
        """Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        left_eofs, right_eofs = self.eofs(n)

        left_phase = np.arctan2(left_eofs.imag,left_eofs.real)
        right_phase = np.arctan2(right_eofs.imag,right_eofs.real)

        attrs = self._attrs

        attrs['name'] = '_'.join([attrs['left_field'],'spatial_phase'])
        left_phase.name  = attrs['name'].replace('_',' ')
        left_phase.attrs = attrs

        attrs['name'] = '_'.join([attrs['right_field'],'spatial_phase'])
        right_phase.name = attrs['name'].replace('_',' ')
        right_phase.attrs = attrs

        # use the real part to force a real output
        return left_phase.real, right_phase.real


    def temporal_amplitude(self, n=None):
        """Return the temporal amplitude functions for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude functions to return. If none, all functions are returned.
            The default is None.

        Returns
        -------
        DataArray
            Temporal amplitude function of left input field.
        DataArray
            Temporal amplitude function of right input field.

        """
        left_pcs, right_pcs = self.pcs(n)

        left_amplitude   = np.sqrt(left_pcs * left_pcs.conjugate())
        right_amplitude  = np.sqrt(right_pcs * right_pcs.conjugate())

        attrs = self._attrs

        attrs['name'] = '_'.join([attrs['left_field'],'temporal_amplitude'])
        left_amplitude.name  = attrs['name'].replace('_',' ')
        left_amplitude.attrs = attrs

        attrs['name'] = '_'.join([attrs['right_field'],'temporal_amplitude'])
        right_amplitude.name = attrs['name'].replace('_',' ')
        right_amplitude.attrs = attrs

        # use the real part to force a real output
        return left_amplitude.real, right_amplitude.real


    def temporal_phase(self, n=None):
        """Return the temporal phase function for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of phase functions to return. If none, all functions are returned.
            The default is None.

        Returns
        -------
        DataArray
            Temporal phase function of left input field.
        DataArray
            Temporal phase function of right input field.

        """
        left_pcs, right_pcs = self.pcs(n)

        left_phase = np.arctan2(left_pcs.imag,left_pcs.real)
        right_phase = np.arctan2(right_pcs.imag,right_pcs.real)

        attrs = self._attrs

        attrs['name'] = '_'.join([attrs['left_field'],'temporal_phase'])
        left_phase.name  = attrs['name']
        left_phase.attrs = attrs

        attrs['name'] = '_'.join([attrs['right_field'],'temporal_phase'])
        right_phase.name = attrs['name']
        right_phase.attrs = attrs

        # use the real part to force a real output
        return left_phase.real, right_phase.real


    def _get_map_boundaries(self, data_array):
        assert(isinstance(data_array, xr.DataArray))

        east 	= data_array.coords['lon'].min()
        west 	= data_array.coords['lon'].max()
        south 	= data_array.coords['lat'].min()
        north 	= data_array.coords['lat'].max()

        boundaries = [east, west, south, north]
        return boundaries


    def _normalize_EOF_to_1(self, data_array):
        return data_array / abs(data_array).max(['lon','lat'])


    def _normalize_PC_to_1(self, data_array):
        return data_array / abs(data_array).max(['time'])


    def _create_figure(self, nrows=3, coltypes=['t','s'], longitude_center=0):
        n_rows, n_cols = [nrows, len(coltypes)]

        # positions of temporal plots
        is_temporal_col 	= [True if i=='t' else False for i in coltypes]

        # set projections associated with temporal/spatial plots
        proj_temporal_plot 	= None
        proj_spatial_plot 	= ccrs.PlateCarree(central_longitude=longitude_center)
        projections = [proj_temporal_plot if i=='t' else proj_spatial_plot for i in coltypes]

        # set relative width of temporal/spatial plots
        width_temporal_plot 	= 4
        width_spatial_plot 	= 5
        widths = [width_temporal_plot if i=='t' else width_spatial_plot for i in coltypes]

        # create figure environment
        fig 	= plt.figure(figsize = (7 * n_cols, 5 * n_rows))
        gs 		= GridSpec(n_rows, n_cols, width_ratios=widths)
        axes 	= np.empty((n_rows, n_cols), dtype=mpl.axes.SubplotBase)

        for i in range(n_rows):
            for j,proj in enumerate(projections):
                axes[i,j] = plt.subplot(gs[i,j], projection=proj)

        axes_pc = axes[:,is_temporal_col]
        axes_eof = axes[:,np.logical_not(is_temporal_col)]

        return fig, axes_pc, axes_eof


    def _validate_signs(self, signs, n):
        """Check if list of signs match the length n.

        Parameters
        ----------
        signs : list
            List of +-1s.
        n : int
            Length to check against.

        Raises
        ------
        ValueError
            If `n` does not match the length of `sign`.

        Returns
        -------
        signs : 1-ndarray
            Signs in the correct form.

        """
        # if nothing provided just take +1 as signs
        if signs is None:
            signs = np.ones(n)
        # otherwise check if signs provided by the user have correct lenght
        else:
            if (n == len(signs)):
                signs = np.array(signs)
            else:
                raise ValueError('Number of PCs and signs need to the same.')
        return signs


    def _flip_signs(self, data, signs):
        modes = data['mode'].size
        signs = self._validate_signs(signs, modes)

        return signs * data


    def _calculate_correlation(self, x, y):
        assert(self._is_data_array(x))
        assert(self._is_data_array(y))

        x = x - x.mean('time')
        y = y - y.mean('time')

        xy = (x*y).mean('time')
        sigx = x.std('time')
        sigy = y.std('time')

        return xy/sigx/sigy


    def homogeneous_patterns(self, n=None):
        """
        Return left and right homogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all patterns
            are returned. The default is None.

        Returns
        -------
        xr.DataArray
            Left homogeneous correlation maps.
        xr.DataArray
            Right homogeneous correlation maps.

        """

        left_pcs, right_pcs 		= self.pcs(n)
        left_pcs, right_pcs 		= [left_pcs.real, right_pcs.real]

        left_field  = self._left
        right_field = self._right

        left_hom_patterns 	= self._calculate_correlation(left_field,left_pcs)
        right_hom_patterns 	= self._calculate_correlation(right_field,right_pcs)

        return left_hom_patterns, right_hom_patterns


    def heterogeneous_patterns(self, n=None):
        """
        Return left and right heterogeneous correlation maps.

        Parameters
        ----------
        n : int, optional
            Number of patterns (modes) to be returned. If None then all patterns
            are returned. The default is None.

        Returns
        -------
        xr.DataArray
            Left heterogeneous correlation maps.
        xr.DataArray
            Right heterogeneous correlation maps.

        """
        left_pcs, right_pcs 		= self.pcs(n)
        left_pcs, right_pcs 		= [left_pcs.real, right_pcs.real]

        left_field  = self._left
        right_field = self._right

        left_het_patterns 	= self._calculate_correlation(left_field,right_pcs)
        right_het_patterns 	= self._calculate_correlation(right_field,left_pcs)

        return left_het_patterns, right_het_patterns


    def plot_mode(self, n=1, right=False, signs=None, title='', cmap='RdGy_r'):
        """
        Plot mode`n` PC and EOF of left (and right) data field.

        Parameters
        ----------
        n : int, optional
            Mode of PC and EOF to plot. The default is 1.
        right : boolean
            Plot PC and EOF of right field. The default is False.
        signs : list of int, optional
            Either +1 or -1 in order to flip the sign of shown PCs/EOFs.
            The default is None.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        left_pcs, right_pcs 		= self.pcs(n)
        left_pcs, right_pcs 		= [left_pcs.sel(mode=n).real, right_pcs.sel(mode=n).real]

        left_eofs, right_eofs 	= self.eofs(n)
        left_eofs, right_eofs 	= [left_eofs.sel(mode=n).real, right_eofs.sel(mode=n).real]

        var, error 			= self.explained_variance(n)
        var, error 			= [var.sel(mode=n).values, error.sel(mode=n).values]


        # normalize all EOFs/PCs such that they range from -1...+1
        left_eofs 		= self._normalize_EOF_to_1(left_eofs)
        right_eofs 		= self._normalize_EOF_to_1(right_eofs)
        left_pcs 		= self._normalize_PC_to_1(left_pcs)
        right_pcs 		= self._normalize_PC_to_1(right_pcs)

        # flip signs of PCs and EOFs, if needed
        left_eofs 	= self._flip_signs(left_eofs, signs)
        right_eofs 	= self._flip_signs(right_eofs, signs)
        left_pcs 	= self._flip_signs(left_pcs, signs)
        right_pcs 	= self._flip_signs(right_pcs, signs)

        # map boundaries as [east, west, south, north]
        left_map_boundaries  = self._get_map_boundaries(left_eofs)
        right_map_boundaries = self._get_map_boundaries(right_eofs)

        # map_projection and center longitude for
        map_projection = ccrs.PlateCarree()
        longitude_center  = int((left_map_boundaries[0] + left_map_boundaries[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        if right:
            fig, axes_pc, axes_eof = self._create_figure(2,['t','s'],longitude_center)
        else:
            fig, axes_pc, axes_eof = self._create_figure(1,['t','s'],longitude_center)



        # plot PCs/EOFs
        left_pcs.plot(ax = axes_pc[0,0])
        left_eofs.plot(
            ax = axes_eof[0,0], transform = map_projection, cmap = cmap,
            extend = 'neither',	add_colorbar = True, vmin = -1, vmax = 1,
            cbar_kwargs = {'label': 'EOF (normalized)'})
        axes_eof[0,0].set_extent(left_map_boundaries, crs = map_projection)

        if right:
            right_pcs.plot(ax = axes_pc[1,0])
            right_eofs.plot(
                ax=axes_eof[1,0], transform = map_projection, cmap = cmap,
                extend = 'neither', add_colorbar = True, vmin = -1, vmax = 1,
                cbar_kwargs = {'label': 'EOF (normalized)'})
            axes_eof[1,0].set_extent(right_map_boundaries, crs = map_projection)


        for i,a in enumerate(axes_pc[:,0]):
            a.set_ylim(-1,1)
            a.set_xlabel('')
            a.set_ylabel('PC (normalized)')
            a.set_title('')


        for i,a in enumerate(axes_eof[:,0]):
            a.coastlines(resolution='50m', lw=0.5)
            a.add_feature(cfeature.LAND.with_scale('50m'))
            a.set_title('')
            a.set_aspect('auto')


        fig.subplots_adjust(wspace=0.1,hspace=0.2,left=0.05)

        if title == '':
            title = "PC {} ({:.1f} $\pm$ {:.1f} \%)".format(n, var,error)

        if right:
            y_offset = 0.95
        else:
            y_offset = 1.00
        fig.suptitle(title, y=y_offset)


    def cplot_mode(self, n=1, right=False, threshold=0, title='', cmap='pink_r'):
        """
        Plot mode`n` PC and EOF of left (and right) data field.

        Parameters
        ----------
        n : int, optional
            Mode of PC and EOF to plot. The default is 1.
        right : boolean
            Plot PC and EOF of right field. The default is False.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        left_pcs, right_pcs 	= self.pcs(n)
        left_pcs, right_pcs 	= [left_pcs.sel(mode=n).real, right_pcs.sel(mode=n).real]

        left_amplitude, right_amplitude   = self.spatial_amplitude(n)
        left_amplitude, right_amplitude   = [left_amplitude.sel(mode=n), right_amplitude.sel(mode=n)]

        left_phase, right_phase           = self.spatial_phase(n)
        left_phase, right_phase           = [left_phase.sel(mode=n),right_phase.sel(mode=n)]

        var, error 		= self.explained_variance(n)
        var, error 		= [var.sel(mode=n).values, error.sel(mode=n).values]


        # normalize all EOFs/PCs such that they range from -1...+1
        left_amplitude   = self._normalize_EOF_to_1(left_amplitude)
        right_amplitude  = self._normalize_EOF_to_1(right_amplitude)
        left_pcs         = self._normalize_PC_to_1(left_pcs)
        right_pcs        = self._normalize_PC_to_1(right_pcs)

        # apply amplitude threshold
        left_amplitude   = left_amplitude.where(left_amplitude > threshold)
        right_amplitude  = right_amplitude.where(right_amplitude > threshold)
        left_phase       = left_phase.where(left_amplitude > threshold)
        right_phase      = right_phase.where(right_amplitude > threshold)

        # map boundaries as [east, west, south, north]
        left_map_boundaries  = self._get_map_boundaries(left_amplitude)
        right_map_boundaries = self._get_map_boundaries(right_amplitude)

        # map_projection and center longitude for
        map_projection = ccrs.PlateCarree()
        longitude_center  = int((left_map_boundaries[0] + left_map_boundaries[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        # create figure environment
        if right:
            fig, axes_pc, axes_eof = self._create_figure(2,['t','s','s'], longitude_center)
        else:
            fig, axes_pc, axes_eof = self._create_figure(1,['t','s','s'], longitude_center)



        # plot PCs/Amplitude/Phase
        left_pcs.real.plot(ax = axes_pc[0,0])
        left_amplitude.real.plot(
            ax = axes_eof[0,0], transform = map_projection,
            cmap = cmap, extend = 'neither', add_colorbar = True,
            vmin = 0, vmax = 1, cbar_kwargs = {'label' : 'Amplitude (normalized)'})
        left_phase.plot(
            ax = axes_eof[0,1], transform = map_projection,
            cmap = 'twilight_shifted', cbar_kwargs = {'label' : 'Phase (rad)'},
            add_colorbar = True, vmin = -np.pi, vmax = np.pi)

        axes_eof[0,0].set_extent(left_map_boundaries,crs = map_projection)
        axes_eof[0,1].set_extent(left_map_boundaries,crs = map_projection)

        axes_eof[0,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,error))
        axes_eof[0,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,error))


        if right:
            right_pcs.real.plot(ax = axes_pc[1,0])
            right_amplitude.real.plot(
                ax = axes_eof[1,0], transform = map_projection,
                cmap = cmap, extend = 'neither', add_colorbar = True, vmin = 0,
                vmax = 1, cbar_kwargs = {'label' : 'Amplitude (normalized)'})
            right_phase.plot(
                ax = axes_eof[1,1], transform = map_projection,
                cmap = 'twilight_shifted', cbar_kwargs = {'label' : 'Phase (rad)'},
                add_colorbar = True, vmin = -np.pi, vmax = 	np.pi)

            axes_eof[1,0].set_extent(right_map_boundaries,crs = map_projection)
            axes_eof[1,1].set_extent(right_map_boundaries,crs = map_projection)

            axes_eof[1,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,error))
            axes_eof[1,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(n,var,error))

        for a in axes_pc.flatten():
            a.set_ylabel('Real PC (normalized)')
            a.set_xlabel('')
            a.set_title('')


        for a in axes_eof.flatten():
            a.coastlines(lw = 0.5, resolution = '50m')
            a.set_aspect('auto')

        fig.subplots_adjust(wspace = 0.1, hspace = 0.17, left = 0.05)
        fig.suptitle(title)


    def plot_overview(self, n=3, right=False, signs=None, title='', cmap='RdGy_r'):
        """
        Plot first `n` PCs and EOFs of left data field.

        Parameters
        ----------
        n : int, optional
            Number of PCs and EOFs to plot. The default is 3.
        signs : list of int, optional
            List of +-1 in order to flip the sign of shown PCs/EOFs.
            Length of list has to match `n`. The default is None.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        left_pcs, right_pcs 		= self.pcs(n)
        left_pcs, right_pcs 		= [left_pcs.real, right_pcs.real]

        left_eofs, right_eofs 	= self.eofs(n)
        left_eofs, right_eofs 	= [left_eofs.real, right_eofs.real]

        var, error 			= self.explained_variance(n)
        var, error 			= [var.values, error.values]


        # normalize all EOFs/PCs such that they range from -1...+1
        left_eofs 		= self._normalize_EOF_to_1(left_eofs)
        right_eofs 		= self._normalize_EOF_to_1(right_eofs)
        left_pcs 		= self._normalize_PC_to_1(left_pcs)
        right_pcs 		= self._normalize_PC_to_1(right_pcs)

        # flip signs of PCs and EOFs, if needed
        left_eofs 	= self._flip_signs(left_eofs, signs)
        right_eofs 	= self._flip_signs(right_eofs, signs)
        left_pcs 	= self._flip_signs(left_pcs, signs)
        right_pcs 	= self._flip_signs(right_pcs, signs)

        # map boundaries as [east, west, south, north]
        left_map_boundaries = self._get_map_boundaries(left_eofs)
        right_map_boundaries = self._get_map_boundaries(right_eofs)

        # map_projection and center longitude for
        map_projection = ccrs.PlateCarree()
        longitude_center  = int((left_map_boundaries[0] + left_map_boundaries[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        if right:
            fig, axes_pc, axes_eof = self._create_figure(n,['t','s','s','t'], longitude_center)
        else:
            fig, axes_pc, axes_eof = self._create_figure(n,['t','s'], longitude_center)


        # plot PCs/EOFs
        for i in range(n):
            left_pcs.sel(mode = (i+1)).plot(ax = axes_pc[i,0])
            left_eofs.sel(mode = (i+1)).plot(
                ax = axes_eof[i,0],
                transform = map_projection, cmap = cmap, extend = 'neither',
                add_colorbar = True, vmin = -1,	vmax = 1,
                cbar_kwargs = {'label' : 'EOF (normalized)'})
            axes_eof[i,0].set_extent(left_map_boundaries,crs = map_projection)
            axes_eof[i,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))

        if right:
            for i in range(n):
                right_pcs.sel(mode = (i+1)).plot(ax = axes_pc[i,1])
                right_eofs.sel(mode = (i+1)).plot(
                    ax = axes_eof[i,1],
                    transform = map_projection, cmap = cmap, extend = 'neither',
                    add_colorbar = True, vmin = -1, vmax = 1,
                    cbar_kwargs = {'label': 'EOF (normalized)'})
                axes_eof[i,1].set_extent(right_map_boundaries,crs = map_projection)
                axes_eof[i,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))


        for a in axes_pc.flatten():
            a.set_ylim(-1,1)
            a.set_xlabel('')
            a.set_ylabel('PC (normalized)')
            a.set_title('')

        if right:
            for a in axes_pc[:,1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")

        # plot EOFs
        for a in axes_eof.flatten():
            a.coastlines(lw = 0.5)
            a.set_aspect('auto')

        fig.subplots_adjust(wspace = .1, hspace = 0.2, left = 0.05)
        fig.suptitle(title)


    def cplot_overview(self, n=3, right=False, threshold=0, title='', cmap='pink_r'):
        """
        Plot first `n` complex PCs of left data field alongside their corresponding EOFs.

        Parameters
        ----------
        n : int, optional
            Number of PCs and EOFs to plot. The default is 3.
        threshold : int, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        title : str, optional
            Title of figure. The default is ''.

        Returns
        -------
        None.

        """
        left_pcs, right_pcs 		= self.pcs(n)
        left_pcs, right_pcs 		= [left_pcs.real, right_pcs.real]

        left_amplitude, right_amplitude   = self.spatial_amplitude(n)
        left_phase, right_phase           = self.spatial_phase(n)

        var, error 			= self.explained_variance(n)
        var, error 			= [var.values, error.values]


        # normalize all EOFs/PCs such that they range from -1...+1
        left_amplitude 	= self._normalize_EOF_to_1(left_amplitude)
        right_amplitude = self._normalize_EOF_to_1(right_amplitude)
        left_pcs 		= self._normalize_PC_to_1(left_pcs)
        right_pcs 		= self._normalize_PC_to_1(right_pcs)

        # apply amplitude threshold
        left_amplitude   = left_amplitude.where(left_amplitude > threshold)
        right_amplitude  = right_amplitude.where(right_amplitude > threshold)
        left_phase       = left_phase.where(left_amplitude > threshold)
        right_phase      = right_phase.where(right_amplitude > threshold)


        # map boundaries as [east, west, south, north]
        left_map_boundaries     = self._get_map_boundaries(left_amplitude)
        right_map_boundaries    = self._get_map_boundaries(right_amplitude)

        # map_projection and center longitude for
        map_projection = ccrs.PlateCarree()
        longitude_center  = int((left_map_boundaries[0] + left_map_boundaries[1]) / 2)
        # take the center longitude of left field  for both, left and right
        # field as simplification; I don't know a way of specifying
        # multiple projections at the same time

        # create figure environment
        if right:
            fig, axes_pc, axes_eof = self._create_figure(n,['t','s','s','s','s','t'], longitude_center)
        else:
            fig, axes_pc, axes_eof = self._create_figure(n,['t','s','s'], longitude_center)



        # plot PCs/Amplitude/Phase
        for i in range(n):
            left_pcs.sel(mode=(i+1)).real.plot(ax = axes_pc[i,0])

            left_amplitude.sel(mode=(i+1)).real.plot(
                ax = axes_eof[i,0],
                transform = map_projection, cmap = cmap, extend = 'neither',
                add_colorbar = True, vmin = 0, vmax = 1,
                cbar_kwargs = {'label' : 'Amplitude (normalized)'})

            left_phase.sel(mode=(i+1)).plot(
                ax = axes_eof[i,1],
                transform = map_projection, cmap = 'twilight_shifted',
                cbar_kwargs = {'label' : 'Phase (rad)'}, add_colorbar = True,
                vmin = -np.pi, vmax = np.pi)

            axes_eof[i,0].set_extent(left_map_boundaries,crs=map_projection)
            axes_eof[i,1].set_extent(left_map_boundaries,crs=map_projection)

            axes_eof[i,0].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))
            axes_eof[i,1].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))

        if right:
            for i in range(n):
                right_pcs.sel(mode=(i+1)).real.plot(ax = axes_pc[i,1])

                right_amplitude.sel(mode=(i+1)).real.plot(
                    ax = axes_eof[i,2],
                    transform = map_projection, cmap = cmap, extend = 'neither',
                    add_colorbar = True, vmin = 0, vmax = 1,
                    cbar_kwargs = {'label' : 'Amplitude (normalized)'})

                right_phase.sel(mode=(i+1)).plot(
                    ax = axes_eof[i,3],
                    transform = map_projection, cmap = 'twilight_shifted',
                    cbar_kwargs = {'label': 'Phase (rad)'}, add_colorbar = True,
                    vmin = -np.pi, vmax = np.pi)

                axes_eof[i,2].set_extent(right_map_boundaries,crs=ccrs.PlateCarree())
                axes_eof[i,3].set_extent(right_map_boundaries,crs=ccrs.PlateCarree())

                axes_eof[i,2].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))
                axes_eof[i,3].set_title(r'Mode: {:d}: {:.1f} $\pm$ {:.1f} \%'.format(i+1,var[i],error[i]))

        if right:
            for a in axes_pc[:,1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")

        for a in axes_pc.flatten():
            a.set_ylabel('Real PC (normalized)')
            a.set_xlabel('')
            a.set_title('')


        for a in axes_eof.flatten():
            a.coastlines(lw = 0.5, resolution = '50m')
            a.set_aspect('auto')


        fig.subplots_adjust(wspace = 0.1, hspace = 0.17, left = 0.05)
        fig.suptitle(title)


    def split_complex(self, data_array):
        ds = xr.Dataset(
            data_vars = {
                'real': data_array.real,
                'imag': data_array.imag},
            attrs = data_array.attrs)

        return ds


    def to_netcdf(self, data_array, path, *args, **kwargs):
        method_idx  = self._attrs['analysis']
        complex_idx = 'c{:}'.format(int(self._useHilbert))
        rot_idx     = 'r{:02}'.format(self._nRotations)
        power_idx   = 'p{:02}'.format(self._power)
        file_name   = '.'.join([data_array.attrs['name'],'nc'])

        file_name   = '_'.join([method_idx,complex_idx,rot_idx,power_idx,file_name])
        finalPath   = os.path.join(path,file_name)

        if data_array.dtype == np.complex:
            dataset = self.split_complex(data_array)
        else:
            dataset = data_array.to_dataset(promote_attrs=True)

        dataset.to_netcdf(path=finalPath, *args, **kwargs)


    def save_analysis(self, path=None):
        analysis_name   = self._attrs['analysis']
        folder_pca      = self._attrs['left_field']
        folder_mca      = '_'.join([folder_pca, self._attrs['right_field']])
        folder_name     = folder_mca if self._use_MCA else folder_pca

        if path is None:
            path = os.getcwd()
        output_dir   = os.path.join(path, analysis_name)
        analysis_dir = os.path.join(output_dir,folder_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        self.to_netcdf(self.eofs()[0],analysis_dir)
        self.to_netcdf(self.pcs()[0],analysis_dir)
        self.to_netcdf(self.eigenvalues()[0],analysis_dir)

        if self._use_MCA:
            self.to_netcdf(self.eofs()[1],analysis_dir)
            self.to_netcdf(self.pcs()[1],analysis_dir)


    def _dataset_to_data_array(self, dataset):
        if self._is_data_array(dataset):
            return dataset
        else:
            try:
                data_array =  dataset['real'] + 1j * dataset['imag']
            except KeyError:
                raise KeyError('xr.Dataset needs two variables called \'real\' and \'imag\'.')
            except TypeError:
                raise TypeError("The input type seems incorrect. Take xr.DataArray or xr.Dataset")

            data_array.attrs = dataset.attrs
            return data_array


    def load_analysis(self, eofs=None, pcs=None, eigenvalues=None):
        # standardized fields // EOF fields + PCs
        if all(isinstance(var,list) for var in [eofs,pcs]):
            left_eofs, right_eofs   = [eofs[0], eofs[1]]
            left_pcs, right_pcs     = [pcs[0], pcs[1]]
            self._use_MCA           = True
        else:
            left_eofs, right_eofs   = [eofs, eofs]
            left_pcs, right_pcs     = [pcs, pcs]
            self._use_MCA           = False

        left_eofs   = self._dataset_to_data_array(left_eofs)
        right_eofs  = self._dataset_to_data_array(right_eofs)
        left_pcs    = self._dataset_to_data_array(left_pcs)
        right_pcs   = self._dataset_to_data_array(right_pcs)
        eigenvalues = self._dataset_to_data_array(eigenvalues)

        # store meta information of time steps and coordinates
        self._timesteps	    = left_pcs.coords['time'].values
        self._left_lons 	= left_eofs.coords['lon'].values
        self._right_lons 	= right_eofs.coords['lon'].values
        self._left_lats 	= left_eofs.coords['lat'].values
        self._right_lats 	= right_eofs.coords['lat'].values

        # store meta information about analysis
        self._attrs = {
            'analysis'    : 'mca' if self._use_MCA else 'pca',
            'left_field'  : self._get_field_attr(left_eofs,'left_field','left_field'),
            'right_field' : self._get_field_attr(right_eofs,'right_field','right_field')
        }


        MCA.load_analysis(
            self,
            eofs = [left_eofs.data, right_eofs.data],
            pcs =  [left_pcs.data, right_pcs.data],
            eigenvalues = eigenvalues.data)
