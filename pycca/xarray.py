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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from mca.array import MCA
from tools.text import secure_str, boldify_str, wrap_str
from tools.xarray import is_DataArray, check_dims, get_attr, calc_temporal_corr
from tools.xarray import get_lonlat_limits, norm_time_to_1, norm_space_to_1
from tools.xarray import split_complex, set_to_array, array_to_set, create_coords
# =============================================================================
# xMCA
# =============================================================================

class xMCA(MCA):
    """Perform Canonical Correlation Analysis with `xarray.DataArray`.

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
            The default is None.
        right : ndarray, optional
            Right input data. First dimension needs to be time.
            If none is provided, automatically, right field is assumed to be
            the same as left field. In this case, MCA reducdes to normal PCA.
            The default is None.

        Returns
        -------
        None.

        """
        left  = xr.DataArray(np.array([])) if left is None else left.copy()
        right = left if right is None else right.copy()

        is_DataArray(left)
        is_DataArray(right)

        check_dims(left, right)

        # constructor of base class for numpy.ndarray
        MCA.__init__(self, left.data, right.data)

        # store meta information of DataArrays
        # TODO: add left and right timesteps; change also in plot functions
        self._dims          = left.dims
        self._left_coords   = left.coords
        self._right_coords  = right.coords

        # store store meta information about analysis
        self._analysis['left_name'] = 'left' if left.name is None else left.name
        self._analysis['right_name']= 'right' if right.name is None else right.name



    def _get_fields(self):
        left = xr.DataArray(self._left, dims=self._dims, coords=self._left_coords)
        right = xr.DataArray(self._right, dims=self._dims, coords=self._right_coords)
        return left, right


    def set_field_names(self, left = None, right = None):
        if left is not None:
            self._analysis['left_name']     = left
        if right is not None:
            self._analysis['right_name']    = right


    def apply_weights(self,left_weights=None, right_weights=None):
        left, right = self._get_fields()

        if left_weights is not None:
            self._left  = (left * left_weights).data

        if right_weights is not None:
            self._right = (right * right_weights).data


    def apply_coslat(self):
        """Apply area correction to higher latitudes.

        """
        left_weights    = np.cos(np.deg2rad(self._left_coords['lat']))
        right_weights   = np.cos(np.deg2rad(self._right_coords['lat']))


        if (not self._analysis['is_coslat_corrected']):
            self.apply_weights(left_weights, right_weights)
            self._analysis['is_coslat_corrected'] = True
        else:
            print("Coslat correction already applied. Nothing was done.")


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
        values, error = MCA.eigenvalues(self, n)

        # if n is not provided, take all eigenvalues
        if n is None:
            n = values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        values = xr.DataArray(values,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'eigenvalues',
            attrs   = attrs)

        error = xr.DataArray(error,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'error eigenvalues',
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
        attrs = {k: str(v) for k, v in self._analysis.items()}

        variance = xr.DataArray(variance,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'explained variance',
            attrs   = attrs)

        error = xr.DataArray(error,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'error explained variance',
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
        attrs = {k: str(v) for k, v in self._analysis.items()}
        left_name = ' '.join([attrs['left_name'],'pcs'])

        left_pcs = xr.DataArray(
            data        = left_pcs,
            dims        = ['time','mode'],
            coords      = {'time' : self._left_coords['time'], 'mode' : modes},
            name        = left_name,
            attrs       = attrs)

        right_name = ' '.join([attrs['right_name'],'pcs'])
        right_pcs = xr.DataArray(
            data        = right_pcs,
            dims        = ['time','mode'],
            coords      = {'time' : self._right_coords['time'], 'mode' : modes},
            name        = right_name,
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
        attrs = {k: str(v) for k, v in self._analysis.items()}
        left_name = ' '.join([attrs['left_name'],'eofs'])

        left_eofs = xr.DataArray(
            data    = left_eofs,
            dims    = ['lat','lon','mode'],
            coords  = {
                'lon' : self._left_coords['lon'],
                'lat' : self._left_coords['lat'],
                'mode' : modes},
            name    = left_name,
            attrs   = attrs
            )

        right_name = ' '.join([attrs['right_name'],'eofs'])
        right_eofs = xr.DataArray(
            data    = right_eofs,
            dims 	= ['lat','lon','mode'],
            coords  = {
                'lon' : self._right_coords['lon'],
                'lat' : self._right_coords['lat'],
                'mode' : modes},
            name    = right_name,
            attrs   = attrs
            )

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

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'spatial amplitude'])
        left_amplitude.name  = left_name
        left_amplitude.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'spatial amplitude'])
        right_amplitude.name = right_name
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

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'spatial phase'])
        left_phase.name  = left_name
        left_phase.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'spatial phase'])
        right_phase.name = right_name
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

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'temporal amplitude'])
        left_amplitude.name  = left_name
        left_amplitude.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'temporal amplitude'])
        right_amplitude.name = right_name
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

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'temporal phase'])
        left_phase.name  = left_name
        left_phase.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'temporal phase'])
        right_phase.name = right_name
        right_phase.attrs = attrs

        # use the real part to force a real output
        return left_phase.real, right_phase.real


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

        left_hom_patterns 	= cal_temporal_corr(left_field,left_pcs)
        right_hom_patterns 	= cal_temporal_corr(right_field,right_pcs)

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'homogeneous patterns'])
        left_hom_patterns.name = left_name
        left_hom_patterns.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'homogeneous patterns'])
        right_hom_patterns.name = right_name
        right_hom_patterns.attrs = attrs

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

        left_het_patterns 	= cal_temporal_corr(left_field,right_pcs)
        right_het_patterns 	= cal_temporal_corr(right_field,left_pcs)

        attrs = {k: str(v) for k, v in self._analysis.items()}

        left_name = ' '.join([attrs['left_name'],'heterogeneous patterns'])
        left_het_patterns.name = left_name
        left_het_patterns.attrs = attrs

        right_name = ' '.join([attrs['right_name'],'heterogeneous patterns'])
        right_het_patterns.name = right_name
        right_het_patterns.attrs = attrs

        return left_het_patterns, right_het_patterns


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


    def plot(
        self, mode=1, threshold=0, cmap_eof='Blues', cmap_phase='twilight',
        resolution='110m'):
        """
        Plot mode `n`.

        Parameters
        ----------
        n : int, optional
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
        resolution : a named resolution to use from the Natural Earth dataset.
            Currently can be one of '110m', '50m', and '10m'.

        Returns
        -------
        None.

        """

        left_pcs, right_pcs 	= self.pcs(mode)
        left_pcs, right_pcs 	= [left_pcs.sel(mode=mode).real, right_pcs.sel(mode=mode).real]

        if self._analysis['is_complex']:
            left_eofs, right_eofs   = self.spatial_amplitude(mode)
            cmap_eof_range = [0, 1]
            eof_title = 'Amplitude'
        else:
            left_eofs, right_eofs   = self.eofs(mode)
            cmap_eof = 'RdBu_r'
            cmap_eof_range = [-1, 0, 1]
            eof_title = 'EOF'
        left_eofs, right_eofs   = [left_eofs.sel(mode=mode), right_eofs.sel(mode=mode)]

        left_phase, right_phase = self.spatial_phase(mode)
        left_phase, right_phase = [left_phase.sel(mode=mode),right_phase.sel(mode=mode)]

        var, error 		= self.explained_variance(mode)
        var, error 		= [var.sel(mode=mode).values, error.sel(mode=mode).values]

        titles = {
        'pc' : 'PC {:d} ({:.1f} \%)'.format(mode,var),
        'eof': eof_title,
        'phase':'Phase',
        'var1' : self._analysis['left_name'],
        'var2' : self._analysis['right_name']
        }

        titles.update({k: v.replace('_',' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})

        # normalize all EOFs/PCs such that they range from -1...+1
        left_eofs   = norm_space_to_1(left_eofs)
        right_eofs  = norm_space_to_1(right_eofs)
        left_pcs    = norm_time_to_1(left_pcs)
        right_pcs   = norm_time_to_1(right_pcs)

        # apply amplitude threshold
        left_eofs   = left_eofs.where(abs(left_eofs) >= threshold)
        right_eofs  = right_eofs.where(abs(right_eofs) >= threshold)
        left_phase  = left_phase.where(abs(left_eofs) >= threshold)
        right_phase = right_phase.where(abs(right_eofs) >= threshold)

        # map boundaries as [east, west, south, north]
        left_map_boundaries  = get_lonlat_limits(left_eofs)
        right_map_boundaries = get_lonlat_limits(right_eofs)
        map_boundaries = [left_map_boundaries, right_map_boundaries]

        # map projections and center longitude
        map_projection  = ccrs.PlateCarree()
        left_c_lon      = int((left_map_boundaries[0] + left_map_boundaries[1]) / 2)
        right_c_lon     = int((right_map_boundaries[0] + right_map_boundaries[1]) / 2)

        left_proj   = ccrs.PlateCarree(central_longitude=left_c_lon)
        right_proj  = ccrs.PlateCarree(central_longitude=right_c_lon)
        projs = [left_proj, right_proj]

        # data
        pcs             = [left_pcs, right_pcs]
        eofs            = [left_eofs, right_eofs]
        phases          = [left_phase, right_phase]
        height_ratios   = [1, 1]

        n_rows = 2
        n_cols = 3

        # if PCA then right field not necessary
        if (self._analysis['is_bivariate'] == False):
            n_rows = n_rows - 1
            projs.pop()
            pcs.pop()
            eofs.pop()
            phases.pop()
            height_ratios.pop()

        if (self._analysis['is_complex'] == False):
            n_cols = n_cols - 1

        # add additional row for colorbar
        n_rows = n_rows + 1
        height_ratios.append(0.05)

        figsize = (8.3,5) if self._analysis['is_bivariate'] else (8.3,2.5)
        # create figure environment
        fig = plt.figure(figsize=figsize, dpi=150)
        fig.subplots_adjust(hspace=0.1, wspace=.1, left=0.25)
        gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
        axes_pc = [fig.add_subplot(gs[i,0]) for i in range(n_rows-1)]
        axes_eof = [fig.add_subplot(gs[i,1], projection=projs[i]) for i in range(n_rows-1)]
        cbax_eof = fig.add_subplot(gs[-1,1])

        axes_space = axes_eof

        var_names = [titles['var1'], titles['var2']]

        # plot PCs
        for i, pc in enumerate(pcs):
            pc.plot(ax=axes_pc[i])
            axes_pc[i].set_ylim(-1,1)
            axes_pc[i].set_xlabel('')
            axes_pc[i].set_ylabel(var_names[i], fontweight='bold')
            axes_pc[i].set_title('')
            axes_pc[i].set_yticks([-1,0,1])


        #axes_pc[0].xaxis.set_visible(False)
        axes_pc[0].set_title(titles['pc'], fontweight='bold')

        # plot EOFs
        for i, eof in enumerate(eofs):
            cb_eof = eof.plot(
                ax=axes_eof[i], transform=ccrs.PlateCarree(),
                vmin=cmap_eof_range[0], vmax=cmap_eof_range[-1], cmap=cmap_eof,
                add_colorbar = False)
            axes_eof[i].set_extent(map_boundaries[i], crs=ccrs.PlateCarree())
            axes_eof[i].set_title('')

        plt.colorbar(cb_eof, cbax_eof, orientation='horizontal')
        cbax_eof.xaxis.set_ticks(cmap_eof_range)
        axes_eof[0].set_title(titles['eof'], fontweight='bold')

        # plot Phase function (if data is complex)
        if (self._analysis['is_complex']):
            axes_phase = [fig.add_subplot(gs[i,2], projection=projs[i]) for i in range(n_rows-1)]
            cbax_phase = fig.add_subplot(gs[-1,2])

            for i, phase in enumerate(phases):
                cb_phase = phase.plot(
                    ax=axes_phase[i], transform=ccrs.PlateCarree(),
                    vmin=-np.pi, vmax=np.pi, cmap=cmap_phase,
                    add_colorbar = False)
                axes_phase[i].set_extent(map_boundaries[i], crs=ccrs.PlateCarree())
                axes_phase[i].set_title('')

            plt.colorbar(cb_phase, cbax_phase, orientation='horizontal')
            cbax_phase.xaxis.set_ticks([-3.14,0,3.14])
            cbax_phase.set_xticklabels([r'-$\pi$','0',r'$\pi$'])

            for a in axes_phase:
                axes_space.append(a)

            axes_phase[0].set_title(titles['phase'], fontweight='bold')

        # add map features
        for a in axes_space:
            a.coastlines(lw = .5, resolution = resolution)
            a.set_aspect('auto')
            a.add_feature(cfeature.LAND, color='gray', zorder=0)


    def plot_overview(self, n=3, right=False, title='', cmap='RdGy_r'):
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
        left_pcs, right_pcs 	= self.pcs(n)
        left_pcs, right_pcs 	= [left_pcs.real, right_pcs.real]

        left_eofs, right_eofs 	= self.eofs(n)
        left_eofs, right_eofs 	= [left_eofs.real, right_eofs.real]

        var, error 			= self.explained_variance(n)
        var, error 			= [var.values, error.values]


        # normalize all EOFs/PCs such that they range from -1...+1
        left_eofs 		= norm_space_to_1(left_eofs)
        right_eofs 		= norm_space_to_1(right_eofs)
        left_pcs 		= norm_time_to_1(left_pcs)
        right_pcs 		= norm_time_to_1(right_pcs)

        # map boundaries as [east, west, south, north]
        left_map_boundaries     = get_lonlat_limits(left_eofs)
        right_map_boundaries    = get_lonlat_limits(right_eofs)

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
        left_amplitude 	= norm_space_to_1(left_amplitude)
        right_amplitude = norm_space_to_1(right_amplitude)
        left_pcs 		= norm_time_to_1(left_pcs)
        right_pcs 		= norm_time_to_1(right_pcs)

        # apply amplitude threshold
        left_amplitude   = left_amplitude.where(left_amplitude > threshold)
        right_amplitude  = right_amplitude.where(right_amplitude > threshold)
        left_phase       = left_phase.where(left_amplitude > threshold)
        right_phase      = right_phase.where(right_amplitude > threshold)


        # map boundaries as [east, west, south, north]
        left_map_boundaries     = get_lonlat_limits(left_amplitude)
        right_map_boundaries    = get_lonlat_limits(right_amplitude)

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


    def _save_data(self, data_array, path, *args, **kwargs):
        analysis_path   = path
        analysis_name   = self._get_analysis_id()
        var_name        = secure_str('.'.join([data_array.name,'nc']))

        file_name   = '_'.join([analysis_name, var_name])
        output_path = os.path.join(analysis_path,file_name)

        dataset = array_to_set(data_array)
        dataset.to_netcdf(path=output_path, *args, **kwargs)


    def save_analysis(self, path=None):
        analysis_path = self._get_analysis_path(path)

        self._create_info_file(analysis_path)
        self._save_info_to_file(analysis_path)
        self._save_paths_to_file(analysis_path, 'nc')

        left_eofs, right_eofs = self.eofs()
        left_pcs, right_pcs = self.pcs()
        eigenvalues = self.eigenvalues()[0]

        self._save_data(left_eofs, analysis_path)
        self._save_data(left_pcs, analysis_path)
        self._save_data(eigenvalues, analysis_path)

        if self._analysis['is_bivariate']:
            self._save_data(right_eofs, analysis_path)
            self._save_data(right_pcs, analysis_path)



    def load_analysis(self, info_file):
        paths = self._get_locs_from_file(info_file)
        self._set_info_from_file(info_file)

        eigenvalues = xr.open_dataset(paths['eigenvalues'])
        left_eofs   = xr.open_dataset(paths['field1_eofs'])
        left_pcs    = xr.open_dataset(paths['field1_pcs'])
        if self._analysis['is_bivariate']:
            right_eofs  = xr.open_dataset(paths['field2_eofs'])
            right_pcs   = xr.open_dataset(paths['field2_pcs'])
        else:
            right_eofs = left_eofs
            right_pcs = left_pcs

        left_eofs   = set_to_array(left_eofs)
        right_eofs  = set_to_array(right_eofs)
        left_pcs    = set_to_array(left_pcs)
        right_pcs   = set_to_array(right_pcs)
        # TODO: remove ugly .eigen... (only use dataarrays to store data, no datasets)
        eigenvalues = set_to_array(eigenvalues.eigenvalues)

        # store meta information of DataArrays
        self._left_coords = create_coords(
            left_pcs.coords['time'],
            left_eofs.coords['lon'],
            left_eofs.coords['lat']
            )
        self._right_coords = create_coords(
            right_pcs.coords['time'],
            right_eofs.coords['lon'],
            right_eofs.coords['lat']
            )

        MCA.load_analysis(
            self,
            info_file=info_file,
            eofs = [left_eofs.data, right_eofs.data],
            pcs =  [left_pcs.data, right_pcs.data],
            eigenvalues = eigenvalues.data)
