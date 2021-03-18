#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complex rotated maximum covariance analysis of two xarray DataArrays.
"""

import cmath
# =============================================================================
# Imports
# =============================================================================
import os
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from xmca.array import MCA
from tools.text import boldify_str, secure_str, wrap_str
from tools.xarray import (calc_temporal_corr, check_dims, create_coords,
                          get_attr, get_extent, is_DataArray, split_complex)

# =============================================================================
# xMCA
# =============================================================================

class xMCA(MCA):
    """Perform Maximum Covariance Analysis (MCA) for two `xarray.DataArray`.

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

    >>> pca = MCA(data1)
    >>> pca.solve()
    >>> pcs = pca.pcs()
    >>> pcs['left'].sel(mode=1).plot()

    To perform MCA use:

    >>> mca = MCA(data1, data2)
    >>> mca.solve()
    >>> pcsData1, pcsData2 = mca.pcs()
    """

    def __init__(self, *data):
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
        if len(data) > 2:
            raise ValueError("Too many fields. Pass 1 or 2 fields.")

        # set fields
        keys    = ['left', 'right']
        fields  = {keys[i] : field for i,field in enumerate(data)}

        # store meta information of DataArrays
        self._field_dims    = {} # dimensions of fields
        self._field_coords  = {} # coordinates of fields

        for key,field in fields.items():
            self._field_dims[key]   = field.dims
            self._field_coords[key] = field.coords

        # constructor of base class for numpy.ndarray
        fields = {key : field.values for key, field in fields.items()}
        super().__init__(*fields.values())


    def _get_fields(self, original_scale=False):
        dims        = self._field_dims
        coords      = self._field_coords
        field_names = self._field_names
        fields      = super()._get_fields(original_scale=original_scale)

        for key in fields.keys():
            fields[key] = xr.DataArray(
                fields[key],
                dims = dims[key],
                coords = coords[key],
                name = field_names[key])

            if (original_scale & self._analysis['is_coslat_corrected']):
                fields[key] /= np.cos(np.deg2rad(coords[key]['lat']))

        return fields



    def apply_weights(self, **weights):
        fields = self._get_fields()

        try:
            for key, weight in weights.items():
                self._fields[key]  = (fields[key] * weight).data
        except KeyError:
            raise KeyError("Keys not found. Choose `left` or `right`")


    def apply_coslat(self):
        """Apply area correction to higher latitudes.

        """
        coords  = self._field_coords
        weights = {}
        for key, coord in coords.items():
            weights[key] = np.sqrt(np.cos(np.deg2rad(coord['lat'])))

        if (not self._analysis['is_coslat_corrected']):
            self.apply_weights(**weights)
            self._analysis['is_coslat_corrected'] = True
        else:
            print("Coslat correction already applied. Nothing was done.")


    def singular_values(self, n=None):
        """Return first `n` singular_values of the PCA.

        Parameters
        ----------
        n : int, optional
            Number of singular_values to return. If none, then all singular_values are returned.
            The default is None.

        Returns
        -------
        DataArray
            singular_values of PCA.
        DataArray
            Uncertainty of singular_values according to North's rule of thumb.

        """
        # for n=Nonr, all singular_values are returned
        values = super().singular_values(n)

        # if n is not provided, take all singular_values
        if n is None:
            n = values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        values = xr.DataArray(values,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'singular values',
            attrs   = attrs)

        # error = xr.DataArray(error,
        #     dims 	= ['mode'],
        #     coords 	= {'mode' : modes},
        #     name 	= 'error eigenvalues',
        #     attrs   = attrs)

        return values


    def explained_variance(self, n=None):
        """Return the CF of the first `n` modes.

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

        """
        variance 	= super().explained_variance(n)

        # if n is not provided, take all singular_values
        if n is None:
            n = variance.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        variance = xr.DataArray(variance,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'covariance fraction',
            attrs   = attrs)

        # error = xr.DataArray(error,
        #     dims 	= ['mode'],
        #     coords 	= {'mode' : modes},
        #     name 	= 'error explained variance',
        #     attrs   = attrs)

        return variance

    def scf(self, n=None):
        """Return the SCF of the first `n` modes.

        The squared covariance fraction (SCF) is a measure of
        importance of each mode. It is calculated as the squared singular
        values divided by the sum of squared singular values. In contrast to CF,
        SCF is invariant under CCA.

        Parameters
        ----------
        n : int, optional
            Number of modes to return. The default is None.

        Returns
        -------
        DataArray
            Fraction of described squared covariance of each mode.

        """
        variance 	= super().scf(n)

        # if n is not provided, take all singular_values
        if n is None:
            n = variance.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        variance = xr.DataArray(variance,
            dims 	= ['mode'],
            coords 	= {'mode' : modes},
            name 	= 'squared covariance fraction',
            attrs   = attrs)

        # error = xr.DataArray(error,
        #     dims 	= ['mode'],
        #     coords 	= {'mode' : modes},
        #     name 	= 'error explained variance',
        #     attrs   = attrs)

        return variance


    def pcs(self, n=None, scaling=None, phase_shift=0):
        """Return first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of PCs to return. If none, then all PCs are returned.
        The default is None.
        scaling : {None, 'eigen', 'max', 'std'}, optional
            Scale by singular_values ('eigen'), maximum value ('max') or
            standard deviation ('std'). The default is None.

        Returns
        -------
        DataArray
            PCs of left input field.
        DataArray
            PCs of right input field.

        """
        pcs = super().pcs(n, scaling, phase_shift)

        if n is None:
            n = self._singular_values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        coords      = self._field_coords
        field_names = self._field_names
        for key, pc in pcs.items():
            pcs[key] = xr.DataArray(
                data        = pc,
                dims        = ['time','mode'],
                coords      = {'time' : coords[key]['time'], 'mode' : modes},
                name        = ' '.join([field_names[key],'pcs']),
                attrs       = attrs)

        return pcs


    def eofs(self, n=None, scaling=None, phase_shift=0):
        """Return the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of EOFs to return If none, all EOFs are returned.
            The default is None.
        scaling : {None, 'eigen', 'max', 'std'}, optional
            Scale by singular_values ('eigen'), maximum value ('max') or
            standard deviation ('std'). The default is None.

        Returns
        -------
        DataArray
            EOFs of left input field.
        DataArray
            EOFs of right input field.

        """
        eofs = super().eofs(n, scaling, phase_shift)

        if n is None:
            n = self._singular_values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}

        coords      = self._field_coords
        field_names = self._field_names
        for key, eof in eofs.items():
            eofs[key] = xr.DataArray(
                data    = eof,
                dims    = ['lat','lon','mode'],
                coords  = {
                    'lon' : coords[key]['lon'],
                    'lat' : coords[key]['lat'],
                    'mode' : modes},
                name    = ' '.join([field_names[key],'eofs']),
                attrs   = attrs
                )

        return eofs


    def spatial_amplitude(self, n=None, scaling=None):
        """Return the spatial amplitude fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude fields to return. If none, all fields are returned.
            The default is None.
        scaling : {None, 'max'}, optional
            Scale by maximum value ('max'). The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        amplitudes = super().spatial_amplitude(n, scaling)

        if n is None:
            n = self._singular_values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}
        coords      = self._field_coords
        field_names = self._field_names

        for key, amp in amplitudes.items():
            amplitudes[key] = xr.DataArray(
                data    = amp,
                dims    = ['lat','lon','mode'],
                coords  = {
                'lon' : coords[key]['lon'],
                'lat' : coords[key]['lat'],
                'mode' : modes},
                name    = ' '.join([field_names[key],'spatial amplitude']),
                attrs   = attrs
                )

        return amplitudes


    def spatial_phase(self, n=None, phase_shift=0):
        """Return the spatial phase fields for the first `n` EOFs.

        Parameters
        ----------
        n : int, optional
            Number of phase fields to return. If none, all fields are returned.
            The default is None.
        scaling : {None, 'max', 'std'}, optional
            Scale by maximum value ('max') or
            standard deviation ('std'). The default is None.

        Returns
        -------
        DataArray
            Fields of left input field.
        DataArray
            Fields of right input field.

        """
        eofs = self.eofs(n, phase_shift=phase_shift)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names
        phases = {}
        for key, eof in eofs.items():
            phases[key]         = np.arctan2(eof.imag,eof.real).real
            phases[key].name    = ' '.join([field_names[key],'spatial phase'])
            phases[key].attrs   = attrs

        return phases


    def temporal_amplitude(self, n=None, scaling=None):
        """Return the temporal amplitude functions for the first `n` PCs.

        Parameters
        ----------
        n : int, optional
            Number of amplitude functions to return. If none, all functions are returned.
            The default is None.
        scaling : {None, 'max'}, optional
            Scale by maximum value ('max'). The default is None.

        Returns
        -------
        DataArray
            Temporal amplitude function of left input field.
        DataArray
            Temporal amplitude function of right input field.

        """
        amplitudes = super().temporal_amplitude(n, scaling)

        if n is None:
            n = self._singular_values.size

        modes = list(range(1,n+1))
        attrs = {k: str(v) for k, v in self._analysis.items()}
        coords      = self._field_coords
        field_names = self._field_names

        for key, amp in amplitudes.items():
            amplitudes[key] = xr.DataArray(
                data    = amp,
                dims    = ['time','mode'],
                coords  = {'time' : coords[key]['time'], 'mode' : modes},
                name    = ' '.join([field_names[key],'temporal amplitude']),
                attrs   = attrs
                )

        return amplitudes


    def temporal_phase(self, n=None, phase_shift=0):
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
        pcs = self.pcs(n, phase_shift=phase_shift)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names

        phases = {}
        for key,pc in pcs.items():
            phases[key] = np.arctan2(pc.imag,pc.real).real
            name = ' '.join([field_names[key],'temporal phase'])
            phases[key].name  = name
            phases[key].attrs = attrs

        return phases


    def homogeneous_patterns(self, n=None, phase_shift=0):
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

        fields  = self._get_fields()
        pcs     = self.pcs(n, phase_shift)

        field_names = self._field_names
        attrs = {k: str(v) for k, v in self._analysis.items()}
        hom_patterns = {}
        for key, field in fields.items():
            hom_patterns[key] = calc_temporal_corr(fields[key],pcs[key].real)
            name = ' '.join([field_names[key],'homogeneous patterns'])
            hom_patterns[key].name  = name
            hom_patterns[key].attrs = attrs

        return hom_patterns


    def heterogeneous_patterns(self, n=None, phase_shift=0):
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
        fields  = self._get_fields()
        pcs     = self.pcs(n, phase_shift)

        field_names = self._field_names
        attrs = {k: str(v) for k, v in self._analysis.items()}
        het_patterns = {}
        reverse = {'left' : 'right', 'right' : 'left'}
        for key, field in fields.items():
            try:
                het_patterns[key] = calc_temporal_corr(fields[key],pcs[reverse[key]].real)
            except KeyError:
                raise KeyError("Key not found. Two fields needed for heterogenous maps.")
            name = ' '.join([field_names[key],'heterogenous patterns'])
            het_patterns[key].name  = name
            het_patterns[key].attrs = attrs

        return het_patterns


    def reconstructed_fields(self, mode):
        eofs    = self.eofs(scaling=None)
        pcs     = self.pcs(scaling='eigen')
        coords  = self._field_coords
        std     = self._field_stds
        mean    = self._field_means

        rec_fields = {}
        for key in self._fields.keys():
            eofs[key]   = eofs[key].sel(mode=mode)
            pcs[key]    = pcs[key].sel(mode=mode)
            rec_fields[key] = xr.dot(pcs[key],eofs[key].conjugate(),dims=['mode'])
            rec_fields[key] = rec_fields[key].real

            if self._analysis['is_coslat_corrected']:
                rec_fields[key] /= np.sqrt(np.cos(np.deg2rad(coords[key]['lat'])))

            if self._analysis['is_normalized']:
                rec_fields[key] *= std[key]

            # add mean fields
            rec_fields[key]  += mean[key]

        return rec_fields


    def _create_gridspec(self, figsize=None, orientation='horizontal', projection=None):
        is_bivariate    = self._analysis['is_bivariate']
        is_complex      = self._analysis['is_complex']

        n_rows =  2 if is_bivariate else 1
        n_cols =  3 if is_complex else 2
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
                grid['phase'] = {'left':[0,2]}

            # positions for right field
            if is_bivariate:
                for k, panel in grid.items():
                    yx = panel.get('left')
                    grid[k]['right'] = [yx[0]+1,yx[1]]

            # positions for colorbars
            for k, panel in grid.items():
                if k in ['eof','phase']:
                    row_cb = len(panel)
                    col_cb = panel.get('left')[1]
                    grid[k]['cb'] = [row_cb,col_cb]

        # vertical layout
        if orientation == 'vertical':
            grid = {
                'pc'   : {'left': [-1, 1]},
                'eof'  : {'left': [0, 1]}
                }

            # position for phase
            if is_complex:
                grid['phase'] = {'left':[1,1]}

            # positions for right field
            if is_bivariate:
                for k, panel in grid.items():
                    yx = panel.get('left')
                    grid[k]['right'] = [yx[0], yx[1]+1]

            # positions for colorbars
            for k, panel in grid.items():
                if k in ['eof','phase']:
                    row, col = panel.get('left')
                    grid[k]['cb'] = [row,col-1]

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
        for key_data,data in grid.items():
            axes[key_data] = {}
            for key_pos,pos in data.items():
                row,col = pos
                axes[key_data][key_pos] = fig.add_subplot(gs[row,col], projection=map_projs[key_data][key_pos])


        return fig, axes


    def plot(
        self, mode, threshold=0, phase_shift=0,
        cmap_eof=None, cmap_phase=None, figsize=(8.3,5.0), resolution='110m',
        projection=None, c_lon=None, orientation='horizontal', land=True):
        """
        Plot results for `mode`.

        Parameters
        ----------
        mode : int, optional
            Mode to plot. The default is 1.
        threshold : float, optional
            Amplitude threshold below which the fields are masked out.
            The default is 0.
        cmap_eof : str or Colormap
            The colormap used to map the spatial patterns.
            The default is `Blues`.
        cmap_phase : str or Colormap
            The colormap used to map the spatial phase function.
            The default is `twilight`.
        resolution : {None, '110m', '50m', '10m'}
            A named resolution to use from the Natural Earth dataset.
            Currently can be one of `110m`, `50m`, and `10m`. If None, no
            coastlines will be drawn. Default is `110m`.
        orientation : {'horizontal', 'vertical'}
            Orientation of the plot. Default is horizontal.

        Returns
        -------
        fig :
            Figure instance.
        axes :
            Dictionary of axes containing `pcs`, `eofs` and `phase`, if complex.

        """
        complex     = self._analysis['is_complex']
        bivariate   = self._analysis['is_bivariate']

        # Get data
        var 		= self.explained_variance(mode).sel(mode=mode).values
        pcs         = self.pcs(mode, scaling='max', phase_shift=phase_shift)
        eofs        = self.eofs(mode, scaling='max')
        phases      = self.spatial_phase(mode, phase_shift=phase_shift)
        if complex:
            eofs = self.spatial_amplitude(mode, scaling='max')

        # ticks
        ticks = {
            'pc'       : [-1, 0, 1],
            'eof'      : [0, 1] if complex else [-1,0,1],
            'phase'    : [-np.pi, 0, np.pi]
        }

        # tick labels
        tick_labels = {
            'phase' : [r'-$\pi$','0',r'$\pi$']
        }

        # colormaps
        cmaps = {
            'eof'      : 'Blues' if complex else 'RdBu_r',
            'phase'    : 'twilight'
        }

        if not cmap_eof is None:
            cmaps['eof'] = cmap_eof
        if not cmap_phase is None:
            cmaps['phase'] = cmap_phase

        # titles
        titles = {
            'pc'        : 'PC',
            'eof'       : 'Amplitude' if complex else 'EOF',
            'phase'     : 'Phase',
            'mode'      : 'Mode {:d} ({:.1f} \%)'.format(mode,var)
        }

        for key, name in self._field_names.items():
            titles[key] = name

        titles.update({k: v.replace('_',' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})

        # map projections and boundaries
        map = {
            # center longitude of maps
            'c_lon'     : {'left' : c_lon, 'right' : c_lon},
            # projections pf maps
            'projection' : {
                'left':ccrs.PlateCarree(),
                'right':ccrs.PlateCarree()
                },
            # west, east, south, north limit of maps
            'boundaries' : {'left': None, 'right' : None}
        }
        if projection is not None:
            try:
                map['projection'].update(projection)
            except TypeError:
                map['projection'] = {k: projection for k in map['projection'].keys()}
        data_projection  = ccrs.PlateCarree()

        # plot PCs, EOFs, and Phase
        for key in pcs.keys():
            pcs[key] = pcs[key].sel(mode=mode).real
            eofs[key] = eofs[key].sel(mode=mode)
            phases[key] = phases[key].sel(mode=mode)

            # apply amplitude threshold
            eofs[key]   = eofs[key].where(abs(eofs[key]) >= threshold)
            phases[key] = phases[key].where(abs(eofs[key]) >= threshold)

            # # map projections and center longitude
            # if map['c_lon'][key] is None:
            #     map['c_lon'][key]  = eofs[key].lon[[0,-1]].mean()
            #
            # # if projection is None:
            # #     map['projection'][key]  = ccrs.PlateCarree(central_longitude=map['c_lon'][key])
            # # else:
            # #     map['projection'][key]  = projection(central_longitude=map['c_lon'][key])

            # map boundaries as [east, west, south, north]
            c_lon = map['projection'][key].proj4_params['lon_0']
            map['boundaries'][key] = get_extent(eofs[key], c_lon)


        fig, axes = self._create_gridspec(figsize=figsize, orientation=orientation, projection=map['projection'])

        for i, key in enumerate(pcs.keys()):
            # plot PCs
            pcs[key].plot(ax=axes['pc'][key])
            axes['pc'][key].set_ylim(-1.2,1.2)
            axes['pc'][key].set_yticks([-1,0,1])
            axes['pc'][key].set_ylabel(titles[key], fontweight='bold')
            axes['pc'][key].set_xlabel('')
            axes['pc'][key].set_title('')
            axes['pc'][key].spines['right'].set_visible(False)
            axes['pc'][key].spines['top'].set_visible(False)

            # plot EOFs
            cb_eof = eofs[key].plot(
                ax=axes['eof'][key], transform=data_projection,
                vmin=ticks['eof'][0], vmax=ticks['eof'][-1], cmap=cmaps['eof'],
                add_colorbar = False)
            axes['eof'][key].set_extent(map['boundaries'][key], crs=data_projection)
            axes['eof'][key].set_title('')

            if resolution in ['110m','50m','10m']:
                axes['eof'][key].coastlines(lw = .4, resolution = resolution)
            if land:
                axes['eof'][key].add_feature(cfeature.LAND, color='#808080', zorder=0)
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
                    cmap=cmaps['phase'], add_colorbar = False)
                axes['phase'][key].set_extent(map['boundaries'][key], crs=data_projection)
                axes['phase'][key].set_title('')

                plt.colorbar(cb_phase, axes['phase']['cb'], orientation=orientation)
                if orientation=='horizontal':
                    axes['phase']['cb'].xaxis.set_ticks(ticks['phase'])
                    axes['phase']['cb'].set_xticklabels(tick_labels['phase'])
                elif orientation == 'vertical':
                    axes['phase']['cb'].yaxis.set_ticks(ticks['phase'])
                    axes['phase']['cb'].set_yticklabels(tick_labels['phase'])

                if resolution in ['110m','50m','10m']:
                    axes['phase'][key].coastlines(lw = .4, resolution = resolution)
                if land:
                    axes['phase'][key].add_feature(cfeature.LAND, color='#808080', zorder=0)
                axes['phase'][key].set_aspect('auto')
                axes['phase']['left'].set_title(titles['phase'], fontweight='bold')


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
                axes['eof']['right'].set_title(titles['right'], fontweight='bold')

            if complex:
                axes['phase']['cb'].set_ylabel(titles['phase'], fontweight='bold')
                axes['phase']['left'].set_title('')
                axes['phase']['cb'].yaxis.set_label_position('left')
                axes['phase']['cb'].yaxis.set_ticks_position('left')



        fig.subplots_adjust(wspace=.1)
        fig.suptitle(titles['mode'], horizontalalignment='left')

        return fig, axes


    def _save_data(self, data_array, path, engine='h5netcdf', *args, **kwargs):
        analysis_path   = path
        analysis_name   = self._get_analysis_id()
        var_name        = secure_str('.'.join([data_array.name,'nc']))

        file_name   = '_'.join([analysis_name, var_name])
        output_path = os.path.join(analysis_path,file_name)

        invalid_netcdf = True
        if engine != 'h5netcdf':
            invalid_netcdf = False
        data_array.to_netcdf(
            path=output_path,
            engine=engine, invalid_netcdf=invalid_netcdf, *args, **kwargs
            )


    def save_analysis(self, path=None, engine='h5netcdf'):
        analysis_path = self._get_analysis_path(path)

        self._create_info_file(analysis_path)

        fields      = self._get_fields(original_scale = True)
        eofs        = self.eofs()
        pcs         = self.pcs()
        singular_values = self.singular_values()

        self._save_data(singular_values, analysis_path, engine)
        for key in pcs.keys():
            self._save_data(fields[key], analysis_path, engine)
            self._save_data(eofs[key], analysis_path, engine)
            self._save_data(pcs[key], analysis_path, engine)



    def load_analysis(self, path, engine='h5netcdf'):
        self._set_info_from_file(path)
        path_folder,_ = os.path.split(path)
        file_names = self._get_file_names(format='nc')

        path_eigen   = os.path.join(path_folder,file_names['singular'])
        singular_values = xr.open_dataarray(path_eigen, engine = engine).data

        fields  = {}
        pcs     = {}
        eofs    = {}
        for key in self._field_names.keys():
            path_fields   = os.path.join(path_folder,file_names['fields'][key])
            path_pcs   = os.path.join(path_folder,file_names['pcs'][key])
            path_eofs   = os.path.join(path_folder,file_names['eofs'][key])

            fields[key] = xr.open_dataarray(path_fields, engine = engine)
            pcs[key]    = xr.open_dataarray(path_pcs, engine = engine)
            eofs[key]   = xr.open_dataarray(path_eofs, engine = engine)


        self._field_coords = {}
        for key, field in fields.items():
            self._field_coords[key] = field.coords
            self._field_dims[key]   = field.dims

            fields[key]     = fields[key].data
            eofs[key]       = eofs[key].data
            pcs[key]        = pcs[key].data


        super().load_analysis(
            path = path,
            fields = fields,
            eofs = eofs,
            pcs  = pcs,
            singular_values = singular_values)

        if self._analysis['is_coslat_corrected']:
            self.apply_coslat()
