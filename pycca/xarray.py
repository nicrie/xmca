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
from datetime import datetime
import cmath

from pycca.array import CCA
from tools.text import secure_str, boldify_str, wrap_str
from tools.xarray import is_DataArray, check_dims, get_attr, calc_temporal_corr
from tools.xarray import get_extent, norm_time_to_1, norm_space_to_1
from tools.xarray import split_complex, create_coords
# =============================================================================
# xCCA
# =============================================================================

class xCCA(CCA):
    """Perform Canonical Correlation Analysis (CCA) for two `xarray.DataArray`.

    CCA is a generalized form of Principal Component Analysis (PCA)
    for two input fields (left, right). If both data fields are the same,
    it is equivalent to PCA. Non-normalized CCA is called Maximum Covariance
    Analysis (MCA).

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
        CCA.__init__(self, *fields.values())


    def _get_fields(self, original_scale=False):
        dims        = self._field_dims
        coords      = self._field_coords
        field_names = self._field_names
        fields      = CCA._get_fields(self, original_scale=original_scale)

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
            weights[key] = np.cos(np.deg2rad(coord['lat']))

        if (not self._analysis['is_coslat_corrected']):
            self.apply_weights(**weights)
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
        values = CCA.eigenvalues(self, n)

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

        # error = xr.DataArray(error,
        #     dims 	= ['mode'],
        #     coords 	= {'mode' : modes},
        #     name 	= 'error eigenvalues',
        #     attrs   = attrs)

        return values


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
        variance 	= CCA.explained_variance(self, n)

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

        # error = xr.DataArray(error,
        #     dims 	= ['mode'],
        #     coords 	= {'mode' : modes},
        #     name 	= 'error explained variance',
        #     attrs   = attrs)

        return variance


    def pcs(self, n=None, scaling=0, phase_shift=0):
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
        pcs = CCA.pcs(self, n, scaling, phase_shift)

        if n is None:
            n = self._eigenvalues.size

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


    def eofs(self, n=None, scaling=0, phase_shift=0):
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
        eofs = CCA.eofs(self, n, scaling, phase_shift)

        if n is None:
            n = self._eigenvalues.size

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
        eofs = self.eofs(n)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names
        amplitudes = {}
        for key, eof in eofs.items():
            amplitudes[key]         = np.sqrt(eof * eof.conjugate()).real
            amplitudes[key].name    = ' '.join([field_names[key],'spatial amplitude'])
            amplitudes[key].attrs   = attrs

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
        pcs = self.pcs(n)

        attrs = {k: str(v) for k, v in self._analysis.items()}
        field_names = self._field_names
        amplitudes = {}
        for key,pc in pcs.items():
            amplitudes[key]         = np.sqrt(pc * pc.conjugate()).real
            name = ' '.join([field_names[key],'temporal amplitude'])
            amplitudes[key].name    = name
            amplitudes[key].attrs   = attrs

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
        eofs    = self.eofs(scaling=0)
        pcs     = self.pcs(scaling=1)
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
                rec_fields[key] /= np.cos(np.deg2rad(coords[key]['lat']))

            if self._analysis['is_normalized']:
                rec_fields[key] *= std[key]

            # add mean fields
            rec_fields[key]  += mean[key]

        return rec_fields


    def plot(
        self, mode, threshold=0, phase_shift=0,
        cmap_eof=None, cmap_phase=None, figsize=(8.3,5.0),
        resolution='110m', projection=None, c_lon=None):
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
        resolution : a named resolution to use from the Natural Earth dataset.
            Currently can be one of '110m', '50m', and '10m'.

        Returns
        -------
        None.

        """

        pcs         = self.pcs(mode, phase_shift=phase_shift)
        eofs        = self.eofs(mode)
        phases      = self.spatial_phase(mode, phase_shift=phase_shift)
        var 		= self.explained_variance(mode)
        var 		= var.sel(mode=mode).values
        cmap_eof_range  = [-1, 0, 1]
        eof_title   = 'EOF'

        n_cols          = 2
        n_rows          = len(pcs)
        height_ratios   = [1] * n_rows

        # add additional row for colorbar
        n_rows += 1
        height_ratios.append(0.05)

        if self._analysis['is_complex']:
            n_cols          += 1
            eofs            = self.spatial_amplitude(mode)
            cmap_eof_range  = [0, 1]
            cmap_eof        = 'Blues' if cmap_eof is None else cmap_eof
            cmap_phase      = 'twilight' if cmap_phase is None else cmap_phase
            eof_title       = 'Amplitude'
        else:
            cmap_eof        = 'RdBu_r' if cmap_eof is None else cmap_eof

        titles = {
        'pc'    : 'PC {:d} ({:.1f} \%)'.format(mode,var),
        'eof'   : eof_title,
        'phase' :'Phase'
        }
        for key, name in self._field_names.items():
            titles[key] = name

        titles.update({k: v.replace('_',' ') for k, v in titles.items()})
        titles.update({k: boldify_str(v) for k, v in titles.items()})

        map_c_lons      = {} # center longitude of maps
        map_projs       = {} # projections for maps
        map_boundaries  = {} # west, east, south, north limit of maps

        for key in pcs.keys():
            pcs[key] = pcs[key].sel(mode=mode).real
            eofs[key] = eofs[key].sel(mode=mode)
            phases[key] = phases[key].sel(mode=mode)

            # normalize all EOFs/PCs such that they range from -1...+1
            eofs[key]   = norm_space_to_1(eofs[key])
            pcs[key]    = norm_time_to_1(pcs[key])

            # apply amplitude threshold
            eofs[key]   = eofs[key].where(abs(eofs[key]) >= threshold)
            phases[key] = phases[key].where(abs(eofs[key]) >= threshold)

            # map projections and center longitude
            data_projection  = ccrs.PlateCarree()
            if c_lon is None:
                map_c_lons[key]  = eofs[key].lon[[0,-1]].mean()
            else:
                map_c_lons[key]  = c_lon

            if projection is None:
                projection  = ccrs.PlateCarree
            map_projs[key]  = projection(central_longitude=map_c_lons[key])

            # map boundaries as [east, west, south, north]
            map_boundaries[key] = get_extent(eofs[key], map_c_lons[key])

        # create figure environment
        fig = plt.figure(figsize=figsize, dpi=150)
        fig.subplots_adjust(hspace=0.1, wspace=.1, left=0.25)
        gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios)
        axes_pc     = {}
        axes_eof    = {}
        axes_phase  = {}
        cbax_eof    = fig.add_subplot(gs[-1,1])

        for i, key in enumerate(pcs.keys()):
            axes_pc[key]    = fig.add_subplot(gs[i,0])
            axes_eof[key]   = fig.add_subplot(gs[i,1], projection=map_projs[key])

            # plot PCs
            pcs[key].plot(ax=axes_pc[key])
            axes_pc[key].set_ylim(-1.2,1.2)
            axes_pc[key].set_yticks([-1,0,1])
            axes_pc[key].set_ylabel(titles[key], fontweight='bold')
            axes_pc[key].set_xlabel('')
            axes_pc[key].set_title('')
            axes_pc[key].spines['right'].set_visible(False)
            axes_pc[key].spines['top'].set_visible(False)

            # plot EOFs
            cb_eof = eofs[key].plot(
                ax=axes_eof[key], transform=data_projection,
                vmin=cmap_eof_range[0], vmax=cmap_eof_range[-1], cmap=cmap_eof,
                add_colorbar = False)
            axes_eof[key].set_extent(map_boundaries[key], crs=data_projection)
            axes_eof[key].set_title('')

            plt.colorbar(cb_eof, cbax_eof, orientation='horizontal')
            cbax_eof.xaxis.set_ticks(cmap_eof_range)

            axes_eof[key].coastlines(lw = .5, resolution = resolution)
            axes_eof[key].set_aspect('auto')
            axes_eof[key].add_feature(cfeature.LAND, color='gray', zorder=0)

            # plot Phase function (if data is complex)
            if (self._analysis['is_complex']):
                axes_phase[key] = fig.add_subplot(gs[i,2], projection=map_projs[key])
                cbax_phase = fig.add_subplot(gs[-1,2])

                # plot Phase
                cb_phase = phases[key].plot(
                    ax=axes_phase[key], transform=data_projection,
                    vmin=-np.pi, vmax=np.pi, cmap=cmap_phase,
                    add_colorbar = False)
                axes_phase[key].set_extent(map_boundaries[key], crs=data_projection)
                axes_phase[key].set_title('')

                plt.colorbar(cb_phase, cbax_phase, orientation='horizontal')
                cbax_phase.xaxis.set_ticks([-3.14,0,3.14])
                cbax_phase.set_xticklabels([r'-$\pi$','0',r'$\pi$'])

                axes_phase[key].coastlines(lw = .5, resolution = resolution)
                axes_phase[key].set_aspect('auto')
                axes_phase[key].add_feature(cfeature.LAND, color='gray', zorder=0)
                axes_phase['left'].set_title(titles['phase'], fontweight='bold')


        # if more than 1 row, remove xaxis
        if (len(pcs) == 2):
            axes_pc['left'].xaxis.set_visible(False)
            axes_pc['left'].spines['bottom'].set_visible(False)

        axes_pc['left'].set_title(titles['pc'], fontweight='bold')
        axes_eof['left'].set_title(titles['eof'], fontweight='bold')


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

        fields = self._get_fields(original_scale = True)
        eofs = self.eofs()
        pcs = self.pcs()
        eigenvalues = self.eigenvalues()

        self._save_data(eigenvalues, analysis_path, engine)
        for key in pcs.keys():
            self._save_data(fields[key], analysis_path, engine)
            self._save_data(eofs[key], analysis_path, engine)
            self._save_data(pcs[key], analysis_path, engine)



    def load_analysis(self, path, engine='h5netcdf'):
        self._set_info_from_file(path)
        path_folder,_ = os.path.split(path)
        file_names = self._get_file_names(format='nc')

        path_eigen   = os.path.join(path_folder,file_names['eigenvalues'])
        eigenvalues = xr.open_dataarray(path_eigen, engine = engine).data

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

            fields[key]     = fields[key].data
            eofs[key]       = eofs[key].data
            pcs[key]        = pcs[key].data


        CCA.load_analysis(
            self,
            path = path,
            fields = fields,
            eofs = eofs,
            pcs  = pcs,
            eigenvalues = eigenvalues)
