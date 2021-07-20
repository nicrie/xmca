#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for xarray.DataArray modifications. '''

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr

# =============================================================================
# Tools
# =============================================================================

def is_DataArray(data):
    '''Check if data is of type `xr.DataArray`.

    Parameters
    ----------
    A : DataArray
        Input data.

    Returns
    -------
    bool
        Input data is of type `DataArray`.

    '''
    if(isinstance(data,xr.DataArray)):
        pass
    else:
        raise TypeError("Data format has to be xarray.DatArray.")


def check_dims(da1, da2):
    if (da1.dims == da2.dims):
        pass
    else:
        raise ValueError("Dimensions of input data has to be the same.")


def get_attr(data_array, attr, fallback='undefined'):
    try:
        return data_array.attrs[attr]
    except KeyError:
        return fallback


def calc_temporal_corr(x, y):
    is_DataArray(x)
    is_DataArray(y)

    x = x - x.mean('time')
    y = y - y.mean('time')

    xy = xr.dot(x * y, dims='time') / x.shape[0]
    sigx = x.std('time')
    sigy = y.std('time')
    corr_coef = xy / sigx / sigy
    return corr_coef


def wrap_lon_to_180(da, lon='lon'):
    '''
    Wrap longitude coordinates of DatArray to -180..179

    Parameters
    ----------
    da : DatArray
        object with longitude coordinates
    lon : string
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    '''

    # wrap 0..359 to -180..179
    da = da.assign_coords(lon=(((da[lon] + 180) % 360) - 180))

    # sort the data
    return da.sortby(lon)


def get_extent(data_array, central_longitude=0):
    try:
        data_array = wrap_lon_to_180(data_array)
        east 	= data_array.coords['lon'].min() + central_longitude + 0.001
        west 	= data_array.coords['lon'].max() + central_longitude - 0.001
        south 	= data_array.coords['lat'].min()
        north 	= data_array.coords['lat'].max()

        return [east, west, south, north]

    except KeyError:
        KeyError("Spatial coordinates need to be called `lon` and `lat`.")


def norm_space_to_1(data_array):
    try:
        return data_array / abs(data_array).max(['lon','lat'])
    except KeyError:
        KeyError("Spatial coordinates need to be called `lon` and `lat`.")


def norm_time_to_1(data_array):
    try:
        return data_array / abs(data_array).max(['time'])
    except KeyError:
        KeyError("Temporal coordinate needs to be called `time`.")


def split_complex(data_array):
    dataset = xr.Dataset(
        data_vars = {
            'real': data_array.real,
            'imag': data_array.imag},
        attrs = data_array.attrs)

    return dataset


def create_coords(*coords):
    dims = [c.name for c in coords]
    sizes = [c.size for c in coords]

    data_array = np.random.randint(1,size=sizes)
    data_array = xr.DataArray(
        data_array,
        dims = dims,
        coords = coords)

    return data_array.coords
