#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of tools for xarray.DataArray modifications. """

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr

# =============================================================================
# Tools
# =============================================================================

def is_DataArray(data):
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

    xy = (x*y).mean('time')
    sigx = x.std('time')
    sigy = y.std('time')

    return xy/sigx/sigy


def get_lonlat_limits(data_array):
    try:
        east 	= data_array.coords['lon'].min()
        west 	= data_array.coords['lon'].max()
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


def array_to_set(data_array):
    if data_array.dtype == np.complex:
        dataset = split_complex(data_array)
    else:
        dataset = data_array.to_dataset(promote_attrs=True)
    return dataset


def set_to_array(dataset):
    if isinstance(dataset,xr.DataArray):
        data_array = dataset
    else:
        try:
            data_array =  dataset['real'] + 1j * dataset['imag']
            data_array.attrs = dataset.attrs
        except KeyError:
            raise KeyError('xr.Dataset needs two variables called `real` and `imag`.')
        except TypeError:
            raise TypeError("Input type needs to be `xr.DataArray` or `xr.Dataset`")

    return data_array


def create_coords(*coords):
    dims = [c.name for c in coords]
    sizes = [c.size for c in coords]

    data_array = np.random.randint(1,size=sizes)
    data_array = xr.DataArray(
        data_array,
        dims = dims,
        coords = coords)

    return data_array.coords
