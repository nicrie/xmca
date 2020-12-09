#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of tools for xarray.DataArray modifications. """

# =============================================================================
# Imports
# =============================================================================
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


def get_attr(da, attr, fallback='undefined'):
    try:
        return da.attrs[attr]
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


def get_lonlat_limits(da):
    try:
        east 	= da.coords['lon'].min()
        west 	= da.coords['lon'].max()
        south 	= da.coords['lat'].min()
        north 	= da.coords['lat'].max()

        return [east, west, south, north]

    except KeyError:
        KeyError("Spatial coordinates need to be called `lon` and `lat`.")


def norm_space_to_1(da):
    try:
        return da / abs(da).max(['lon','lat'])
    except KeyError:
        KeyError("Spatial coordinates need to be called `lon` and `lat`.")


def norm_time_to_1(da):
    try:
        return da / abs(da).max(['time'])
    except KeyError:
        KeyError("Temporal coordinate needs to be called `time`.")

def split_complex(da):
    ds = xr.Dataset(
        data_vars = {
            'real': da.real,
            'imag': da.imag},
        attrs = da.attrs)

    return ds

def Dataset_to_DataArray(dataset):
    if is_DataArray(dataset):
        return dataset
    else:
        try:
            da =  dataset['real'] + 1j * dataset['imag']
        except KeyError:
            raise KeyError('xr.Dataset needs two variables called `real` and `imag`.')
        except TypeError:
            raise TypeError("Input type needs to be `xr.DataArray` or `xr.Dataset`")

        da.attrs = dataset.attrs
        return da
