#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for xarray.DataArray modifications. '''

# =============================================================================
# Imports
# =============================================================================
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
    if not (isinstance(data, xr.DataArray)):
        raise TypeError("Data format has to be xarray.DatArray.")


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
        east = data_array.coords['lon'].min() + central_longitude + 0.001
        west = data_array.coords['lon'].max() + central_longitude - 0.001
        south = data_array.coords['lat'].min()
        north = data_array.coords['lat'].max()

        return [east, west, south, north]

    except KeyError:
        KeyError("Spatial coordinates need to be called `lon` and `lat`.")
