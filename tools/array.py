#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Collection of tools for numpy.array modifications. """

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import warnings
import textwrap

# =============================================================================
# Tools
# =============================================================================

def arrs_are_equal(arr1 ,arr2):
    """ True if arrays are the same. Also works for np.nan entries."""
    if arr1.shape == arr2.shape:
        return ((np.isnan(arr1) & np.isnan(arr2)) | (arr1 == arr2)).all()
    else:
        return False


def is_arr(data):
    if (isinstance(data,np.ndarray)):
        return True
    else:
        raise TypeError('Data needs to be np.ndarray.')


def check_time_dims(arr1, arr2):
    if (arr1.shape[0] == arr2.shape[0]):
        pass
    else:
        raise ValueError('Both input fields need to have same time dimensions.')


def remove_mean(arr):
    """Remove the mean of an array along the first dimension.

    If a variable (column) has at least 1 errorneous observation (row)
    the entire column will be set to NaN.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return arr - arr.mean(axis=0)


def remove_nan_cols(arr):
    """Remove NaN columns in array.

    Parameters
    ----------
    arr : ndarray
        Array to be cleaned.

    Returns
    -------
    new_data : ndarray
        Array without NaN columns.
    index : 1darray
        Index of columns without NaN entries from original data

    """
    " Remove all columns where the first row contains"
    index = np.where(~(np.isnan(arr[0])))[0]
    new_arr  = arr[:,index]

    return new_arr, index


def check_nan_rows(arr):
    """Check if `arr` contains at least one row filled with NaN.

    A NaN row is problematic since it will lead to only NaN data when removing
    the mean via `np.mean()`.
    """
    if (np.isnan(arr).all(axis=1).any()):
        raise ValueError(textwrap.fill(textwrap.dedent("""
        Gaps (np.NaN) in time series detected. Either remove or interpolate
        all NaN time steps in your data.""")))
    else:
        pass


def is_not_empty(arr):
    if (arr.size > 0):
        pass
    else:
        raise ValueError('Input field is empty or contains NaN only.')


def norm_to_1(arr, axis):
    return arr / np.nanmax(abs(arr), axis=axis)
