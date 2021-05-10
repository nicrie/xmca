#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for numpy.array modifications. '''

import warnings

import numpy as np
try:
    import dask.array as da
    dask_support = True
except ImportError:
    dask_support = False

# =============================================================================
# Tools
# =============================================================================

if dask_support:
    def dask_hilbert(x, N=None, axis=-1):
        x = da.asarray(x)
        if da.iscomplex(x).any():
            raise ValueError('x must be real.')

        if N is None:
            N = x.shape[axis]
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf = da.fft.fft(x, N, axis=axis)
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        if x.ndim > 1:
            ind = [np.newaxis] * x.ndim
            ind[axis] = slice(None)
            h = h[tuple(ind)]
        # maybe add dask support for h
        h = da.from_array(h)
        x = da.fft.ifft(Xf * h, axis=axis)
        return x


def hilbert(x, N=None, axis=-1):
    x = np.asarray(x)
    if np.iscomplex(x).any():
        raise ValueError('x must be real.')

    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft.fft(x, N, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = np.fft.ifft(Xf * h, axis=axis)
    return x


def arrs_are_equal(arr1, arr2):
    ''' True if arrays are the same. Also works for np.nan entries.'''
    if arr1.shape == arr2.shape:
        return ((np.isnan(arr1) & np.isnan(arr2)) | (arr1 == arr2)).all()
    else:
        return False


def is_arr(data):
    if (isinstance(data, np.ndarray)):
        return True
    else:
        raise TypeError('Data needs to be np.ndarray.')


def check_time_dims(arr1, arr2):
    if (arr1.shape[0] == arr2.shape[0]):
        pass
    else:
        raise ValueError('Both input fields need to have same time dimensions.')


def remove_mean(arr):
    '''Remove the mean of an array along the first dimension.

    If a variable (column) has at least 1 errorneous observation (row)
    the entire column will be set to NaN.

    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return arr - arr.mean(axis=0)


def remove_nan_cols(arr):
    '''Remove NaN columns in array.

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

    '''
    " Remove all columns where the first row contains"
    if isinstance(arr, np.ndarray):
        index = np.where(~(np.isnan(arr[0])))[0]
    elif isinstance(arr, da.Array):
        index = da.where(~(da.isnan(arr[0])))[0]
        # dask slicing using another dask does not work; first index has
        # to be computed
        index = index.compute()
    else:
        raise TypeError('Must be either `np.ndarray` or `dask.array.Array`')
    new_arr  = arr[:, index]
    return new_arr, index


def has_nan_time_steps(array):
    ''' Checks if an array has NaN time steps.

    Time is assumed to be on axis=0. The array is then reshaped to be 2D with
    time along axis 0 and variables along axis 1. A NaN time step is a row which
    contain NaN only.
    '''

    return (np.isnan(array).all(axis=tuple(range(1, array.ndim))).any())


def is_not_empty(arr):
    if (arr.size > 0):
        pass
    else:
        raise ValueError('Input field is empty or contains NaN only.')


def norm_to_1(arr, axis):
    return arr / np.nanmax(abs(arr), axis=axis)
