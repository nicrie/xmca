#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for numpy.array modifications. '''

import warnings

import numpy as np


# =============================================================================
# Tools
# =============================================================================
def remove_mean(arr):
    '''Remove the mean of an array along the first dimension.

    If a variable (column) has at least 1 errorneous observation (row)
    the entire column will be set to NaN.

    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return arr - arr.mean(axis=0)


def get_nan_cols(arr: np.ndarray) -> np.ndarray:
    '''Get NaN columns from an array.

    Parameters
    ----------
    arr : ndarray
        Array to be scanned.

    Returns
    -------
    index : 1darray
        Index of columns with NaN entries from original data

    '''
    nan_index = np.isnan(arr).any(axis=0)

    return nan_index


def remove_nan_cols(arr: np.ndarray) -> np.ndarray:
    '''Remove NaN columns in array.

    Parameters
    ----------
    arr : ndarray
        Array to be cleaned.

    Returns
    -------
    new_data : ndarray
        Array without NaN columns.

    '''
    idx = get_nan_cols(arr)
    new_arr  = arr[:, ~idx]

    return new_arr


def has_nan_time_steps(array):
    ''' Checks if an array has NaN time steps.

    Time is assumed to be on axis=0. The array is then reshaped to be 2D with
    time along axis 0 and variables along axis 1. A NaN time step is a row
    which contain NaN only.
    '''

    return (np.isnan(array).all(axis=tuple(range(1, array.ndim))).any())


def block_permutations(
        arr: np.ndarray, block_size: int, replace: bool = True) -> np.ndarray:
    n_obs, n_vars = arr.shape
    try:
        block_arr = arr.reshape(-1, block_size, arr.shape[1])
    except ValueError as err:
        msg = 'Length of data array ({:}) must be a multiple of block size {:}'
        msg = msg.format(n_obs, block_size)
        raise ValueError(msg) from err
    n_samples = block_arr.shape[0]
    idx_samples = np.random.choice(n_samples, size=n_samples, replace=replace)
    samples = block_arr[idx_samples]
    new_arr = samples.reshape(arr.shape)
    return new_arr
