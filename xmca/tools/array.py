#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of tools for numpy.array modifications. '''

import warnings

import numpy as np
import scipy.stats


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


def pearsonr(x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError('Time dimensions are different.')
    n = x.shape[0]

    r = np.corrcoef(x, y, rowvar=False)
    r = r[:x.shape[1], x.shape[1]:]

    # get p-values
    dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
    p = 2 * dist.cdf(-abs(r))

    return r, p


def block_bootstrap(
        arr: np.ndarray,
        axis : int = 0,
        block_size: int = 1,
        replace: bool = True) -> np.ndarray:
    '''Perform (moving-block) bootstrapping on a 2darray.

    Parameters
    ----------
    arr : np.ndarray
        Array to perform bootstrapping on. Must be 2d.
    axis : int
        Axis on which to bootstrap on. The default is 0.
    block_size : int
        Block size to keep intact when bootstrapping. By default 1.
    replace : bool
        Whether to resample with replacement (bootstrapping) or without
        (permutation). By default with replacement.

    Returns
    -------
    np.ndarray
        Resampled array.

    '''
    if axis == 0:
        pass
    elif axis == 1:
        arr = arr.T
    else:
        msg = '{:} not a valid axis. either 0 or 1.'.format(axis)
        raise ValueError(msg)

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

    if axis == 1:
        new_arr = new_arr.T
    return new_arr
