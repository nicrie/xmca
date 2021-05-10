import unittest

import numpy as np
import xarray as xr

from xmca.xarray import xMCA

try:
    import dask.array
    dask_support = True
except ImportError:
    dask_support = False


class TestXarray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(7)
        A = np.random.rand(500, 20, 15)
        np.random.seed(8)
        B = np.random.rand(500, 15, 10)
        np.random.seed(9)
        C = np.random.rand(500, 10, 5)
        self.A = xr.DataArray(A, dims=['time', 'lat', 'lon'])
        self.B = xr.DataArray(B, dims=['time', 'lat', 'lon'])
        self.C = xr.DataArray(C, dims=['time', 'lat', 'lon'])

        n_var_A = np.product(self.A.shape[1:])
        n_var_B = np.product(self.B.shape[1:])
        self.rank = np.min([n_var_A, n_var_B])

    def test_xmca_input(self):
        xMCA()
        xMCA(self.A)
        xMCA(self.A, self.B)
        with self.assertRaises(ValueError):
            xMCA(self.A, self.B, self.A)

        with self.assertRaises(TypeError):
            xMCA(np.array([1, 2, 3]))

        if dask_support:
            temp = xr.tutorial.open_dataset(
                'air_temperature',
                chunks={'lat': 25, 'lon': 25, 'time': -1}
            )
            temp = temp.air
            xMCA(temp)
            xMCA(temp, temp)

    @classmethod
    def tearDownClass(self):
        pass
