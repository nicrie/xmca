import unittest

import numpy as np
import xarray as xr

from xmca.xarray import xMCA

np.random.seed(777)

class TestXarray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        A = np.random.rand(500,20,15)
        B = np.random.rand(500,15,10)
        C = np.random.rand(500,10,5)
        self.A = xr.DataArray(A, dims=['time','lat','lon'])
        self.B = xr.DataArray(B, dims=['time','lat','lon'])
        self.C = xr.DataArray(C, dims=['time','lat','lon'])

        self.rank = np.min([np.product(self.A.shape[1:]), np.product(self.B.shape[1:])])

    def test_xmca_input(self):
        xMCA()
        xMCA(self.A)
        xMCA(self.A, self.B)
        with self.assertRaises(ValueError):
            xMCA(self.A, self.B, self.A)

        with self.assertRaises(TypeError):
            xMCA(np.array([1,2,3]))

    @classmethod
    def tearDownClass(self):
        pass
