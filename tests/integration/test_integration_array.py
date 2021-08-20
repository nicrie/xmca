import contextlib
import os
import unittest
import warnings
from os import getcwd
from os.path import join
from shutil import rmtree

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_raises
from parameterized import parameterized

from xmca.array import MCA


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # catch deprecation warnings from cartopy
        warnings.simplefilter('ignore', category=DeprecationWarning)

        # Load test data
        self.path = 'tests/integration/fixtures'
        # ignore some deprecation warnings of xarray
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.A = xr.open_dataarray(join(self.path, 'sst.nc')).data
            self.B = xr.open_dataarray(join(self.path, 'prcp.nc')).data

        # how many modes to compare
        self.modes = 100
        # acceptable tolerance for comparison
        self.tols = {'atol': 1e-3, 'rtol': 1e-3}

        self.files = {
            'svalues' : 'singular_values.nc',
            'V1' : 'sst_eofs.nc',
            'V2' : 'prcp_eofs.nc',
        }

    def name_func_get(testcase_func, param_num, param):
        return "{:}_{:s}".format(
            testcase_func.__name__,
            parameterized.to_safe_name('_'.join([str(a) for a in param.args])),
        )

    @parameterized.expand([
        ('uni', 'std', 1),
        ('uni', 'cplx', 2),
        ('uni', 'varmx', 3),
        ('bi', 'std', 1),
        ('bi', 'cplx', 2),
        ('bi', 'varmx', 3),
    ], name_func=name_func_get)
    def test_plot(self, analysis, flavour, n):
        cplx = False,
        n_rot = 0
        if flavour == 'cplx':
            cplx = True
        if flavour == 'varmx':
            n_rot = 10
        if analysis == 'uni':
            model = MCA(self.A)
        elif analysis == 'bi':
            model = MCA(self.A, self.B)

        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)
        model.plot(n)

    @classmethod
    def tearDownClass(self):
        pass
