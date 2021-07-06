import unittest
import warnings
from os import getcwd
from os.path import join
from shutil import rmtree

import numpy as np
import xarray as xr
from parameterized import parameterized
from numpy.testing import assert_allclose, assert_raises

from xmca.xarray import xMCA


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Load test data
        self.path = 'tests/integration/fixtures'
        # ignore some deprecation warnings of xarray
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.A = xr.open_dataarray(join(self.path, 'sst.nc'))
            self.B = xr.open_dataarray(join(self.path, 'prcp.nc'))

        # how many modes to compare
        self.modes = 100
        # acceptable tolerance for comparison
        self.tols = {'atol': 1e-5, 'rtol': 1e-5}

        self.files = {
            'svalues' : 'singular_values.nc',
            'V1' : 'sst_eofs.nc',
            'V2' : 'prcp_eofs.nc',
        }

    def name_func(testcase_func, param_num, param):
        return "{:}_{:s}".format(
            testcase_func.__name__,
            parameterized.to_safe_name('_'.join([param.args[0]])),
        )

    @parameterized.expand([
        ('std', False, 0),
        ('rot', False, 10),
        ('cplx', True, 0),
    ], name_func=name_func)
    def test_perform_svd(self, analysis, cplx, n_rot):
        path = join(self.path, analysis)
        files = self.files
        n = self.modes

        svalues = xr.open_dataarray(
            join(path, files['svalues'])
        )[:n]
        V1 = xr.open_dataarray(
            join(path, files['V1']),
            engine='h5netcdf'
        )[..., :n]
        V2 = xr.open_dataarray(
            join(path, files['V2']),
            engine='h5netcdf'
        )[..., :n]

        model = xMCA(self.A, self.B)
        model.set_field_names('sst', 'prcp')
        model.solve(complexify=cplx)
        vals = model.singular_values(n)
        eofs = model.eofs(n)
        # fields = mca.reconstructed_fields()
        assert_allclose(
            svalues, vals, err_msg='svalues do not match', **self.tols
        )
        assert_allclose(
            V1, eofs['left'], err_msg='eofs A do not match', **self.tols
        )
        assert_allclose(
            V2, eofs['right'], err_msg='eofs B do not match', **self.tols
        )

    @parameterized.expand([
        ('std', False, 0),
        ('rot', False, 10),
        ('cplx', True, 0),
    ], name_func=name_func)
    def test_save_load(self, analysis, cplx, n_rot):
        path = join(self.path, analysis)
        files = self.files
        n = self.modes

        svalues = xr.open_dataarray(
            join(path, files['svalues'])
        )[:n]
        V1 = xr.open_dataarray(
            join(path, files['V1']),
            engine='h5netcdf'
        )[..., :n]
        V2 = xr.open_dataarray(
            join(path, files['V2']),
            engine='h5netcdf'
        )[..., :n]

        model = xMCA(self.A, self.B)
        model.set_field_names('sst', 'prcp')
        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)
        model.save_analysis('tests/integration/temp')
        new = xMCA()
        new.load_analysis('tests/integration/temp/info.xmca')
        vals = new.singular_values(n)
        eofs = new.eofs(n, original=True)
        assert_allclose(
            svalues, vals, err_msg='svalues do not match', **self.tols
        )
        assert_allclose(
            V1, eofs['left'], err_msg='eofs A do not match', **self.tols
        )
        assert_allclose(
            V2, eofs['right'], err_msg='eofs B do not match', **self.tols
        )
        rmtree(join(getcwd(), 'tests/integration/temp/'))

    @parameterized.expand([
        ('std', False, False, 0, 1),
        ('cplx', False, True, 0, 1),
        ('varmx', False, False, 10, 1),
        ('cplx_varmx', False, True, 10, 1),
        ('promx', False, False, 10, 4),
        ('cplx_promx', False, True, 10, 4),
        ('std', True, False, 0, 1),
        ('cplx', True, True, 0, 1),
        ('varmx', True, False, 10, 1),
        ('cplx_varmx', True, True, 10, 1),
        ('promx', False, False, 10, 4),
        ('cplx_promx', True, True, 10, 4)
    ], name_func=name_func)
    def test_orthogonality(self, analysis, norm, cplx, n_rot, power):
        model = xMCA(self.A, self.B)
        model.set_field_names('sst', 'prcp')
        if norm:
            model.normalize()
        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)
        V = model._get_V()
        for k, v in V.items():
            result = (v.conjugate().T @ v).real
            expected = np.eye(v.shape[1])
            if not model._analysis['is_rotated']:
                assert_allclose(result, expected, **self.tols)
            else:
                assert_raises(AssertionError, assert_allclose, result, expected)

    @parameterized.expand([
        ('std', False, False, 0, 1),
        ('cplx', False, True, 0, 1),
        ('varmx', False, False, 10, 1),
        ('cplx_varmx', False, True, 10, 1),
        ('promx', False, False, 10, 4),
        ('cplx_promx', False, True, 10, 4),
        ('std', True, False, 0, 1),
        ('cplx', True, True, 0, 1),
        ('varmx', True, False, 10, 1),
        ('cplx_varmx', True, True, 10, 1),
        ('promx', False, False, 10, 4),
        ('cplx_promx', True, True, 10, 4)
    ], name_func=name_func)
    def test_correlation(self, analysis, norm, cplx, n_rot, power):
        n = self.modes
        model = xMCA(self.A, self.B)
        model.set_field_names('sst', 'prcp')
        if norm:
            model.normalize()
        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot, power)
            n = n_rot
        U = model._get_U()
        result = (U['left'].conjugate().T @ U['right']).real
        result = result[:n, :n]
        expected = np.eye(n)

        if model._analysis['power'] > 1:
            assert_raises(AssertionError, assert_allclose, result, expected)
        else:
            assert_allclose(result, expected, **self.tols)

        @classmethod
        def tearDownClass(self):
            pass
