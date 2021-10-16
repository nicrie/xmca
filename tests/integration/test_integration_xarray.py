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

from xmca.xarray import xMCA


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
            self.A = xr.open_dataarray(join(self.path, 'sst.nc'))
            self.B = xr.open_dataarray(join(self.path, 'prcp.nc'))

        # how many modes to compare
        self.modes = 100
        # acceptable tolerance for comparison
        self.tols = {'atol': 1e-3, 'rtol': 1e-3}

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
        eofs = new.eofs(n, rotated=False)
        assert_allclose(
            svalues, vals, err_msg='svalues do not match', **self.tols
        )
        assert_allclose(
            V1, eofs['left'], err_msg='eofs A do not match', **self.tols
        )
        assert_allclose(
            V2, eofs['right'], err_msg='eofs B do not match', **self.tols
        )

        # test cpsñat weighting
        if analysis == 'std':
            model = xMCA(self.A, self.B)
            model.normalize()
            model.apply_coslat()
            model.solve()
            fields = model.fields()
            model.save_analysis('tests/integration/temp/')

            reload = xMCA()
            reload.load_analysis('tests/integration/temp/info.xmca')
            reload.apply_coslat()
            fields_reloaded = reload.fields()
            for f, r in zip(fields.values(), fields_reloaded.values()):
                assert_allclose(
                    f, r,
                    err_msg='mismatch after coslat', **self.tols
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
        n_modes = self.modes
        dof = self.A.shape[0] - 1

        model = xMCA(self.A, self.B)
        model.set_field_names('sst', 'prcp')
        if norm:
            model.normalize()
        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot, power)
            n_modes = n_rot
        U = model._get_U()
        result = (U['left'].conjugate().T @ U['right']).real / dof
        result = result[:n_modes, :n_modes]
        expected = np.eye(n_modes)

        if model._analysis['power'] > 1:
            assert_raises(AssertionError, assert_allclose, result, expected)
        else:
            assert_allclose(result, expected, **self.tols)

    def name_func_get(testcase_func, param_num, param):
        return "{:}_{:s}".format(
            testcase_func.__name__,
            parameterized.to_safe_name('_'.join([str(a) for a in param.args])),
        )

    @parameterized.expand([
        ('std', None, 'None', 0, True),
        ('cplx', None, 'None', 0, True),
        ('varmx', None, 'None', 0, True),
        ('std', 100, 'None', 0, True),
        ('cplx', 100, 'None', 0, True),
        ('varmx', 100, 'None', 0, True),
        ('std', None, 'max', 0, True),
        ('cplx', None, 'std', 0, True),
        ('varmx', None, 'eigen', 0, True),
        ('std', 100, 'eigen', 0, True),
        ('cplx', 100, 'std', 0, True),
        ('varmx', 100, 'max', 0, True),
        ('cplx', 100, 'std', 1.234, True),
        ('varmx', 100, 'max', 3, True),
        ('std', 100, 'eigen', -2, False),
        ('cplx', 100, 'std', 1.234, False),
        ('varmx', 100, 'max', 3, False),
    ], name_func=name_func_get)
    def test_getter(self, analysis, n, scaling, phase_shift, rotated):
        cplx = False,
        n_rot = 0
        if analysis == 'cplx':
            cplx = True
        if analysis == 'varmx':
            n_rot = 10
        model = xMCA(self.A, self.B)
        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)
        model.pcs(n, scaling, phase_shift, rotated)
        model.eofs(n, scaling, phase_shift, rotated)
        model.spatial_amplitude(n, scaling, rotated)
        model.spatial_phase(n, phase_shift, rotated)
        model.temporal_amplitude(n, scaling, rotated)
        model.temporal_phase(n, phase_shift, rotated)

    @parameterized.expand([
        ('std'),
        ('cplx'),
        ('varmx')
    ], name_func=name_func_get)
    def test_hom_het_patterns(self, analysis):
        cplx = False
        n_rot = 0
        if analysis == 'cplx':
            cplx = True
        if analysis == 'varmx':
            n_rot = 10
        model = xMCA(self.A, self.B)
        model.solve(complexify=cplx)
        if n_rot > 0:
            model.rotate(n_rot)
        hom_pat, _ = model.homogeneous_patterns(10)
        het_pat, _ = model.heterogeneous_patterns(10)

        self.assertGreaterEqual(1, abs(hom_pat['left']).max())
        self.assertGreaterEqual(1, abs(hom_pat['right']).max())
        self.assertGreaterEqual(1, abs(het_pat['left']).max())
        self.assertGreaterEqual(1, abs(het_pat['right']).max())

    @parameterized.expand([
        ('std'),
        ('cplx'),
        ('varmx')
    ], name_func=name_func_get)
    def test_field(self, analysis):
        expected = {'left' : self.A, 'right': self.B}
        cplx = False
        n_rot = 0
        if analysis == 'cplx':
            cplx = True
        if analysis == 'varmx':
            n_rot = 10
        model = xMCA(self.A, self.B)
        model.solve(complexify=cplx)
        if n_rot > 0:
            model.rotate(n_rot)
        model.fields()
        result = model.fields(original_scale=True)

        assert_allclose(result['left'].real, expected['left'], **self.tols)
        assert_allclose(result['right'].real, expected['right'], **self.tols)

    @parameterized.expand([
        ('std'),
        ('cplx'),
        ('varmx')
    ], name_func=name_func_get)
    def test_field_scaling(self, analysis):
        expected = {'left' : self.A, 'right': self.B}
        cplx = False
        n_rot = 0
        if analysis == 'cplx':
            cplx = True
        if analysis == 'varmx':
            n_rot = 10
        model = xMCA(self.A, self.B)
        result1 = model.fields(original_scale=True)
        model.normalize()
        result2 = model.fields(original_scale=True)
        model.apply_coslat()
        result3 = model.fields(original_scale=True)
        model.solve(complexify=cplx)
        result4 = model.fields(original_scale=True)
        if n_rot > 0:
            model.rotate(n_rot)
        result5 = model.fields(original_scale=True)

        assert_allclose(result1['left'].real, expected['left'], **self.tols)
        assert_allclose(result2['left'].real, expected['left'], **self.tols)
        assert_allclose(result3['left'].real, expected['left'], **self.tols)
        assert_allclose(result4['left'].real, expected['left'], **self.tols)
        assert_allclose(result5['left'].real, expected['left'], **self.tols)
        assert_allclose(result1['right'].real, expected['right'], **self.tols)
        assert_allclose(result2['right'].real, expected['right'], **self.tols)
        assert_allclose(result3['right'].real, expected['right'], **self.tols)
        assert_allclose(result4['right'].real, expected['right'], **self.tols)
        assert_allclose(result5['right'].real, expected['right'], **self.tols)

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
            model = xMCA(self.A)
        elif analysis == 'bi':
            model = xMCA(self.A, self.B)

        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)
        model.plot(n)

    @parameterized.expand([
        ('uni', 'std', 1, 'None', 0, 'no_weight'),
        ('uni', 'varmx', 15, 'None', 0, 'no_weight'),
        ('uni', 'std', 1, 'max', 0, 'no_weight'),
        ('uni', 'std', 1, 'eigen', 0, 'no_weight'),
        ('uni', 'varmx', 15, 'std', 0, 'no_weight'),
        ('uni', 'varmx', 15, 'std', 0.5, 'no_weight'),
        ('uni', 'varmx', 15, 'eigen', 0.5, 'no_weight'),
        ('bi', 'std', 1, 'None', 0, 'no_weight'),
        ('bi', 'varmx', 15, 'None', 0, 'no_weight'),
        ('bi', 'std', 1, 'max', 0, 'no_weight'),
        ('bi', 'varmx', 15, 'max', 0, 'no_weight'),
        ('bi', 'varmx', 15, 'std', 0.5, 'no_weight'),
        ('bi', 'std', 1, 'None', 0, 'coslat'),
        ('bi', 'std', 1, 'eigen', 0, 'coslat'),
        ('bi', 'varmx', 15, 'None', 0, 'coslat'),
        ('bi', 'std', 1, 'max', 0, 'coslat'),
        ('bi', 'varmx', 15, 'max', 0, 'coslat'),
        ('bi', 'varmx', 15, 'std', 0.5, 'coslat'),
        ('bi', 'varmx', 15, 'eigen', 0.5, 'coslat'),
    ], name_func=name_func_get)
    def test_predict(self, analysis, flavour, n, scaling, phase_shift, weight):
        left = self.A
        right = self.B
        new_left = self.A.isel(time=slice(0, 20))
        new_right = self.A.isel(time=slice(0, 20))

        if analysis == 'uni':
            model = xMCA(left)
        elif analysis == 'bi':
            model = xMCA(left, right)
        if weight == 'coslat':
            model.normalize()
            model.apply_coslat()
        model.solve()
        if flavour == 'varmx':
            model.rotate(10)

        pcs = model.pcs(n=n, scaling=scaling, phase_shift=phase_shift)
        expected = {
            k: p.sel(mode=slice(1, 10)).isel(time=slice(0, 20)) for k, p in pcs.items()
        }
        result = model.predict(
            new_left,
            n=n, scaling=scaling, phase_shift=phase_shift
        )
        if analysis == 'bi':
            model.predict(new_right)
            result = model.predict(
                new_left, new_right,
                n=n, scaling=scaling, phase_shift=phase_shift
            )

        assert_allclose(expected['left'], result['left'], **self.tols)
        # check wrong input
        # missing time dimension
        self.assertRaises(
            ValueError, model.predict, new_left.isel(time=0)
        )
        # wrong spatial dimensions
        self.assertRaises(
            ValueError, model.predict, new_left.isel(lon=slice(0, 10))
        )

    @parameterized.expand([
        (None,), (1,), (10,), (100,),
    ], name_func=name_func_get)
    def test_norm(self, n):
        left = self.A
        right = self.B
        model = xMCA(left, right)
        model.solve(complexify=True)
        model.rotate(10)
        model.norm(n)

    @parameterized.expand([
        (None,), (1,), (10,), (100,),
    ], name_func=name_func_get)
    def test_variance(self, n):
        left = self.A
        right = self.B
        model = xMCA(left, right)
        model.solve(complexify=True)
        model.rotate(10)
        model.variance(n)

    def test_summary(self):
        left = self.A
        right = self.B
        model = xMCA(left, right)
        model.solve()

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            model.summary()

    @parameterized.expand([
        ('uni', 'std', 1),
        ('uni', 'cplx', 1),
        ('uni', 'varmx', 1),
        ('bi', 'std', 1),
        ('bi', 'cplx', 1),
        ('bi', 'varmx', 1),
        ('uni', 'std', 10),
        ('uni', 'cplx', 10),
        ('uni', 'varmx', 10),
        ('bi', 'std', 10),
        ('bi', 'cplx', 10),
        ('bi', 'varmx', 10),
        ('uni', 'std', 100),
        ('uni', 'cplx', 100),
        ('uni', 'varmx', 100),
        ('bi', 'std', 100),
        ('bi', 'cplx', 100),
        ('bi', 'varmx', 100),
    ], name_func=name_func_get)
    def test_truncate(self, analysis, flavour, trunc):
        cplx = False,
        n_rot = 0
        if flavour == 'cplx':
            cplx = True
        if flavour == 'varmx':
            n_rot = 10
        if analysis == 'uni':
            model = xMCA(self.A)
        elif analysis == 'bi':
            model = xMCA(self.A, self.B)

        model.solve(complexify=cplx)
        if n_rot > 1:
            model.rotate(n_rot)

        if (flavour == 'varmx') & (trunc < n_rot):
            assert_raises(ValueError, model.truncate, trunc)
        else:
            model.truncate(trunc)

    def test_apply_weights(self):
        model = xMCA(self.A, self.B)
        weights = {}
        weights['left'] = self.A.coords['lat']
        weights['right'] = self.B.coords['lat']
        model.apply_weights(**weights)

    def test_complex_solver(self):
        model = xMCA(self.A, self.B)
        model.solve(complexify=True, extend=False)
        model.solve(complexify=True, extend='theta', period=12)
        model.solve(complexify=True, extend='exp', period=6)

    def test_solver_errors(self):
        model = xMCA(self.A, self.B)
        with self.assertRaises(RuntimeError):
            model.singular_values()
            model.norms()
            model.pcs()
            model.eofs()
            model.rotation_matrix()
            model.correlation_matrix()

        model.solve()
        model.rotation_matrix()
        model.correlation_matrix()

        model.rotate(10)
        model.rotation_matrix()
        model.correlation_matrix()

    @parameterized.expand([
        ('uni', 'std', 0, True, 1, True, True, 'standard'),
        ('uni', 'std', 0, True, 1, False, False, 'standard'),
        ('uni', 'std', 0, True, 1, True, False, 'standard'),
        ('uni', 'cplx', 0, True, 1, True, False, 'standard'),
        ('uni', 'varmx', 0, True, 1, True, False, 'standard'),
        ('uni', 'std', 1, True, 1, True, False, 'standard'),
        ('uni', 'cplx', 1, False, 1, True, False, 'standard'),
        ('uni', 'varmx', 1, False, 2, True, False, 'standard'),
        ('uni', 'varmx', 1, False, 3, True, False, 'standard'),
        ('bi', 'std', 0, True, 1, True, False, 'standard'),
        ('bi', 'cplx', 0, True, 1, True, False, 'standard'),
        ('bi', 'varmx', 0, True, 1, True, False, 'standard'),
        ('bi', 'std', 1, True, 1, True, False, 'standard'),
        ('bi', 'cplx', 1, False, 1, True, False, 'standard'),
        ('bi', 'varmx', 1, False, 2, True, False, 'standard'),
        ('bi', 'varmx', 1, False, 3, True, False, 'standard'),
        ('bi', 'varmx', 1, False, 3, True, False, 'iterative'),
    ], name_func=name_func_get)
    def test_significance_methods(
            self, analysis, flavour, axis, replace, block_size,
            on_left, on_right, strategy):
        cplx = False,
        n_rot = 0
        if flavour == 'cplx':
            cplx = True
        if flavour == 'varmx':
            n_rot = 10
        if analysis == 'uni':
            model = xMCA(self.A)
        elif analysis == 'bi':
            model = xMCA(self.A, self.B)

        model.solve(complexify=cplx)
        if flavour == 'varmx':
            model.rotate(n_rot)

        model.rule_north(3)
        model.rule_n(3)
        incorrect_params = (
            (axis not in [0, 1]) or
            ((analysis == 'uni') and (on_right == True)) or
            # ((on_left == False) and (on_right == False)) or
            ((self.A.shape[0] % block_size) != 0)
        )
        if incorrect_params:
            assert_raises(
                ValueError,
                model.bootstrapping,
                3, 3, axis, on_left, on_right, block_size, replace, strategy, True
            )
        else:
            model.bootstrapping(
                n_runs=3, n_modes=3, axis=axis,
                on_left=on_left, on_right=on_right,
                block_size=block_size, replace=replace,
                strategy=strategy, disable_progress=True
            )

    @classmethod
    def tearDownClass(self):
        pass
