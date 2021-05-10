import unittest
import warnings
from os import getcwd
from os.path import join
from shutil import rmtree

import numpy as np
import xarray as xr
try:
    import dask.array
    dask_support = True
except ImportError:
    dask_support = False

from numpy.testing import assert_allclose

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
        self.modes = 10
        # acceptable tolerance for comparison
        self.tols = {'atol': 1e-5, 'rtol': 1e-3}

    def test_standard_mca(self):
        files = {
            'svalues' : 'mca_c0_r00_p00_singular_values.nc',
            'eofs_A' : 'mca_c0_r00_p00_sst_eofs.nc',
            'eofs_B' : 'mca_c0_r00_p00_prcp_eofs.nc',
            'pcs_A' : 'mca_c0_r00_p00_sst_pcs.nc',
            'pcs_B' : 'mca_c0_r00_p00_prcp_pcs.nc',
        }
        svalues = xr.open_dataarray(
            join(self.path, files['svalues'])
        )[:self.modes]
        eofs_A = xr.open_dataarray(
            join(self.path, files['eofs_A'])
        )[..., :self.modes]
        eofs_B = xr.open_dataarray(
            join(self.path, files['eofs_B'])
        )[..., :self.modes]
        pcs_A = xr.open_dataarray(
            join(self.path, files['pcs_A'])
        )[:, :self.modes]
        pcs_B = xr.open_dataarray(
            join(self.path, files['pcs_B'])
        )[:, :self.modes]

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve()
        vals = mca.singular_values(self.modes)
        eofs = mca.eofs(self.modes)
        pcs = mca.pcs(self.modes)
        # fields = mca.reconstructed_fields()
        assert_allclose(
            svalues, vals, err_msg='svalues do not match', **self.tols
        )
        assert_allclose(
            eofs_A, eofs['left'], err_msg='eofs A do not match', **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'], err_msg='eofs B do not match', **self.tols
        )
        assert_allclose(
            pcs_A, pcs['left'], err_msg='pcs A do not match', **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'], err_msg='pcs B do not match', **self.tols
        )
        # assert_allclose(self.A, fields['left'], rtol=1e-3)
        # assert_allclose(self.B, fields['right'], rtol=1e-3, atol=9e-2)
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c0_r00_p00.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values(self.modes)
        eofs = mca2.eofs(self.modes)
        pcs = mca2.pcs(self.modes)
        # fields = mca2.reconstructed_fields()
        assert_allclose(
            svalues, vals, err_msg='singular values do not match', **self.tols
        )
        assert_allclose(
            eofs_A, eofs['left'], err_msg='left eofs do not match', **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'],
            err_msg='right eofs do not match', **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'], err_msg='left eofs do not match', **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'], err_msg='right eofs do not match', **self.tols
        )
        # assert_allclose(
        #   self.A, fields['left'],
        #   err_msg='left reconstructed field does not match'
        # )
        # assert_allclose(
        #   self.B, fields['right'],
        #   err_msg='right reconstructed field does not match'
        # )

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    def test_rotated_mca(self):
        files = {
            'svalues' : 'mca_c0_r10_p01_singular_values.nc',
            'eofs_A' : 'mca_c0_r10_p01_sst_eofs.nc',
            'eofs_B' : 'mca_c0_r10_p01_prcp_eofs.nc',
            'pcs_A' : 'mca_c0_r10_p01_sst_pcs.nc',
            'pcs_B' : 'mca_c0_r10_p01_prcp_pcs.nc',
        }

        svalues = xr.open_dataarray(
            join(self.path, files['svalues'])
        )[:self.modes]
        eofs_A = xr.open_dataarray(
            join(self.path, files['eofs_A'])
        )[..., :self.modes]
        eofs_B = xr.open_dataarray(
            join(self.path, files['eofs_B'])
        )[..., :self.modes]
        pcs_A = xr.open_dataarray(
            join(self.path, files['pcs_A'])
        )[:, :self.modes]
        pcs_B = xr.open_dataarray(
            join(self.path, files['pcs_B'])
        )[:, :self.modes]

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve()
        mca.rotate(10)
        vals = mca.singular_values(self.modes)
        eofs = mca.eofs(self.modes)
        pcs = mca.pcs(self.modes)

        assert_allclose(
            svalues, vals,
            err_msg='singular values do not match',
            **self.tols
        )
        assert_allclose(
            eofs_A, eofs['left'],
            err_msg='left eofs do not match',
            **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'],
            err_msg='right eofs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_A, pcs['left'],
            err_msg='left pcs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'],
            err_msg='right pcs do not match',
            **self.tols
        )
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c0_r10_p01.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values(self.modes)
        eofs = mca2.eofs(self.modes)
        pcs = mca2.pcs(self.modes)

        assert_allclose(
            svalues[:20], vals[:20],
            err_msg='singular values do not match',
            **self.tols)
        assert_allclose(
            eofs_A, eofs['left'],
            err_msg='left eofs do not match',
            **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'],
            err_msg='right eofs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_A, pcs['left'],
            err_msg='left pcs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'],
            err_msg='right pcs do not match',
            **self.tols
        )

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    def test_complex_mca(self):
        files = {
            'svalues' : 'mca_c1_r10_p01_singular_values.nc',
            'eofs_A' : 'mca_c1_r10_p01_sst_eofs.nc',
            'eofs_B' : 'mca_c1_r10_p01_prcp_eofs.nc',
            'pcs_A' : 'mca_c1_r10_p01_sst_pcs.nc',
            'pcs_B' : 'mca_c1_r10_p01_prcp_pcs.nc',
        }

        svalues = xr.open_dataarray(
            join(self.path, files['svalues']),
            engine='h5netcdf'
        )[:self.modes]
        eofs_A = xr.open_dataarray(
            join(self.path, files['eofs_A']),
            engine='h5netcdf'
        )[..., :self.modes]
        eofs_B = xr.open_dataarray(
            join(self.path, files['eofs_B']),
            engine='h5netcdf'
        )[..., :self.modes]
        pcs_A = xr.open_dataarray(
            join(self.path, files['pcs_A']),
            engine='h5netcdf'
        )[:, :self.modes]
        pcs_B = xr.open_dataarray(
            join(self.path, files['pcs_B']),
            engine='h5netcdf'
        )[:, :self.modes]

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve(complexify=True, extend='theta', period=12)
        mca.rotate(10)
        vals = mca.singular_values(self.modes)
        eofs = mca.eofs(self.modes)
        pcs = mca.pcs(self.modes)

        assert_allclose(
            svalues, vals,
            err_msg='singular values do not match',
            **self.tols
        )
        assert_allclose(
            eofs_A, eofs['left'],
            err_msg='left eofs do not match',
            **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'],
            err_msg='right eofs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_A, pcs['left'],
            err_msg='left pcs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'],
            err_msg='right pcs do not match',
            **self.tols
        )
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c1_r10_p01.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values(self.modes)
        eofs = mca2.eofs(self.modes)
        pcs = mca2.pcs(self.modes)

        assert_allclose(
            svalues, vals,
            err_msg='singular values do not match',
            **self.tols
        )
        assert_allclose(
            eofs_A, eofs['left'],
            err_msg='left eofs do not match',
            **self.tols
        )
        assert_allclose(
            eofs_B, eofs['right'],
            err_msg='right eofs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_A, pcs['left'],
            err_msg='left pcs do not match',
            **self.tols
        )
        assert_allclose(
            pcs_B, pcs['right'],
            err_msg='right pcs do not match',
            **self.tols
        )

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    def test_complex_mca_dask(self):
        temp = xr.tutorial.open_dataset(
            'air_temperature',
            chunks={'lat': 25, 'lon': 25, 'time': -1}
        )
        temp = temp.coarsen({'lat': 2, 'lon': 2}, boundary='trim').mean()
        temp = temp.air
        n_var = np.product(temp.shape[1:])
        svd_kwargs = {'k' : 0.5 * n_var}

        mca = xMCA(temp, temp)
        mca.set_field_names('temp', 'temp')
        mca.solve(complexify=True, extend='exp', svd_kwargs=svd_kwargs)
        mca.rotate(10)
        mca.singular_values(self.modes)
        mca.eofs(self.modes)
        mca.pcs(self.modes)
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/temp_temp/mca_c1_r10_p01.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        mca2.singular_values(self.modes)
        mca2.eofs(self.modes)
        mca2.pcs(self.modes)
        mca2.plot(1)

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    @classmethod
    def tearDownClass(self):
        pass
