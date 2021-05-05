import unittest
import warnings
from os import getcwd
from os.path import join
from shutil import rmtree

import numpy as np
import xarray as xr
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

    def test_standard_mca(self):
        files = {
            'svalues' : 'mca_c0_r00_p00_singular_values.nc',
            'eofs_A' : 'mca_c0_r00_p00_sst_eofs.nc',
            'eofs_B' : 'mca_c0_r00_p00_prcp_eofs.nc',
            'pcs_A' : 'mca_c0_r00_p00_sst_pcs.nc',
            'pcs_B' : 'mca_c0_r00_p00_prcp_pcs.nc',
        }
        svalues = xr.open_dataarray(join(self.path, files['svalues']))
        eofs_A = xr.open_dataarray(join(self.path, files['eofs_A']))
        eofs_B = xr.open_dataarray(join(self.path, files['eofs_B']))
        pcs_A = xr.open_dataarray(join(self.path, files['pcs_A']))
        pcs_B = xr.open_dataarray(join(self.path, files['pcs_B']))

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve()
        vals = mca.singular_values()
        eofs = mca.eofs()
        pcs = mca.pcs()
        fields = mca.reconstructed_fields()
        assert_allclose(svalues, vals, err_msg='svalues do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='eofs A do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='eofs B do not match')
        assert_allclose(pcs_A, pcs['left'], err_msg='pcs A do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='pcs B do not match')
        assert_allclose(self.A, fields['left'])
        assert_allclose(self.B, fields['right'], atol=9e-2)
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c0_r00_p00.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values()
        eofs = mca2.eofs()
        pcs = mca2.pcs()
        fields = mca2.reconstructed_fields()
        assert_allclose(svalues, vals, err_msg='singular values do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='left eofs do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='right eofs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='left eofs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='right eofs do not match')
        assert_allclose(self.A, fields['left'], err_msg='left reconstructed field does not match')
        assert_allclose(self.B, fields['right'], atol=9e-2, err_msg='right reconstructed field does not match')

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    def test_rotated_mca(self):
        files = {
            'svalues' : 'mca_c0_r10_p01_singular_values.nc',
            'eofs_A' : 'mca_c0_r10_p01_sst_eofs.nc',
            'eofs_B' : 'mca_c0_r10_p01_prcp_eofs.nc',
            'pcs_A' : 'mca_c0_r10_p01_sst_pcs.nc',
            'pcs_B' : 'mca_c0_r10_p01_prcp_pcs.nc',
        }
        svalues = xr.open_dataarray(join(self.path, files['svalues']))
        eofs_A = xr.open_dataarray(join(self.path, files['eofs_A']))
        eofs_B = xr.open_dataarray(join(self.path, files['eofs_B']))
        pcs_A = xr.open_dataarray(join(self.path, files['pcs_A']))
        pcs_B = xr.open_dataarray(join(self.path, files['pcs_B']))

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve()
        mca.rotate(10)
        vals = mca.singular_values()
        eofs = mca.eofs()
        pcs = mca.pcs()

        assert_allclose(svalues, vals, err_msg='singular values do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='left eofs do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='right eofs do not match')
        assert_allclose(pcs_A, pcs['left'], err_msg='left pcs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='right pcs do not match')
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c0_r10_p01.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values()
        eofs = mca2.eofs()
        pcs = mca2.pcs()
        assert_allclose(svalues, vals, err_msg='singular values do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='left eofs do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='right eofs do not match')
        assert_allclose(pcs_A, pcs['left'], err_msg='left pcs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='right pcs do not match')

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

    def test_complex_mca(self):
        files = {
            'svalues' : 'mca_c1_r10_p01_singular_values.nc',
            'eofs_A' : 'mca_c1_r10_p01_sst_eofs.nc',
            'eofs_B' : 'mca_c1_r10_p01_prcp_eofs.nc',
            'pcs_A' : 'mca_c1_r10_p01_sst_pcs.nc',
            'pcs_B' : 'mca_c1_r10_p01_prcp_pcs.nc',
        }
        svalues = xr.open_dataarray(join(self.path, files['svalues']), engine='h5netcdf')
        eofs_A = xr.open_dataarray(join(self.path, files['eofs_A']), engine='h5netcdf')
        eofs_B = xr.open_dataarray(join(self.path, files['eofs_B']), engine='h5netcdf')
        pcs_A = xr.open_dataarray(join(self.path, files['pcs_A']), engine='h5netcdf')
        pcs_B = xr.open_dataarray(join(self.path, files['pcs_B']), engine='h5netcdf')

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst', 'prcp')
        mca.solve(complexify=True, extend='theta', period=12)
        mca.rotate(10)
        vals = mca.singular_values()
        eofs = mca.eofs()
        pcs = mca.pcs()
        assert_allclose(svalues, vals, err_msg='singular values do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='left eofs do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='right eofs do not match')
        assert_allclose(pcs_A, pcs['left'], err_msg='left pcs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='right pcs do not match')
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        path = './tests/integration/xmca/sst_prcp/mca_c1_r10_p01.info'
        mca2 = xMCA()
        mca2.load_analysis(path)
        vals = mca2.singular_values()
        eofs = mca2.eofs()
        pcs = mca2.pcs()
        assert_allclose(svalues, vals, err_msg='singular values do not match')
        assert_allclose(eofs_A, eofs['left'], err_msg='left eofs do not match')
        assert_allclose(eofs_B, eofs['right'], err_msg='right eofs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='left pcs do not match')
        assert_allclose(pcs_B, pcs['right'], err_msg='right pcs do not match')

        rmtree(join(getcwd(), 'tests/integration/xmca/'))

        @classmethod
        def tearDownClass(self):
            pass
