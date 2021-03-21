import unittest
import warnings
from os import getcwd
from os.path import join
from shutil import rmtree

import numpy as np
import xarray as xr

from xmca.xarray import xMCA

# xmca.xarray.xMCA(np.random.randn(2,4,3))

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Load test data
        self.path = 'tests/integration/fixtures'
        # ignore some deprecation warnings of xarray
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            self.A = xr.open_dataarray(join(self.path,'sst.nc'))
            self.B = xr.open_dataarray(join(self.path,'prcp.nc'))

    def test_standard_mca(self):
        svalues = xr.open_dataarray(join(self.path,'mca_c0_r00_p00_singular_values.nc'))
        eofs_A = xr.open_dataarray(join(self.path,'mca_c0_r00_p00_sst_eofs.nc'))
        eofs_B = xr.open_dataarray(join(self.path,'mca_c0_r00_p00_prcp_eofs.nc'))
        pcs_A = xr.open_dataarray(join(self.path,'mca_c0_r00_p00_sst_pcs.nc'))
        pcs_B = xr.open_dataarray(join(self.path,'mca_c0_r00_p00_prcp_pcs.nc'))

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst','prcp')
        mca.solve()
        np.testing.assert_allclose(svalues, mca.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='left pcs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='right pcs do not match')
        np.testing.assert_allclose(self.A, mca.reconstructed_fields()['left'])
        np.testing.assert_allclose(self.B, mca.reconstructed_fields()['right'], atol=9e-2)
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        mca2 = xMCA()
        mca2.load_analysis('./tests/integration/xmca/sst_prcp/mca_c0_r00_p00.info')
        np.testing.assert_allclose(svalues, mca2.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca2.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca2.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'],err_msg='left eofs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(self.A, mca2.reconstructed_fields()['left'], err_msg='left reconstructed field does not match')
        np.testing.assert_allclose(self.B, mca2.reconstructed_fields()['right'], atol=9e-2, err_msg='right reconstructed field does not match')

        rmtree(join(getcwd(),'tests/integration/xmca/'))


    def test_rotated_mca(self):
        svalues = xr.open_dataarray(join(self.path,'mca_c0_r10_p01_singular_values.nc'))
        eofs_A = xr.open_dataarray(join(self.path,'mca_c0_r10_p01_sst_eofs.nc'))
        eofs_B = xr.open_dataarray(join(self.path,'mca_c0_r10_p01_prcp_eofs.nc'))
        pcs_A = xr.open_dataarray(join(self.path,'mca_c0_r10_p01_sst_pcs.nc'))
        pcs_B = xr.open_dataarray(join(self.path,'mca_c0_r10_p01_prcp_pcs.nc'))

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst','prcp')
        mca.solve()
        mca.rotate(10)
        np.testing.assert_allclose(svalues, mca.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='left pcs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='right pcs do not match')
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        mca2 = xMCA()
        mca2.load_analysis('./tests/integration/xmca/sst_prcp/mca_c0_r10_p01.info')
        np.testing.assert_allclose(svalues, mca2.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca2.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca2.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'], err_msg='left pcs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'], err_msg='right pcs do not match')

        rmtree(join(getcwd(),'tests/integration/xmca/'))


    def test_complex_mca(self):
        svalues = xr.open_dataarray(join(self.path,'mca_c1_r10_p01_singular_values.nc'), engine='h5netcdf')
        eofs_A = xr.open_dataarray(join(self.path,'mca_c1_r10_p01_sst_eofs.nc'), engine='h5netcdf')
        eofs_B = xr.open_dataarray(join(self.path,'mca_c1_r10_p01_prcp_eofs.nc'), engine='h5netcdf')
        pcs_B = xr.open_dataarray(join(self.path,'mca_c1_r10_p01_prcp_pcs.nc'), engine='h5netcdf')
        pcs_A = xr.open_dataarray(join(self.path,'mca_c1_r10_p01_sst_pcs.nc'), engine='h5netcdf')

        mca = xMCA(self.A, self.B)
        mca.set_field_names('sst','prcp')
        mca.solve(complexify=True, theta=True, period=12)
        mca.rotate(10)
        np.testing.assert_allclose(svalues, mca.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='left pcs do not match')
        np.testing.assert_allclose(pcs_B, mca.pcs()['right'], err_msg='right pcs do not match')
        mca.plot(1)
        mca.save_analysis('./tests/integration')

        mca2 = xMCA()
        mca2.load_analysis('./tests/integration/xmca/sst_prcp/mca_c1_r10_p01.info')
        np.testing.assert_allclose(svalues, mca2.singular_values(), err_msg='singular values do not match')
        np.testing.assert_allclose(eofs_A, mca2.eofs()['left'], err_msg='left eofs do not match')
        np.testing.assert_allclose(eofs_B, mca2.eofs()['right'], err_msg='right eofs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'], err_msg='left pcs do not match')
        np.testing.assert_allclose(pcs_B, mca2.pcs()['right'], err_msg='right pcs do not match')

        rmtree(join(getcwd(),'tests/integration/xmca/'))

        @classmethod
        def tearDownClass(self):
            pass
