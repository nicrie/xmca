import xarray as xr
import xmca.xarray as xmca

prcp = xr.open_dataarray('prcp.nc')
sst = xr.open_dataarray('sst.nc')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve()
model.save_analysis('std')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve()
model.rotate(10, 1)
model.save_analysis('rot')

model = xmca.xMCA(sst, prcp)
model.set_field_names('sst', 'prcp')
model.solve(complexify=True)
model.save_analysis('cplx')
