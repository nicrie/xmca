import numpy as np
import xarray as xr
from xmca.xarray import xMCA
import matplotlib.pyplot as plt

#%%
data = xr.tutorial.open_dataset('air_temperature').air
west = data.sel(lon=slice(200, 260))
east = data.sel(lon=slice(260, 360))
#%%

pca = xMCA(west)                        # PCA of west coast
pca.solve(complexify=False)            # True for complex PCA

eigenvalues = pca.singular_values()     # singular vales = eigenvalues for PCA
expvar      = pca.explained_variance()  # explained variance
pcs         = pca.pcs()                 # Principal component scores (PCs)
eofs        = pca.eofs()                # spatial patterns (EOFs)

#%%
pca.rotate(n_rot=10, power=1)

expvar_rot  = pca.explained_variance()  # explained variance
pcs_rot     = pca.pcs()                 # Principal component scores (PCs)
eofs_rot    = pca.eofs()                # spatial patterns (EOFs)

# MCA #%%
# -----------------------------------------------------------------------------
mca = xMCA(west, east)                  # MCA of field A and B
mca.apply_coslat()                      # area weighting based on latitude
mca.solve(complexify=False)             # True for complex MCA

svals = mca.singular_values()           # singular vales
pcs = mca.pcs()                         # expansion coefficient (PCs)
eofs = mca.eofs()                       # spatial patterns (EOFs)

# Significance analysis based on Rule N
# -----------------------------------------------------------------------------
surr = mca.rule_n(20)
median = surr.median('run')
q99 = surr.quantile(.99, dim='run')
q01 = surr.quantile(.01, dim='run')

cutoff = np.sum((svals - q99 > 0)).values


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
svals.plot(ax=ax, yscale='log', label='true')
median.plot(ax=ax, yscale='log', color='.5', label='rule N')
q99.plot(ax=ax, yscale='log', color='.5', ls=':')
q01.plot(ax=ax, yscale='log', color='.5', ls=':')
ax.axvline(cutoff + 0.5, ls=':')
ax.set_xlim(-2, 200)
ax.set_ylim(1e-1, 2.5e4)
ax.set_title('Significance based on Rule N')
ax.legend()
plt.savefig('../figs/rule-n.png', dpi=90)

# Visual inspection #%%
# -----------------------------------------------------------------------------
mca.set_field_names('West', 'East')
pkwargs = {'orientation' : 'vertical'}
mca.plot(mode=1, **pkwargs)
