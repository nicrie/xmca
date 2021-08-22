import numpy as np
import xarray as xr
from xmca.xarray import xMCA
import matplotlib.pyplot as plt

#%%
data = xr.tutorial.open_dataset('air_temperature').air
west = data.sel(lon=slice(200, 260))
east = data.sel(lon=slice(260, 360))
#%%

# MCA #%%
# -----------------------------------------------------------------------------
mca = xMCA(west, east)
mca.solve()

svals = mca.singular_values()
expvar = mca.explained_variance()
pcs = mca.pcs()
eofs = mca.eofs()

# Significance analysis
# -----------------------------------------------------------------------------
mca = xMCA(west, east)
mca.normalize()
mca.apply_coslat()
mca.solve()
svals = mca.singular_values()

# North's Rule of Thumb #%%
# -----------------------------------------------------------------------------
svals_diff = svals.to_dataframe().diff(-1)
svals_err_north = mca.rule_north().to_dataframe()
np.argmax((svals_diff - (2 * svals_err_north)) < 0)

# Rule N #%%
# -----------------------------------------------------------------------------
svals_rule_n = mca.rule_n(100)
median = svals_rule_n.median('run')
q99 = svals_rule_n.quantile(.99, dim='run')
q01 = svals_rule_n.quantile(.01, dim='run')
cutoff = np.argmax(((svals - q99) < 0).values)


# Monte Carlo Bootstrapping #%%
# -----------------------------------------------------------------------------
svals_boot = mca.bootstrapping(100, on_left=True, on_right=True, axis=1, replace=True)



q01 = svals_boot.quantile(0.01, 'run')
q99 = svals_boot.quantile(0.99, 'run').shift({'mode' : -1})
np.argmax((q01 - q99).dropna('mode').values < 0)

pivot = svals_boot.sel(mode=slice(0, 30)).to_dataframe().reset_index().pivot(
    index='run', columns='mode', values='singular values'
)
plt.plot(range(1, 11), svals[:10], marker='o', ls='')
_ = plt.boxplot(pivot, whis=(.05, .95), showfliers=False)
plt.yscale('log')


#%%
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
svals.plot(ax=ax, yscale='log', label='true')
median.plot(ax=ax, yscale='log', color='.5', label='rule N')
q99.plot(ax=ax, yscale='log', color='.5', ls=':')
q01.plot(ax=ax, yscale='log', color='.5', ls=':')
_ = ax.boxplot(pivot, whis=(.05, .95), showfliers=False)
ax.axvline(cutoff + 0.5, ls=':')
ax.set_xlim(-2, 30)
ax.set_ylim(1e-2, 1e3)
ax.set_title('Significance based on Rule N')
ax.legend()
# plt.savefig('../figs/rule-n.png', dpi=90)






# Visual inspection #%%
# -----------------------------------------------------------------------------
mca.set_field_names('West', 'East')
pkwargs = {'orientation' : 'vertical'}
mca.plot(mode=1, **pkwargs)
