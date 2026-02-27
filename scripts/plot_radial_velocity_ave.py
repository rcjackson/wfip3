import xarray as xr
import matplotlib.pyplot as plt
import os
import cmweather

from distributed import Client, LocalCluster

gridded_data_path = '/lcrc/group/earthscience/rjackson/leaff/gridded'

hours = [22, 23, 0, 1]
hours_local = [15, 16, 17, 18]
with Client(LocalCluster(n_workers=32)) as c:
    with xr.open_mfdataset(os.path.join(gridded_data_path, '*.nc')) as ds:
        ds_groupby_hour = ds.groupby("time.hour").mean()
        fig, ax = plt.subplots(len(hours), 1, figsize=(3*len(hours), 1))
        for i, hr in enumerate(hours):
            ds_groupby_hour.sel(hour=hr).sel(z=50.).plot(vmin=-10, vmax=10, cmap='balance', ax=ax[i])
            ax[i].set_xlim([-2000, 2000])
            ax[i].set_ylim([-2000, 2000])
            ax[i].set_title(f"{hours_local[i]} LDT")
        fig.tight_layout()
        fig.savefig('wref_paper_fig_mean_radial_vel.png', dpi=150)


    
