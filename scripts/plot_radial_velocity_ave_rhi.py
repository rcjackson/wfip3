import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
import cmweather
import pandas as pd
import numpy as np
import pygmt

from distributed import Client, LocalCluster

gridded_data_path = '/lcrc/group/earthscience/rjackson/leaff/gridded'
#dates = pd.date_range('2025-08-03', '2025-08-05', freq='1d')
dates = ["20250729", "20250730"]
out_file = 'rad_vel_hist.nc'



if __name__ == "__main__":
    hours = [20, 21, 22, 23, 0, 1]
    hours_local = [13, 14, 15, 16, 17, 18]
    rad_vel = None
    total_points = None
    if True:
        ds = []
        for d in dates:
            print(d)
            with xr.open_mfdataset(os.path.join(gridded_data_path, f'*{d}*rhi*.nc')) as in_ds:
                if d == "20250729":
                    ds.append(in_ds.sel(time=slice("2025-07-29T20:00:00", "2025-07-30T00:00:00")))
                elif d == "20250730":
                    ds.append(in_ds.sel(time=slice("2025-07-30T00:00:00", "2025-07-30T06:00:00")))
        ds = xr.concat(ds, dim='time')

        ds_groupby_hour = ds.groupby("time.hour").mean().reindex(hour=range(24))
        ds_groupby_count = ds.groupby("time.hour").count().reindex(hour=range(24))
        print(ds_groupby_hour)
        ds_groupby_count = ds_groupby_count.load()
        ds_groupby_hour = ds_groupby_hour.load()
        if total_points is None:
            total_points = ds_groupby_count["radial_velocity"].sel(hour=hours)
        else:
            total_points += ds_groupby_count["radial_velocity"].sel(hour=hours)

        if rad_vel is None:
            rad_vel = ds_groupby_hour["radial_velocity"].sel(hour=hours)
        else:
            rad_vel = ds_groupby_hour["radial_velocity"].sel(hour=hours)


    print(rad_vel)

    fig, ax = plt.subplots(2, len(hours) // 2, figsize=(2*len(hours), 6))
    rad_vel["x"] = rad_vel["x"]/1e3
    rad_vel["z"] = rad_vel["z"]/1e3
    for i, hr in enumerate(hours):
        print(i)
        print(total_points.sel(hour=hr).max())
        vr = rad_vel.sel(hour=hr).where(total_points.sel(hour=hr).values > 1)
        im = vr.plot(vmin=-10, vmax=10,
                cmap='balance', ax=ax[i % 2, i // 2],
                add_colorbar=False, zorder=0, alpha=1)

        ax[i % 2, i // 2].set_xlim([0, 3])
        ax[i % 2, i // 2].set_ylim([0, 1.5])
        ax[i % 2, i // 2].set_xlabel("X [km]")
        ax[i % 2, i // 2].set_ylabel("Z [km]")
        ax[i % 2, i // 2].set_title(f"{hours_local[i]} LDT")
        if i // 2 > 0:
            ax[i % 2, i // 2].set_ylabel("")
        if i % 2 == 0:
            ax[i % 2, i // 2].set_xlabel("")
    # Add fixed colorbar axis
    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cax, label="$v_{r}$ [$m\ s^{-1}$]", shrink=0.6)
    #fig.tight_layout()
    fig.savefig('wref_paper_fig_mean_radial_vel_rhi.png', dpi=150)
    rad_vel.close()
