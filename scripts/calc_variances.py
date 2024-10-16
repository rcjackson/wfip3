import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
from glob import glob
from statsmodels.tsa.stattools import acf

input_path = '/lcrc/group/earthscience/rjackson/dl-sparc-canyons/netcdf/'
output_path = '/lcrc/group/earthscience/rjackson/dl-sparc-canyons/w_variances/'

file_list = input_path + '/%s/*Stare*.nc' % sys.argv[1]
print(file_list)
stares = sorted(glob(file_list))
print(stares)

def get_variances(input_ds):
    dt_range = pd.date_range(input_ds['time'].values[0], input_ds['time'].values[-1], periods=
                         int((input_ds['time'].values[-1] - input_ds['time'].values[0])/0.5e9))
    nperiods = len(dt_range)
    print(nperiods)
    inp_ds = input_ds.reindex(time=dt_range, method='nearest', tolerance=np.timedelta64(1, 's'))
    
    time_10min = inp_ds["time"].resample(time="10min").min().values
    noise_variance = np.zeros((time_10min.shape[0], len(inp_ds['range'])))
    atmos_variance = np.zeros_like(noise_variance)
    w_25 = np.zeros_like(noise_variance)
    w_50 = np.zeros_like(noise_variance)
    w_75 = np.zeros_like(noise_variance)
    w_min = np.zeros_like(w_25)
    w_max = np.zeros_like(w_min)
    variance = np.zeros_like(noise_variance)
    for i, j in enumerate(range(0, nperiods, 1200)):
        for k in range(0, len(inp_ds['range'])):
            series = inp_ds['radial_velocity'].values[j:min([j+3600, nperiods]), k]
            series_intensity = inp_ds['intensity'].values[j:min([j+3600, nperiods])]
            series = series[np.isfinite(series)]
            if len(series) == 0:
                continue
            try:
                v_acf = acf(series, nlags=5)
            except ValueError:
                variance[i, k] = np.nan
                noise_variance[i, k] = np.nan
                atmos_variance[i, k] = np.nan
                w_25[i, k] = np.nan
                w_50[i, k] = np.nan
                w_75[i, k] = np.nan
                w_min[i, k] = np.min(series)
                w_max[i, k] = np.max(series)
                continue
            variance[i, k] = np.var(series)
            noise_variance[i, k] = (v_acf[0] - v_acf[1]) * variance[i, k]
            atmos_variance[i, k] = variance[i, k] - noise_variance[i, k]
            w_25[i, k] = np.percentile(series, 0.25)
            w_50[i, k] = np.percentile(series, 0.5)
            w_75[i, k] = np.percentile(series, 0.75)
            w_min[i, k] = np.min(series)
            w_max[i, k] = np.max(series)
    variance = xr.DataArray(variance, dims=["time", "range"])
    variance.attrs["long_name"] = "Total variance"
    variance.attrs["units"] = "m s-1"
    noise_variance = xr.DataArray(noise_variance, dims=["time", "range"])
    noise_variance.attrs["long_name"] = "Variance due to noise"
    noise_variance.attrs["units"] = "m s-1"
    atmos_variance = xr.DataArray(atmos_variance, dims=["time", "range"])
    atmos_variance.attrs["long_name"] = "Variance due to signal"
    atmos_variance.attrs["units"] = "m s-1"
    w_25 = xr.DataArray(w_25, dims=["time", "range"])
    w_25.attrs["long_name"] = "25th percentile of w"
    w_25.attrs["units"] = "m s-1"
    w_50 = xr.DataArray(w_50, dims=["time", "range"])
    w_50.attrs["long_name"] = "50th percentile of w"
    w_50.attrs["units"] = "m s-1"
    w_75 = xr.DataArray(w_75, dims=["time", "range"])
    w_75.attrs["long_name"] = "75th percentile of w"
    w_75.attrs["units"] = "m s-1"
    w_min = xr.DataArray(w_min, dims=["time", "range"])
    w_min.attrs["long_name"] = "Minimum w"
    w_min.attrs["units"] = "m s-1"
    w_max = xr.DataArray(w_max, dims=["time", "range"])
    w_max.attrs["long_name"] = "Maximum w"
    w_max.attrs["units"] = "m s-1"
    
    variance_ds = xr.Dataset({'time': time_10min,
                              'range': inp_ds["range"],
                              'variance': variance,
                              'noise_variance': noise_variance,
                              'atmos_variance': atmos_variance,
                              'w_25': w_25, 'w_min': w_min,
                              'w_50': w_50, 'w_max': w_max,
                              'w_75': w_75})

    return variance_ds

if __name__ == "__main__":
    ds_list = xr.concat([xr.open_dataset(x).sortby('time') for x in stares], dim='time').sortby('time')
    variance_ds = get_variances(ds_list)
    variance_ds.to_netcdf(output_path + 'crocus.wstats.%s.nc' % sys.argv[1])
