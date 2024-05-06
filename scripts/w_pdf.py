# Code to mass process vertical velocity p.d.f.s from Doppler lidar data
import xarray as xr
import numpy as np
import os
import radtraq
import glob

from scipy.signal import convolve2d
path = '/lcrc/group/earthscience/rjackson/wfip3/caco/pdfs/'
in_path = '/lcrc/group/earthscience/rjackson/wfip3/caco/lidar_ingested/'
window_size = (3, 3)
def output_vel_product(lidar_file):
    base, name = os.path.split(lidar_file)
    ds = xr.open_dataset(lidar_file)
    print(ds.attrs["scan_type"])
    if not ds.attrs["scan_type"] == "fpt":
        ds.close()
        return
    print(ds['radial_velocity'])
    ds['radial_velocity'] = ds['radial_velocity'].where(ds['intensity'] > 1.008, drop=False)
    kernel = 1 / np.prod(window_size) * np.ones(window_size)
    print(kernel.shape)
    print(ds['radial_velocity'])
    vel_window_mean = convolve2d(ds['radial_velocity'], kernel, boundary='symm', mode='same')
    print(vel_window_mean.shape)
    ds['vel_variance'] = (['time', 'range'], convolve2d(
        (ds['radial_velocity'] - vel_window_mean) ** 2, np.ones(window_size),
        boundary='symm', mode='same'))
    
    variance_bins = np.linspace(0, 2, 30)
    vel_bins = np.linspace(-20, 20, 40)
    hist_vel_variance = np.zeros((ds.dims["range"], len(variance_bins) - 1))
    hist_vel = np.zeros((ds.dims["range"], len(vel_bins) - 1))
    for i in range(ds.dims["range"]):
        hist_vel_variance[i, :], bins = np.histogram(ds["vel_variance"].values[:, i], variance_bins)
        hist_vel[i, :], bins = np.histogram(ds["radial_velocity"].values[:, i], vel_bins)
    
    hist_vel_variance = xr.DataArray(hist_vel_variance, dims=('range', 'hist_vel_variance'))
    hist_vel_variance.attrs["long_name"] = "Histogram of velocity variance"
    hist_vel_variance.attrs["units"] = "counts"
    hist_vel = xr.DataArray(hist_vel, dims=('range', 'hist_vel'))
    hist_vel.attrs["long_name"] = "Histogram of velocity"
    hist_vel.attrs["units"] = "counts"
    variance_bins = xr.DataArray(variance_bins, dims=('bin_vel_variance'))
    variance_bins.attrs["long_name"] = "Velocity variance bin edges"
    variance_bins.attrs["units"] = "m-2 s-2"
    vel_bins = xr.DataArray(vel_bins, dims=('bin_vel'))
    vel_bins.attrs["long_name"] = "Velocity bin edges"
    vel_bins.attrs["units"] = "m-1 s-1"
    out_ds_data = {'hist_vel_variance': hist_vel_variance,
                   'hist_vel': hist_vel, 
                   'variance_bins': variance_bins, 'vel_bins': vel_bins}
    out_ds = xr.Dataset(out_ds_data)
    out_name = os.path.join(path, name[:-3] + '.freq.nc')
    out_ds.to_netcdf(out_name)
    ds.close()

file_list = sorted(glob.glob(in_path + '/**/*.nc', recursive=True))
for i in range(1):
    print(file_list[i])
    output_vel_product(file_list[i])
