import xarray as xr

from scipy.signal import convolve2d


def output_vel_product(lidar_file):
    base, name = os.path.split(lidar_file)
    ds = xr.open_dataset(lidar_file)
    ds['radial_wind_velocity'] = ds['radial_wind_velocity'].where(ds['intensity'] > 1.008, drop=False)
    vel_variance = ds['radial_wind_velocity'].resample(time='30m').std(dim='time') 
    
    variance_bins = np.linspace(0, 2, 60)
    vel_bins = np.linspace(-10, 10, 60)
    hist_vel_variance = np.zeros((ds.sizes["range"], len(variance_bins) - 1))
    hist_vel = np.zeros((ds.sizes["range"], len(vel_bins) - 1))
    for i in range(ds.sizes["range"]):
        hist_vel_variance[i, :], bins = np.histogram(
            vel_variance.values[:, i], variance_bins)
        hist_vel[i, :], bins = np.histogram(
            ds["radial_wind_velocity"].values[:, i], vel_bins)

    hist_vel_variance = xr.DataArray(hist_vel_variance, dims=('range', 'bin_vel_variance'))
    hist_vel_variance.attrs["long_name"] = "Histogram of velocity variance"
    hist_vel_variance.attrs["units"] = "counts"
    hist_vel = xr.DataArray(hist_vel, dims=('range', 'bin_vel'))
    hist_vel.attrs["long_name"] = "Histogram of velocity"
    hist_vel.attrs["units"] = "counts"
    variance_bins = xr.DataArray(variance_bins, dims=('bin_vel_variance_edge'))
    variance_bins.attrs["long_name"] = "Velocity variance bin edges"
    variance_bins.attrs["units"] = "m-2 s-2"
    vel_bins = xr.DataArray(vel_bins, dims=('bin_vel_edge'))
    vel_bins.attrs["long_name"] = "Velocity bin edges"
    vel_bins.attrs["units"] = "m-1 s-1"
    variance_mids = xr.DataArray(
        (variance_bins[:-1] + variance_bins[1:]) / 2.0,
        dims=('bin_vel_variance'))
    variance_mids.attrs["long_name"] = "Velocity variance bin midpoints"
    variance_mids.attrs["units"] = "m-2 s-2"
    vel_mids = xr.DataArray((vel_bins[:-1] + vel_bins[1:]) / 2.0, dims=('bin_vel'))
    vel_mids.attrs["long_name"] = "Velocity bin midpoints"
    vel_mids.attrs["units"] = "m-1 s-1"
    out_ds_data = {'hist_vel_variance': hist_vel_variance,
                   'hist_vel': hist_vel,
                   'variance_bin_edges': variance_bins, 'vel_bin_edges': vel_bins,
                   'bin_vel_variance': variance_mids, 'bin_vel': vel_mids,
                   'range': ds['range'],
                   'time': ds['time'][0]}
    out_ds = xr.Dataset(out_ds_data)
    out_name = os.path.join(path, name[:-3] + '.freq.nc')
    out_ds.to_netcdf(out_name)
    ds.close()
