import pydsd
import xarray as xr
import glob
import sys
import os
import numpy as np
import dask.bag as db

from distributed import Client, LocalCluster
date = sys.argv[1]
site = sys.argv[2]
data_dir = f'/lcrc/group/earthscience/rjackson/wfip3/barg/parsivel/raw/{date}/'
out_dir = '/lcrc/group/earthscience/rjackson/wfip3/barg/parsivel/b0/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file_list = sorted(glob.glob(data_dir + f"barg.ld.{site}.00.{date}*"))
print(file_list)
out_ds_list = []
#for fi in file_list:
def process_parsivel(fi):
    my_dsd = pydsd.read_parsivel(fi)
    print("PSD read in")
    # MRR2 Frequency is 24 Ghz
    # W-band is 95 Ghz
    my_dsd.set_scattering_temperature_and_frequency(scattering_freq=915e6)
    my_dsd.Nd["data"] = my_dsd.Nd["data"].filled(0)
    
    params_list = ["D0", "Dmax", "Dm", "Nt", "Nw", "N0", "W", "mu", "Lambda", "rainfall_rate"]
    #my_dsd.calculate_radar_parameters()
    print("Scattering done")
    #my_dsd.calculate_dsd_parameterization()
    out_ds = xr.Dataset()
    out_ds["time"] = ('time', my_dsd.time["data"])
    out_ds["time"] = out_ds["time"].astype("datetime64[s]")
    out_ds["bin_edges"] = ('bin_edges', my_dsd.bin_edges["data"])
    out_ds["bin_edges"].attrs = my_dsd.bin_edges
    out_ds["bins"] = ('bins', (
        my_dsd.bin_edges["data"][1:] + my_dsd.bin_edges["data"][:-1]) / 2.)
    out_ds["bins"].attrs = {"long_name": "Diameter bin midpoints",
            "units": "mm"}
    del out_ds["bin_edges"].attrs["data"]
    out_ds["Nd"] = (['time', 'bins'], my_dsd.Nd["data"])
    out_ds["Nd"].attrs = my_dsd.Nd
    del out_ds["Nd"].attrs["data"]    
    out_ds["Vd"] = (['time', 'bins'], my_dsd.fields["terminal_velocity"]["data"])
    out_ds["Vd"].attrs = my_dsd.fields["terminal_velocity"]
    del out_ds["Vd"].attrs["data"] 
    out_ds["Vd"].attrs["long_name"] = "Fall speed spectrum of drops"
    out_ds["Vd"].attrs["standard_name"] = "Fall Speed Spectra"
    out_ds["rain_rate"] = (['time'], my_dsd.rain_rate["data"])
    out_ds["rain_rate"].attrs = my_dsd.rain_rate
    del out_ds["rain_rate"].attrs["data"] 
    
    # Do QC - remove drops based off of fall speeds
    terminal_velocity = my_dsd.calculate_fall_speed(out_ds["bins"].values)
    finite_inds = np.argwhere(out_ds["Vd"].values[0, :] > 0)
    finite_inds = finite_inds[1:]
    print(np.abs(terminal_velocity - out_ds["Vd"].values[0, finite_inds]) / terminal_velocity[finite_inds])
    if np.any(np.logical_or(
        np.abs(out_ds["Vd"].values[0, finite_inds] < 0.5 * terminal_velocity[finite_inds]),
        np.abs(out_ds["Vd"].values[0, finite_inds] > 1.5 * terminal_velocity[finite_inds]))):
        print("Velocities out of range")
        out_ds["Nd"][:] = np.nan
        out_ds["rain_rate"][:] = np.nan
        
    #my_dsd.calculate_radar_parameters()
    my_dsd.calculate_dsd_parameterization()
    params_list = ["D0", "Dmax", "Dm", "Nt", "Nw", 
                   "N0", "W", "mu", "Lambda",
                   "weather_code_metar", "weather_code_nws",
                   "visibility_mor", "laserband_amplitude",
                   "sensor_temperature", "heating_current",
                   "sensor_voltage"]
    for param in params_list:
        out_ds[param] = (['time'], my_dsd.fields[param]["data"])
        del my_dsd.fields[param]["data"]
        out_ds[param].attrs = my_dsd.fields[param]
    print(out_ds)
    return out_ds

if __name__ == "__main__":
    with Client(LocalCluster(n_workers=18)) as c:
        bag = db.from_sequence(file_list)
        print("Processing %d files." % len(file_list))
        #out_ds_list = [process_parsivel(x) for x in file_list]
        out_ds_list = bag.map(process_parsivel).compute()
        out_ds = xr.concat(out_ds_list, dim="time")
        out_ds.to_netcdf(
            os.path.join(out_dir, f"barg.ld.{site}.b0.{date}.000000.nc"))









