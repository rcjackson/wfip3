import pydsd
import xarray as xr
import glob
import sys
import os
import numpy as np
import dask.bag as db

from distributed import Client, LocalCluster
date = sys.argv[1]
data_dir = '/lcrc/group/earthscience/rjackson/wfip3/nant/parsivel_txt/'
out_dir = '/lcrc/group/earthscience/rjackson/wfip3/nant/parsivel/915mhz/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

file_list = sorted(glob.glob(data_dir + '*.%s.*' % date))
out_ds_list = []
#for fi in file_list:
def process_parsivel(fi):
    my_dsd = pydsd.read_parsivel(fi)
    print("PSD read in")
    # MRR2 Frequency is 24 Ghz
    # W-band is 95 Ghz
    my_dsd.set_scattering_temperature_and_frequency(scattering_freq=915e6)
    my_dsd.Nd["data"] = my_dsd.Nd["data"].filled(0)
    params_list = ["Zh", "Zdr", "delta_co", "Kdp", "Ai", "Adr", "D0", "Dmax", "Dm", "Nt", "Nw", "N0", "W", "mu", "Lambda"]
    my_dsd.calculate_radar_parameters()
    print("Scattering done")
    my_dsd.calculate_dsd_parameterization()
    out_ds = xr.Dataset()
    out_ds["time"] = ('time', my_dsd.time["data"])
    del my_dsd.time["data"]
    out_ds["bin_edges"] = ('bin_edges', my_dsd.bin_edges["data"])
    del my_dsd.bin_edges["data"]
    out_ds["bin_edges"].attrs = my_dsd.bin_edges

    out_ds["Nd"] = (['time', 'bins'], my_dsd.Nd["data"])
    del my_dsd.Nd["data"]
    out_ds["Nd"].attrs = my_dsd.Nd
 
    params_list = ["Zh", "Zdr", "delta_co", "Kdp", "Ai", "Adr", "D0", "Dmax", "Dm", "Nt", "Nw", "N0", "W", "mu", "Lambda"]
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
            os.path.join(out_dir, 'nant.parsivel.915mhz.%s.000000.nc' % date))









