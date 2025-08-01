import xarray as xr
import numpy as np 
import os

dealias_dates = ["20240609", "20240627", "20240818", "20240819"]
interference_dates =  ["20240809", "20240811", "20240921", "20240923", "20240924", "20240925", "20240926",
                    "20240927", "20240928", "20240929", "20240930"]

data_path = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_raw/'
out_path = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_cleaned/'

for day in dealias_dates:
    fname = os.path.join(data_path, f"barg.mrr.z01.b0.{day}.000000.nc")
    ds = xr.open_dataset(fname)
    w_attrs = ds["W"].attrs
    ds["W"] = xr.where(ds["W"] < 0, ds["W"] + 12, ds["W"])
    ds["W"].attrs = w_attrs
    fname = os.path.join(out_path, f"barg.mrr.z01.b0.{day}.000000.nc")
    ds.to_netcdf(fname)
    ds.close()
    print(f"{day} dealiased")

for day in interference_dates:
    fname = os.path.join(data_path, f"barg.mrr.z01.b0.{day}.000000.nc")
    ds = xr.open_dataset(fname)
    mask = ds["spectral_width"] > 0.2
    field_list = ["W", "spectral_width", "Skewness", "Kurtosis", "LWC",
            "RR", "SR", "Z", "Za", "Ze", "Nd_liquid", "Nd", "SNR",
            "Noise", "Nw", "Dm"]
    for f in field_list:
        ds[f] = ds[f].where(mask)

    fname = os.path.join(out_path, f"barg.mrr.z01.b0.{day}.000000.nc")
    ds.to_netcdf(fname)
    ds.close()
    print(f"{day} interference masked")    
