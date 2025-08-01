import pandas as pd
import xarray as xr
import glob
import os
import numpy as np

data_dir = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_cleaned/*.nc'
out_dir = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_final/'
file_list = glob.glob(data_dir)
lat_lon_csv = '/lcrc/group/earthscience/rjackson/wfip3/barg/WHOI_WFIP3_barge_bowstern_GPS_22-Oct-2024.csv'
lat_lon_df = pd.read_csv(lat_lon_csv, index_col='time', parse_dates=True).to_xarray()
print(lat_lon_df)
for fi in file_list:
    dataset = xr.open_dataset(fi)
    print(dataset)
    print(lat_lon_df.time)
    lat = lat_lon_df["stern_lat"].reindex(
            time=dataset["time"].values, method='nearest', tolerance=pd.Timedelta("5min"))
    lon = lat_lon_df["stern_lon"].reindex(
            time=dataset["time"].values, method='nearest', tolerance=pd.Timedelta("5min"))
    print(lat.values)
    print(lon.values)
    dataset["lat"] = (["time"], lat.values)
    dataset["lat"].attrs["long_name"] = "latitude"
    dataset["lat"].attrs["standard_name"] = "latitude"
    dataset["lat"].attrs["units"] = "degree"
    dataset["lon"] = (["time"], lon.values)
    dataset["lon"].attrs["long_name"] = "longitude"
    dataset["lon"].attrs["standard_name"] = "longitude"
    dataset["lon"].attrs["units"] = "degree"
    path, name = os.path.split(fi)
    dataset.to_netcdf(os.path.join(out_dir, name))
    print(f"{name} processed")
