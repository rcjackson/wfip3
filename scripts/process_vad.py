import act
import glob
import os
import sys
import xarray as xr
import numpy as np

file_date = sys.argv[1]
DEFAULT_SOURCE_PATH = f'/Volumes/Untitled/wfip3_adaptive_scanning_data/bloc.lidar.z01.a0/*{file_date}*'
DEFAULT_OUTPUT_PATH = '/Volumes/Untitled/wfip3_adaptive_scanning_data/bloc_lidar_profiles/'
print(DEFAULT_SOURCE_PATH)
date = sys.argv[1]
file_list = glob.glob(DEFAULT_SOURCE_PATH + '*user1.nc')
def process_vad(file_name):
    try:
        ds = xr.open_dataset(file_name)
    except OSError:
        return None
    try:
        dataset = act.retrievals.compute_winds_from_ppi(ds,
                snr_name="SNR", intensity_name="intensity", 
                radial_velocity_name="radial_wind_speed")
        ds.close()
    except (np.linalg.LinAlgError, KeyError):
        return None
    print(file_name + " processed")
    return dataset

vad_list0 = [process_vad(f) for f in file_list]
vad_list = []
for x in vad_list0:
   if x is not None:
       vad_list.append(x)
vad_list = xr.concat(vad_list, dim='time').sortby('time')
vad_list.to_netcdf(os.path.join(DEFAULT_OUTPUT_PATH, 'bloc.lidar.vad.%s.nc' % date))


    

