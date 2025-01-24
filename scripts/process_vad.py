import act
import glob
import os
import sys
import xarray as xr
import numpy as np

DEFAULT_SOURCE_PATH = '/lcrc/group/earthscience/rjackson/wfip3/caco/lidar_ingested/%s/' % sys.argv[1]
DEFAULT_OUTPUT_PATH = '/lcrc/group/earthscience/rjackson/wfip3/caco/vad/'
print(DEFAULT_SOURCE_PATH)
date = sys.argv[1]
file_list = glob.glob(DEFAULT_SOURCE_PATH + '*user5.nc')
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
vad_list.to_netcdf(os.path.join(DEFAULT_OUTPUT_PATH, 'caco.lidar.vad.%s.nc' % date))


    

