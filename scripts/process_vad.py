import act
import glob
import os
import sys
import xarray as xr
import numpy as np

DEFAULT_SOURCE_PATH = '/lcrc/group/earthscience/rjackson/dl-sparc-canyons/netcdf/%s/' % sys.argv[1]
DEFAULT_OUTPUT_PATH = '/lcrc/group/earthscience/rjackson/dl-sparc-canyons/vad/'
print(DEFAULT_SOURCE_PATH)
date = sys.argv[1]
file_list = glob.glob(DEFAULT_SOURCE_PATH + '*VAD*.nc')
def process_vad(file_name):
    try:
        ds = xr.open_dataset(file_name)
    except OSError:
        return None
    if "scan_type" not in ds.attrs.keys():
        ds.close()
        return None
    if ds.attrs["scan_type"] == "fpt":
        ds.close()
        return None
    try:
        dataset = act.retrievals.compute_winds_from_ppi(ds)
        ds.close()
    except (np.linalg.LinAlgError, KeyError):
        return None
    return dataset

vad_list0 = [process_vad(f) for f in file_list]
vad_list = []
for x in vad_list0:
   if x is not None:
       vad_list.append(x)
vad_list = xr.concat(vad_list, dim='time').sortby('time')
vad_list.to_netcdf(os.path.join(DEFAULT_OUTPUT_PATH, 'canyons.lidar.vad.%s.nc' % date))


    

