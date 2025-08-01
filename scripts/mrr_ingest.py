import xradar as xd
import xarray as xr
import glob

input_path = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr/'

file_list = glob.glob(input_path + '*202409*.pro')

for fi in file_list:
    ds = xr.open_dataset(fi, engine='metek')
    print(ds)
    del ds["time"].attrs["units"]
    ds.to_netcdf(fi[:-4] + '.nc')
    ds.close()
    
    
