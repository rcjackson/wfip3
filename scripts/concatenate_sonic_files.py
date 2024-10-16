import xarray as xr
import glob 
import sys

which = sys.argv[1] 
date = sys.argv[2]
path = '/lcrc/group/earthscience/rjackson/wfip3/caco/sonic/caco.sonic.%s.b1.%s*.nc' % (which, date)
out_path = '/lcrc/group/earthscience/rjackson/wfip3/caco/sonic/caco.sonic.%s.b1.daily.%s.nc' % (which, date)
ds = xr.open_mfdataset(path)
ds.to_netcdf(out_path)
ds.close()

