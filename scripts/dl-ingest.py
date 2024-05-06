import numpy as np
from netCDF4 import Dataset
import os
import datetime
import xarray as xr
import pandas as pd
import matplotlib.dates as mdates
import argparse
import matplotlib.pyplot as plt
import cmweather

from glob import glob
from datetime import datetime, timedelta
from distributed import LocalCluster, Client, wait

DEFAULT_SOURCE_PATH = '/lcrc/group/earthscience/rjackson/wfip3/caco/lidar/'
DEFAULT_OUTPUT_PATH = '/lcrc/group/earthscience/rjackson/wfip3/caco/lidar_ingested/'
DEFAULT_QUICKLOOKS_PATH = '/lcrc/group/earthscience/rjackson/wfip3/caco/quicklooks'

neiu_lat = 42 + 1/60 + 56.86/3600
neiu_lon = -70 - 3/60 - 12.37/3600
neiu_alt = 51.4
radial_vel_cal = 0.025

'''
Import of StreamLine .hpl (txt) files and save locally in directory. Therefore
the data is converted into matrices with dimension "number of range gates" x "time stamp/rays".
In newer versions of the StreamLine software, the spectral width can be 
stored as additional parameter in the .hpl files.
'''
def hpl2dict(file_path):
    #import hpl files into intercal storage
    with open(file_path, 'r') as text_file:
        lines=text_file.readlines()

    #write lines into Dictionary
    data_temp=dict()

    header_n=17 #length of header
    data_temp['filename']=lines[0].split()[-1]
    data_temp['system_id']=int(lines[1].split()[-1])
    data_temp['number_of_gates']=int(lines[2].split()[-1])
    data_temp['range_gate_length_m']=float(lines[3].split()[-1])
    data_temp['gate_length_pts']=int(lines[4].split()[-1])
    data_temp['pulses_per_ray']=int(lines[5].split()[-1])
    data_temp['number_of_waypoints_in_file']=int(lines[6].split()[-1])
    rays_n=(len(lines)-header_n)/(data_temp['number_of_gates']+1)
    
    '''
    number of lines does not match expected format if the number of range gates 
    was changed in the measuring period of the data file (especially possible for stare data)
    '''
    if not rays_n.is_integer():
        print('Number of lines does not match expected format')
        return np.nan
    
    data_temp['no_of_rays_in_file']=int(rays_n)
    data_temp['scan_type']=' '.join(lines[7].split()[2:])
    data_temp['focus_range']=lines[8].split()[-1]
    data_temp['start_time']=pd.to_datetime(' '.join(lines[9].split()[-2:]))
    data_temp['resolution']=('%s %s' % (lines[10].split()[-1],'m s-1'))
    data_temp['range_gates']=np.arange(0,data_temp['number_of_gates'])
    data_temp['center_of_gates']=(data_temp['range_gates']+0.5)*data_temp['range_gate_length_m']

    #dimensions of data set
    gates_n=data_temp['number_of_gates']
    rays_n=data_temp['no_of_rays_in_file']

    # item of measurement variables are predefined as symetric numpy arrays filled with NaN values
    data_temp['radial_velocity'] = np.full([gates_n,rays_n],np.nan) #m s-1
    data_temp['intensity'] = np.full([gates_n,rays_n],np.nan) #SNR+1
    data_temp['beta'] = np.full([gates_n,rays_n],np.nan) #m-1 sr-1
    data_temp['spectral_width'] = np.full([gates_n,rays_n],np.nan)
    data_temp['elevation'] = np.full(rays_n,np.nan) #degrees
    data_temp['azimuth'] = np.full(rays_n,np.nan) #degrees
    data_temp['decimal_time'] = np.full(rays_n,np.nan) #hours
    data_temp['pitch'] = np.full(rays_n,np.nan) #degrees
    data_temp['roll'] = np.full(rays_n,np.nan) #degrees
    
    for ri in range(0,rays_n): #loop rays
        lines_temp = lines[header_n+(ri*gates_n)+ri+1:header_n+(ri*gates_n)+gates_n+ri+1]
        header_temp = np.asarray(lines[header_n+(ri*gates_n)+ri].split(),dtype=float)
        data_temp['decimal_time'][ri] = header_temp[0]
        data_temp['azimuth'][ri] = header_temp[1]
        data_temp['elevation'][ri] = header_temp[2]
        data_temp['pitch'][ri] = header_temp[3]
        data_temp['roll'][ri] = header_temp[4]
        for gi in range(0,gates_n): #loop range gates
            line_temp=np.asarray(lines_temp[gi].split(),dtype=float)
            data_temp['radial_velocity'][gi,ri] = line_temp[1]
            data_temp['intensity'][gi,ri] = line_temp[2]
            data_temp['beta'][gi,ri] = line_temp[3]
            if line_temp.size>4:
                data_temp['spectral_width'][gi,ri] = line_temp[4]

    return data_temp


def convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta

def read_as_netcdf(file, lat, lon, alt, n_sweeps=1):
    field_dict = hpl2dict(file)
    initial_time = pd.to_datetime(field_dict['start_time'])

    time = pd.to_datetime([convert_to_hours_minutes_seconds(x, initial_time) for x in field_dict['decimal_time']])

    ds = xr.Dataset(coords={'range': field_dict['center_of_gates'],
                            'time': time,
                            'azimuth': ('time', field_dict['azimuth']),
                            'elevation': ('time', field_dict['elevation']),
                            'pitch': ('time', field_dict['pitch']),
                            'roll': ('time', field_dict['roll'])} ,
                    data_vars={'radial_velocity':(['time', 'range'],
                                                  field_dict['radial_velocity'].T),
                               'beta': (('time', 'range'),
                                        field_dict['beta'].T),
                               'intensity': (('time', 'range'),
                                             field_dict['intensity'].T),
                               'spectral_width': (('time', 'range'),
                               field_dict['spectral_width'].T)
                              }
                   )
    ds['azimuth'] = xr.where(ds['azimuth'] < 360., ds['azimuth'], ds['azimuth'] - 360.)
    ds["radial_velocity"].attrs["long_name"] = "Radial velocity of scatterers away from instrument."
    ds["radial_velocity"].attrs["standard_name"] = "radial_velocity_of_scatterers_away_from_instrument"
    ds["radial_velocity"].attrs["units"] = "m s-1"
    ds["spectral_width"].attrs["long_name"] = "Spectral width"
    ds["spectral_width"].attrs["standard_name"] = "spectral_width_of_radio_wave_in_air_scattered_by_air"
    ds["spectral_width"].attrs["units"] = "m s-1"
    ds["beta"].attrs["long_name"] = 'Backscatter coefficient'
    ds["beta"].attrs["units"] = "m-1 sr-1"
    ds["beta"].attrs["standard_name"] = 'backscattering_ratio'
    ds["latitude"] = lat
    ds["latitude"].attrs["long_name"] = 'latitude'
    ds["latitude"].attrs["standard_name"] = 'latitude'
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["latitude"].attrs["_FillValue"] = np.nan
    ds["longitude"] = lon
    ds["longitude"].attrs["long_name"] = 'longitude'
    ds["longitude"].attrs["standard_name"] = 'longitude'
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["longitude"].attrs["_FillValue"] = np.nan
    ds["pitch"].attrs["long_name"] = 'Pitch angle'
    ds["pitch"].attrs["standard_name"] = 'platform_pitch'
    ds["pitch"].attrs["units"] = "degrees"
    ds["pitch"].attrs["_FillValue"] = np.nan
    ds["roll"].attrs["long_name"] = 'Roll angle'
    ds["roll"].attrs["standard_name"] = 'platform_roll'
    ds["roll"].attrs["units"] = "degrees"
    ds["roll"].attrs["_FillValue"] = np.nan
    ds["azimuth"].attrs["long_name"] = 'Azimuth angle'
    ds["azimuth"].attrs["standard_name"] = 'sensor_azimuth_angle'
    ds["azimuth"].attrs["units"] = "degrees"
    ds["azimuth"].attrs["_FillValue"] = np.nan
    ds["elevation"].attrs["long_name"] = 'Elevation angle'
    ds["elevation"].attrs["units"] = "degrees"
    ds["elevation"].attrs["_FillValue"] = np.nan
    ds["range"].attrs["long_name"] = "Range from lidar"
    ds["range"].attrs["units"] = "meters"
    ds["range"].attrs["_FillValue"] = np.nan
    ds["altitude"] = alt
    ds["altitude"].attrs["long_name"] = "altitude"
    ds["altitude"].attrs["standard_name"] = "altitude"
    ds["altitude"].attrs["units"] = "meters"
    ds["altitude"].attrs["_FillValue"] = np.nan
    num_rays = ds.dims['time']
    ds.attrs["Conventions"] = "CF-1.7"
    ds.attrs["version"] = "R0"
    ds.attrs["mentor"] = "Bobby Jackson"
    ds.attrs['mentor_email'] = "rjackson@anl.gov"
    ds.attrs['mentor_institution'] = 'Argonne National Laboratory'
    ds.attrs['mentor_orcid'] = "0000-0003-2518-1234"
    ds.attrs['contributors'] = "Bobby Jackson, Scott Collis, Paytsar Muradyan, Max Grover, Joseph O'Brien"
    ds = ds.sortby('time')
    return ds

def process_file(fi):
    print("Processing %s" % fi)
    base, name = os.path.split(fi)
    scan_type = name.split(".")[-2]
    ds = read_as_netcdf(fi, neiu_lat, neiu_lon, neiu_alt)

    # Calibration of radial velocity
    ds["radial_velocity"] = ds["radial_velocity"] + radial_vel_cal
    date = name.split(".")[4]
    time = name.split(".")[5]
    ds.attrs["scan_type"] = scan_type
    out_name = 'dl.wfip3.caco.%s.%s.r0.nc' % (date, time)

    dest_path = os.path.join(args.dest_path, date)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    out_name = os.path.join(dest_path, out_name)
    print("Saving to %s" % out_name)
    ds.to_netcdf(out_name)
    quick_dir_path = os.path.join(args.quicklooks_path, date)
    if not os.path.exists(quick_dir_path):
        os.makedirs(quick_dir_path)
    quick_dir_path = os.path.join(quick_dir_path, '%s.%s.%s.png' % (scan_type, date, time))
    fig, ax = plt.subplots(3, 1, figsize=(10, 7))
    cbar_kwargs = {'label': '1 - log10(intensity)'}
    (1 - np.log10(ds['intensity'])).T.plot(ax=ax[0], cmap='ChaseSpectral', vmin=0, vmax=1,
        cbar_kwargs=cbar_kwargs)
    ds['radial_velocity'].T.plot(ax=ax[1], cmap='balance', vmin=-5, vmax=5)
    ds['spectral_width'].T.plot(ax=ax[2], cmap='Spectral_r', vmin=5, vmax=10)
    ax[0].set_ylim([0, 3000])
    ax[1].set_ylim([0, 3000])
    ax[2].set_ylim([0, 3000])

    fig.tight_layout()
    fig.savefig(quick_dir_path, bbox_inches='tight')
    plt.close(fig)
    del fig
    print("Quicklook for %s saved in %s!" % (out_name, quick_dir_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='HaloIngest',
            description='Halo photonics Doppler lidar R1-level ingest')
    parser.add_argument('--source_path', default=DEFAULT_SOURCE_PATH,
            help="Source path for .hpl files")
    parser.add_argument('--dest_path', default=DEFAULT_OUTPUT_PATH,
            help="Destination path for .nc files")
    parser.add_argument('--date', default=None, help="Date in YYYYMMDD, default is today")
    parser.add_argument('--quicklooks_path', default=DEFAULT_QUICKLOOKS_PATH,
            help="Destination path for quicklooks")
    args = parser.parse_args()
    date = args.date
    if date is None:
        input_list = glob(args.source_path + "/**/*.hpl", recursive=True)
    else:
        input_list = glob(args.source_path + "/**/*" + date + "*.hpl", recursive=True)
    with Client(LocalCluster(n_workers=16)) as c:
        results = c.map(process_file, input_list)
        wait(results)
