import pandas as pd
import xarray as xr
import utils
import numpy as np
import sage_data_client
import requests
import sys
import os
import matplotlib.pyplot as plt
import cmweather

from datetime import datetime, timedelta

def plot_rhi(dataset, vel_key="radial_wind_speed", rng_key="distance"):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes()
    #dataset = dataset.where(dataset['intensity'] > 1.01)
    az_deg = dataset['azimuth'].values
    azi = np.deg2rad(dataset['azimuth'])
    el = np.deg2rad(dataset['elevation'])
    
    rng = dataset[rng_key]
    el, rng = np.meshgrid(el, rng, indexing='ij')
    x = rng * np.cos(el)
    y = rng * np.sin(el)
    c = ax.pcolormesh(x/1e3, y, dataset[vel_key].where(
        dataset['intensity'] > 1.008), cmap='balance', vmin=-20, vmax=20)
    ax.contourf(x/1e3, y, dataset['intensity'], levels=[1.3, np.inf])
    plt.colorbar(c, ax=ax, 
                 label='Radial velocity [m/s]', location='bottom')
    ax.set_ylim([0, 500])
    ax.set_xlim([-2, 2])
    #ax.set_rlim([0, 2.0])
    #ax.set_rticks([0, 0.25, .500, 0.75, 1.000, ])
    #ax.set_thetalim([np.deg2rad(5), np.deg2rad(15)])
    #ax.set_theta_zero_location("E") 
    ax.set_ylabel('Z [m]', labelpad=40)
    ax.set_xlabel('X [km]')
    ax.set_title(str(dataset['time'].values[0]) + f' {az_deg[0]:.1f} degrees')
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.tick_right()
    return fig

def convert_to_hours_minutes_seconds(decimal_hour, initial_time):
    delta = timedelta(hours=decimal_hour)
    return datetime(initial_time.year, initial_time.month, initial_time.day) + delta

def read_as_netcdf(file, lat, lon, alt, transition_threshold_azi=0.01,
                  transition_threshold_el=0.1, round_azi=1, round_el=1):
    field_dict = utils.hpl2dict(file)
    initial_time = pd.to_datetime(field_dict['start_time'])

    time = pd.to_datetime([convert_to_hours_minutes_seconds(x, initial_time) for x in field_dict['decimal_time']])

    ds = xr.Dataset(coords={'range':field_dict['center_of_gates'],
                            'time': time,
                            'azimuth': ('time', np.round(field_dict['azimuth'], round_azi)),
                            'elevation': ('time', np.round(field_dict['elevation'], round_el))} ,
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
    # Fake field for PYDDA
    ds['reflectivity'] = -99 * xr.ones_like(ds['beta'])
    ds['azimuth'] = xr.where(ds['azimuth'] >= 360.0, ds['azimuth'] - 360.0, ds['azimuth'])
    diff_azimuth = ds['azimuth'].diff(dim='time').values
    diff_elevation = np.pad(ds['elevation'].diff(dim='time').values, pad_width=(1, 0))
    unique_elevations = np.unique(ds["elevation"].where(diff_elevation <= transition_threshold_el))
    unique_elevations = unique_elevations[np.isfinite(unique_elevations)]
    counts = np.zeros_like(unique_elevations)
    
    for i in range(len(unique_elevations)):
        counts[i] = np.sum(ds["elevation"].values == unique_elevations[i])
    
    if np.sum(np.abs(diff_azimuth) > transition_threshold_azi) <= 2  and not np.all(ds['elevation'] == 90.0):
        sweep_mode = 'rhi'
        n_sweeps = 1
    elif np.all(ds['elevation'] == 90.0):
        sweep_mode = 'vertical_pointing'
        n_sweeps = 1
        unique_elevations = [90.0]
    else:
        # We will filter out the transitions between sweeps
        sweep_mode = "azimuth_surveillance"
        n_sweeps = len(unique_elevations)
        print(n_sweeps)
        
    ds['sweep_mode'] = xr.DataArray(np.array([sweep_mode.lower()], dtype='S32'), dims=['string_length_32'])
    ds['azimuth'] = xr.where(ds['azimuth'] < 360., ds['azimuth'], ds['azimuth'] - 360.)
    
    if sweep_mode == 'rhi':
        ds['fixed_angle'] = ('sweep', np.unique(ds['azimuth'].data[np.argwhere(np.abs(diff_azimuth) <= transition_threshold_azi) + 1]))
    elif sweep_mode == "azimuth_surveillance" or sweep_mode == "vertical_pointing":
        ds['fixed_angle'] = ('sweep', unique_elevations)
        
    ds['sweep_number'] = ('sweep', np.arange(0, n_sweeps))
    ds['sweep_number'].attrs["long_name"] = "sweep_index_number_0_based"
    ds['sweep_number'].attrs["units"] = ""
    ds['sweep_number'].attrs["_FillValue"] = -9999
    ds["latitude"] = lat
    ds["latitude"].attrs["long_name"] = 'latitude'
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["latitude"].attrs["_FillValue"] = -9999.
    ds["longitude"] = lon
    ds["longitude"].attrs["long_name"] = 'longitude'
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["longitude"].attrs["_FillValue"] = -9999.
    ds["altitude"] = alt
    ds["altitude"].attrs["long_name"] = alt
    ds["altitude"].attrs["units"] = "meters"
    ds["altitude"].attrs["_FillValue"] = -9999.
    num_rays = ds.dims['time']
    diff_elevation = ds["elevation"].diff(dim='time').values
    transitions = np.pad(np.abs(diff_elevation) > transition_threshold_el, (1, 0))
     
    ds["antenna_transition"] = ('time', transitions)
    ds["antenna_transition"].attrs["long_name"] = "antenna_transition"
    ds["antenna_transition"].attrs["units"] = "1 = transition, 0 = not"
    ds["antenna_transition"].attrs["_FillValue"] = -99.
    ds.attrs["Conventions"] = "CF-1.7"
    ds.attrs["version"] = "CF-Radial-1.4"
    return ds

def main(node, start_date, end_date):
    start_day = datetime.strptime(start_date, '%Y%m%d')
    end_day = datetime.strptime(end_date, '%Y%m%d')
    start_day = start_day.strftime("%Y-%m-%dT%H:%M:%S")
    end_day = end_day.strftime("%Y-%m-%dT%H:%M:%S")
    df_files = sage_data_client.query(
            start=start_day,
            end=end_day,
            filter={
                "name" : 'upload',
                "plugin": "registry.sagecontinuum.org/rjackson/lidar-control:2025.7.23",
                "vsn" : node,
            }
        )
    print(df_files.head())
    print(df_files["value"][0])
    df_strategy = sage_data_client.query(
            start=start_day,
            end=end_day,
            filter={
                "name" : 'lidar.strategy',
                "plugin": "registry.sagecontinuum.org/rjackson/lidar-control:2025.7.23",
                "vsn" : node,
            }
        )
    strategies = df_strategy["value"]
    print(strategies)
    strategy_time = df_strategy["timestamp"]
    for i, file_name in enumerate(df_files["value"]):
        time_diff = strategy_time - df_files["timestamp"][i]      
        ind_where = np.argmin(np.abs(time_diff)).squeeze()
        print(time_diff[ind_where])

        if time_diff[ind_where].total_seconds() > 300:
            continue
        if strategies[ind_where] == 0:
            print(f"Skipping {file_name}")
            continue
        base, name = os.path.split(file_name)
        output_name = name.split("-")[1]
        if os.path.exists(os.path.join(output_dir, output_name)):
            continue
        response = requests.get(file_name, auth=(username, password))
        base, name = os.path.split(file_name)
        if response.status_code == 200:
            # Save the file locally
            with open(os.path.join(output_dir, output_name), "wb") as file:
                file.write(response.content)
            print(f"{output_name} Downloaded!")
            try:
                my_nc = read_as_netcdf(os.path.join(output_dir, output_name),
                                       lat=41.2807, lon=70.1658, alt=0)
                my_nc.to_netcdf(os.path.join(output_dir, output_name[:-4] + '.nc'))
                print(f"{output_name} Processed!")
                fig = plot_rhi(my_nc, vel_key="radial_velocity", rng_key="range")
                fig.savefig(os.path.join(os.path.join(output_dir, 'png'),
                    output_name + '.png'), bbox_inches='tight')         
            except (ValueError, TypeError) as e:
                print(f"Processing {output_name} failed: {e}")

username = 'rjackson'         
password = '49GOS28FFE6I8REWMMD6'
node = sys.argv[1]
if node == "W0BE":
   out_data = "nant.lidar.z02.a1"
elif node == "W0C3":
   out_data = "bloc.lidar.z02.a1"
elif node == "W0C0":
   out_data = "ttp.lidar.z02.a1"
output_dir = f'/Volumes/Untitled/wfip3_adaptive_scanning_data/{out_data}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if __name__ == "__main__":
    node = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    main(node, start_date, end_date)
