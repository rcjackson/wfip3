import scipy
import xarray as xr
import numpy as np
import argparse
import glob

from scipy.interpolate import griddata
from distributed import Client, LocalCluster, wait
from pyart.core.transforms import antenna_to_cartesian

lidar_data_path = '/Volumes/Untitled/leaff/nc/'
gridded_data_path = '/Volumes/Untitled/leaff/gridded/'

def return_gridded_radial_velocity(file_name, dx=30, dy=30., dz=30., snr_threshold=0.008,
                                    vel_key='radial_velocity', rng_key='range',
                                      min_x=-2000, max_x=2000, min_y=-2000, max_y=2000,
                                      min_z=0, max_z=1000, verbose=False):
    """
    Returns gridded radial velocity from a Lidar PPI scan on a regular grid.

    Parameters
    ----------
    file_name : str
        The path to the NetCDF file containing the Lidar PPI data.
    dx : float, optional
        The x grid spacing in meters. Default is 30 m.
    dy : float, optional
        The y grid spacing in meters. Default is 30 m.
    dz : float, optional
        The vertical grid spacing in meters. Default is 30 m.
    snr_threshold : float, optional
        The signal-to-noise ratio threshold for masking low-quality data. Default is 0.008.
    vel_key : str, optional
        The key in the dataset for the radial velocity. Default is 'radial_velocity'.
    rng_key : str, optional
        The key in the dataset for the range. Default is 'range'.
    min_x, max_x : float, optional
        The x extent of the output grid in meters. Default is -2000 to 2000 m.
    min_y, max_y : float, optional
        The y extent of the output grid in meters. Default is -2000 to 2000 m.
    min_z, max_z : float, optional
        The vertical extent of the output grid in meters. Default is 0 to 1000 m.

    Returns
    -------
    out_dataset : xarray.Dataset
        The gridded radial velocity on a regular grid.
    """
    if verbose:
        print(f"Processing file: {file_name}")
    try:
        with xr.open_dataset(file_name) as dataset:
            az = dataset['azimuth'].values
            el = dataset['elevation'].values
            if len(np.unique(el)) > 10 or len(np.unique(az)) < 5:
                if verbose:
                    print("Skipping file due to insufficient azimuthal coverage or too many elevation angles.")
                return None
            mask = dataset['intensity'] > (1 + snr_threshold)
            
            nrays = len(az)
            rng = dataset[rng_key].values
            ngates = len(rng)
            rng = np.tile(rng, (nrays, 1))

            az = np.tile(az[:, np.newaxis], (1, ngates))
            el = np.tile(el[:, np.newaxis], (1, ngates))
            x, y, z = antenna_to_cartesian(rng/1e3, az, el)
            dataset["radial_wind_speed"] = dataset["radial_wind_speed"].where(mask)
            if verbose:
                print("Data shapes - x:", x.shape, "y:", y.shape, "z:", z.shape, "radial_wind_speed:", dataset["radial_wind_speed"].shape)
                print("Number of valid points after masking:", np.sum(~np.isnan(dataset["radial_wind_speed"].values)))
            
            # Create a regular grid
            x_grid = np.arange(min_x, max_x, dx)
            y_grid = np.arange(min_y, max_y, dy)
            z_grid = np.arange(min_z, max_z, dz)
            z_grid, y_grid, x_grid = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')
            if verbose:
                print("Minimum and maximum of x, y, z:", np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y), np.nanmin(z), np.nanmax(z))
            
            # Interpolate the streamwise velocity onto the regular grid
            from scipy.interpolate import griddata
            # Remove data points outside of minimum and maximum x
            x_mask = np.logical_and(x >= min_x, x <= max_x)
            y_mask = np.logical_and(y >= min_y, y <= max_y)
            z_mask = np.logical_and(z >= min_z, z <= max_z)
            mask = np.logical_and(x_mask, y_mask)
            mask = np.logical_and(mask, z_mask)
            x = x[mask]
            y = y[mask]
            z = z[mask]
            if verbose:
                print("Number of valid points after applying spatial mask:", len(x))
            points = np.column_stack((z.flatten(), y.flatten(), x.flatten()))
            values = dataset["radial_wind_speed"].values.flatten()
            values = values[mask.flatten()]
            points = points[~np.isnan(values)]
            values = values[~np.isnan(values)]
            try:
                radial_velocity_grid = griddata(points, values, (z_grid, y_grid, x_grid), method='linear')
            except Exception as e:
                if verbose:
                    print("Error interpolating radial velocity data.")
                    print("Error details:", e)
                return None
            if verbose:
                print("Number of valid points after interpolation:", np.sum(~np.isnan(radial_velocity_grid)))
        
            if verbose:
                print("Radial velocity grid shape:", radial_velocity_grid.shape)
            radial_velocity = xr.DataArray(radial_velocity_grid,
                                            coords=[z_grid[:, 0, 0], y_grid[0, :, 0], x_grid[0, 0, :]], dims=['z', 'y', 'x'])
            radial_velocity.attrs['units'] = 'm/s'
            radial_velocity.attrs['long_name'] = 'Radial Velocity'
            out_dataset = xr.Dataset({'radial_velocity': radial_velocity.astype(np.float32),
                                    'azimuth': ('time', [az.mean()]),
                                    'time': ('time', [dataset['time'].values[0]])})
            out_dataset['azimuth'].attrs['units'] = 'degrees'
            out_dataset['azimuth'].attrs['long_name'] = 'Mean Azimuth Angle'
    except ValueError as e:
        if verbose:
            print(f"Error processing file {file_name}. Skipping.")
            print("Error message:", e)
        return None
        
    return out_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Lidar PPI data to compute gridded radial velocity.')
    parser.add_argument('--input_dir', type=str, 
                        help='Path to the input directory containing NetCDF files with Lidar RHI data.',
                        default=lidar_data_path)
    parser.add_argument('--output_dir', type=str, 
                        help='Path to the output directory to save the gridded streamwise velocity NetCDF files.',
                        default=gridded_data_path)
    parser.add_argument('--date', type=str, default=None,
                        help='Date string (YYYYMMDD) to filter files for processing. Default is None')
    parser.add_argument('--dx', type=float, default=30,
                        help='X grid spacing in meters. Default is 30 m.')
    parser.add_argument('--dy', type=float, default=30,
                        help='Y grid spacing in meters. Default is 30 m.')
    parser.add_argument('--dz', type=float, default=30,
                        help='Vertical grid spacing in meters. Default is 30 m.')
    parser.add_argument('--snr_threshold', type=float, default=0.008,
                        help='Signal-to-noise ratio threshold for masking low-quality data. Default is 0.008.')
    parser.add_argument('--elevation', type=float, default=5.,
                        help='Elevation angle in degrees to process. Default is 5 degrees.')
    parser.add_argument('--vel_key', type=str, default='radial_velocity',
                        help='The key in the dataset for the radial velocity. Default is "radial_velocity."')
    parser.add_argument('--rng_key', type=str, default='range',
                        help='The key in the dataset for the range. Default is "range."')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for parallel processing. Default is 4.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output during processing.')
    parser.add_argument('--min_x', type=float, default=-2000,
                        help='Minimum x value for the output grid. Default is -2000 m.')
    parser.add_argument('--max_x', type=float, default=2000,
                        help='Maximum x value for the output grid. Default is 2000 m.')
    parser.add_argument('--min_y', type=float, default=-2000,
                        help='Minimum y value for the output grid. Default is -2000 m.')
    parser.add_argument('--max_y', type=float, default=2000,
                        help='Maximum y value for the output grid. Default is 2000 m.')
    parser.add_argument('--min_z', type=float, default=0,
                        help='Minimum z value for the output grid. Default is 0 m.')
    parser.add_argument('--max_z', type=float, default=1000,
                        help='Maximum z value for the output grid. Default is 1000 m.')
    args = parser.parse_args()

    # Load the dataset
    if args.date is None:
        file_list = glob.glob(f"{args.input_dir}/**/*user1.nc", recursive=True)
    else:
        file_list = glob.glob(f"{args.input_dir}/**/*{args.date}*user1.nc", recursive=True)
    if args.num_workers > 1:
        with LocalCluster(n_workers=args.num_workers) as cluster:
            with Client(cluster) as client:
                futures = [client.submit(return_gridded_radial_velocity, file_name, args.dx, args.dy, args.dz,
                                         args.snr_threshold, 
                                         args.vel_key, args.rng_key,
                                         args.min_x, args.max_x, args.min_y, args.max_y,
                                         args.min_z, args.max_z, args.verbose) for file_name in file_list]
                results = client.gather(futures)
    else:
        results = [return_gridded_radial_velocity(file_name, args.dx, args.dy, args.dz,
                                                   args.snr_threshold, 
                                                   args.vel_key, args.rng_key,
                                                   args.min_x, args.max_x, args.min_y, args.max_y,
                                                   args.min_z, args.max_z, args.verbose)
                   for file_name in file_list]
    results = [x for x in results if x is not None]    
    radial_velocity = xr.concat(results, dim='time').sortby('time')
    out_file_name = file_list[0].split('/')[-1].replace('.nc', '_radial_velocity.nc')
    args.output_file = f"{args.output_dir}/{out_file_name}"
    # Save the gridded streamwise velocity to a new NetCDF file
    radial_velocity.to_netcdf(args.output_file)
