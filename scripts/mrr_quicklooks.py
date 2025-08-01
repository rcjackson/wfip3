import matplotlib.pyplot as plt 
import xarray as xr
import cmweather
import glob
import numpy
import os
import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter

if __name__ == "__main__":
    mrr_path = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_raprom/*.nc'
    mrr_quicklook_path = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr_raprom/quicklooks/'
    if not os.path.exists(mrr_quicklook_path):
        os.makedirs(mrr_quicklook_path)
        
    mrr_variables = ["Ze" , "W", "spectral width"]
    mrr_colormaps = ["ChaseSpectral", "balance", "Spectral_r"]
    mrr_limits = [(0, 50), (-12, 12), (0, 10)]
    n_variables = len(mrr_variables) + 1
    mrr_files = glob.glob(mrr_path)
    for mrr_file in mrr_files:
        base, name = os.path.split(mrr_file)
        out_png_name = os.path.join(mrr_quicklook_path, name[:-3] + '.png')
        fig, ax = plt.subplots(n_variables, 1, figsize=(15, 3*n_variables))
        mrr_ds = xr.open_dataset(mrr_file)
        mrr_ds["time"] = mrr_ds["time"].astype("datetime64[ns]")
        for i in range(n_variables):
            if i < n_variables - 1:
                variable = mrr_variables[i]
                cmap = mrr_colormaps[i]
                vmin, vmax = mrr_limits[i]
                mrr_ds[variable].T.plot(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax[i])
            else:
                variable = "Type"
                cat_colors = {'Hail': 'yellow',
                    'Snow': 'cyan',
                    'Mixed': 'white',
                    'Mixed': 'red',
                    'Drizzle': 'blue',
                    'Rain': 'green',
                    'Rain': 'black',
                    'Unknown': 'brown'}
                lab_colors = [list(cat_colors.values())]
                cmap = 'tab10'
                vmin = -20
                vmax = 20
                
                # Even out spacing of categories
                tick_locs = np.array([-20, -10, 0, 5, 10, 20]) 
                locator = FixedLocator(tick_locs)
                catty_list = ["Hail", "Snow", "Mixed", "Drizzle", "Rain", "Unknown"]
                formatter = FixedFormatter(catty_list)
                cbar_kwargs = {'format': formatter, 'ticks': locator}
                mrr_ds[variable].T.plot(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax[i], cbar_kwargs=cbar_kwargs)
            
            ax[i].set_ylim([0, 5000])
        fig.tight_layout()
        fig.savefig(out_png_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Quicklooks for {mrr_file} complete!")
        mrr_ds.close()