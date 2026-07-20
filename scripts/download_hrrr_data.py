"""
Download HRRR analysis (f00) data at fixed lat/lon points and save monthly
netCDF files per site.

Pulls two HRRR products per hourly timestep via Herbie (byte-range subset,
not full-file downloads):
  - sfc: cloud base/ceiling/top height, terrain height, low cloud cover,
    surface precip rate, surface pressure
  - prs: geopotential height, rain water mixing ratio, temperature, specific
    humidity, u/v wind, and vertical velocity (VVEL, Pa/s) on the 1000-850 mb
    standard pressure levels

Output is resumable: each calendar month is written to its own netCDF file,
and months whose output file already exists are skipped on a re-run.
"""
import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from herbie import Herbie

_grid_index_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SITES = {
    "NANT": (41.242, -70.125),
    "BARG": (40.900, -70.783),
}

PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 875, 850]  # mb, surface to 850 mb

SFC_SEARCH = (
    ":(HGT:(cloud base|cloud ceiling|cloud top|surface)"
    "|LCDC:low cloud layer|PRATE:surface|PRES:surface):"
)
PRS_SEARCH = ":(HGT|TMP|SPFH|UGRD|VGRD|RWMR|VVEL):(%s) mb:" % "|".join(
    str(level) for level in PRESSURE_LEVELS
)

# (cfgrib shortName, GRIB_typeOfLevel) -> (output name, long name, units)
SFC_VAR_MAP = {
    ("gh", "cloudBase"): ("cloud_base_height", "cloud base height", "m"),
    ("gh", "cloudCeiling"): ("cloud_ceiling_height", "cloud ceiling height", "m"),
    ("gh", "cloudTop"): ("cloud_top_height", "cloud top height", "m"),
    ("orog", "surface"): ("terrain_height", "surface (terrain) height", "m"),
    ("lcc", "lowCloudLayer"): ("low_cloud_cover", "low cloud cover", "%"),
    ("prate", "surface"): ("precip_rate", "surface precipitation rate", "kg m-2 s-1"),
    ("sp", "surface"): ("surface_pressure", "surface pressure", "Pa"),
}

# cfgrib shortName -> (output name, long name, units)
PRS_VAR_MAP = {
    "gh": ("geopotential_height", "geopotential height", "m"),
    "rwmr": ("rain_water_mixing_ratio", "rain water mixing ratio", "kg kg-1"),
    "t": ("temperature", "temperature", "K"),
    "q": ("specific_humidity", "specific humidity", "kg kg-1"),
    "u": ("u_wind", "u-component of wind", "m s-1"),
    "v": ("v_wind", "v-component of wind", "m s-1"),
    "w": (
        "vertical_velocity",
        "vertical velocity (pressure velocity, omega)",
        "Pa s-1",
    ),
}


def find_nearest_index(lat2d, lon2d, site_lat, site_lon):
    """Return (j, i) index of the HRRR grid cell nearest to (site_lat, site_lon)."""
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    dist2 = (lat2d - site_lat) ** 2 + (lon2d - site_lon) ** 2
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(j), int(i)


def _as_dataset_list(ds_or_list):
    return ds_or_list if isinstance(ds_or_list, list) else [ds_or_list]


def extract_sfc_point(ds_or_list, j, i):
    """Extract the SFC_VAR_MAP variables at grid cell (j, i) as a flat dict."""
    values = {}
    for ds in _as_dataset_list(ds_or_list):
        for name, da in ds.data_vars.items():
            key = (name, da.attrs.get("GRIB_typeOfLevel"))
            if key not in SFC_VAR_MAP:
                continue
            out_name = SFC_VAR_MAP[key][0]
            values[out_name] = float(da.values[j, i])
    return values


def extract_prs_point(ds, j, i):
    """Extract the PRS_VAR_MAP profile variables at grid cell (j, i) as a flat dict of arrays."""
    values = {}
    point = ds.isel(y=j, x=i)
    for name, da in point.data_vars.items():
        if name not in PRS_VAR_MAP:
            continue
        out_name = PRS_VAR_MAP[name][0]
        values[out_name] = da.sortby("isobaricInhPa", ascending=False).values
    return values


def download_timestep(timestamp, grid_indices):
    """
    Download one hourly HRRR analysis timestep and extract point values for
    every site in SITES.

    grid_indices is a mutable {product: {site: (j, i)}} cache; it is filled in
    lazily from the first successful download of each product.

    Returns {site: {var_name: value_or_array}}, or None if both downloads fail.
    """
    site_records = {site: {} for site in SITES}

    try:
        H_sfc = Herbie(timestamp, model="hrrr", product="sfc", fxx=0)
        ds_sfc = H_sfc.xarray(SFC_SEARCH, remove_grib=True)
        sfc_list = _as_dataset_list(ds_sfc)
        if "sfc" not in grid_indices:
            with _grid_index_lock:
                if "sfc" not in grid_indices:
                    lat2d, lon2d = sfc_list[0].latitude.values, sfc_list[0].longitude.values
                    grid_indices["sfc"] = {
                        site: find_nearest_index(lat2d, lon2d, lat, lon)
                        for site, (lat, lon) in SITES.items()
                    }
        for site in SITES:
            j, i = grid_indices["sfc"][site]
            site_records[site].update(extract_sfc_point(sfc_list, j, i))
    except Exception as exc:
        logger.warning("sfc download failed for %s: %s", timestamp, exc)

    try:
        H_prs = Herbie(timestamp, model="hrrr", product="prs", fxx=0)
        ds_prs = H_prs.xarray(PRS_SEARCH, remove_grib=True)
        if "prs" not in grid_indices:
            with _grid_index_lock:
                if "prs" not in grid_indices:
                    lat2d, lon2d = ds_prs.latitude.values, ds_prs.longitude.values
                    grid_indices["prs"] = {
                        site: find_nearest_index(lat2d, lon2d, lat, lon)
                        for site, (lat, lon) in SITES.items()
                    }
        for site in SITES:
            j, i = grid_indices["prs"][site]
            site_records[site].update(extract_prs_point(ds_prs, j, i))
    except Exception as exc:
        logger.warning("prs download failed for %s: %s", timestamp, exc)

    if all(not record for record in site_records.values()):
        return None
    return site_records


def build_site_dataset(site, times, records, grid_indices):
    """Assemble one xr.Dataset for a site from a list of per-timestep dicts."""
    all_vars = set(SFC_VAR_MAP[k][0] for k in SFC_VAR_MAP) | set(
        v[0] for v in PRS_VAR_MAP.values()
    )
    data_vars = {}
    for out_name in all_vars:
        is_profile = out_name in [v[0] for v in PRS_VAR_MAP.values()]
        if is_profile:
            arr = np.full((len(times), len(PRESSURE_LEVELS)), np.nan)
            for t_idx, record in enumerate(records):
                if out_name in record:
                    arr[t_idx, :] = record[out_name]
            data_vars[out_name] = (("time", "level"), arr)
        else:
            arr = np.full(len(times), np.nan)
            for t_idx, record in enumerate(records):
                if out_name in record:
                    arr[t_idx] = record[out_name]
            data_vars[out_name] = (("time",), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "level": PRESSURE_LEVELS},
    )

    long_name_units = {v[0]: (v[1], v[2]) for v in SFC_VAR_MAP.values()}
    long_name_units.update({v[0]: (v[1], v[2]) for v in PRS_VAR_MAP.values()})
    for name, (long_name, units) in long_name_units.items():
        ds[name].attrs["long_name"] = long_name
        ds[name].attrs["units"] = units

    ds["level"].attrs = {"long_name": "pressure level", "units": "mb"}

    nominal_lat, nominal_lon = SITES[site]
    ds.attrs["site"] = site
    ds.attrs["nominal_latitude"] = nominal_lat
    ds.attrs["nominal_longitude"] = nominal_lon
    for product in ("sfc", "prs"):
        if product in grid_indices and site in grid_indices[product]:
            j, i = grid_indices[product][site]
            ds.attrs[f"{product}_grid_index_j"] = j
            ds.attrs[f"{product}_grid_index_i"] = i
    ds.attrs["source"] = "NOAA HRRR analysis (fxx=0), downloaded via Herbie"
    ds.attrs["description"] = (
        "vertical_velocity is HRRR VVEL (pressure velocity/omega, Pa/s), "
        "not true vertical velocity in m/s."
    )
    return ds


def download_hrrr_for_range(start_date, end_date, sites, output_dir, max_threads=10):
    months = pd.period_range(start_date, end_date, freq="M")
    grid_indices = {}

    for month in months:
        month_start = max(pd.Timestamp(start_date), month.start_time)
        month_end = min(pd.Timestamp(end_date) + pd.Timedelta(hours=23), month.end_time)
        times = pd.date_range(month_start, month_end, freq="h")

        pending_sites = []
        for site in sites:
            out_path = os.path.join(output_dir, site, f"hrrr_{site}_{month.strftime('%Y%m')}.nc")
            if os.path.exists(out_path):
                logger.info("Skipping %s %s, already exists", site, month)
            else:
                pending_sites.append(site)
        if not pending_sites:
            continue

        logger.info("Processing %s (%d hourly timesteps)", month, len(times))
        records = {site: [None] * len(times) for site in pending_sites}
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(download_timestep, timestamp, grid_indices): t_idx
                for t_idx, timestamp in enumerate(times)
            }
            for future in futures:
                t_idx = futures[future]
                result = future.result()
                for site in pending_sites:
                    records[site][t_idx] = result[site] if result else {}

        for site in pending_sites:
            ds = build_site_dataset(site, times, records[site], grid_indices)
            out_dir = os.path.join(output_dir, site)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"hrrr_{site}_{month.strftime('%Y%m')}.nc")
            ds.to_netcdf(out_path)
            logger.info("Wrote %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download HRRR analysis data at fixed points and save monthly netCDF files."
    )
    parser.add_argument("--start-date", type=str, default="2023-12-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2025-09-15", help="End date (YYYY-MM-DD), inclusive.")
    parser.add_argument(
        "--sites",
        type=str,
        nargs="+",
        default=list(SITES.keys()),
        choices=list(SITES.keys()),
        help="Sites to download data for.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_wfip3/hrrr",
        help="Directory to write per-site, per-month netCDF files to.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=10,
        help="Number of concurrent HRRR downloads per month.",
    )
    args = parser.parse_args()

    download_hrrr_for_range(args.start_date, args.end_date, args.sites, args.output_dir, args.max_threads)


if __name__ == "__main__":
    main()
