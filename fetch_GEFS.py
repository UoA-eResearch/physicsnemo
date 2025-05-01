#!/usr/bin/env python3

import xarray as xr
import pandas as pd
from herbie import Herbie
from tqdm.contrib.concurrent import thread_map

def download_GEFS(date):
    try:
        H = Herbie(date, model="gefs_wave_reforecast", member=0, save_dir="./GEFS", verbose=False)
        ds = H.xarray(rf":(?:HTSGW|DIRPW|PERPW):surface:(?:anl|(?:3|6|9|12|15|18|21) hour fcst)", remove_grib=True, backend_kwargs=dict(decode_timedelta=False))
        ds = ds.sel(longitude=slice(165, 180), latitude=slice(-33,-48))
        ds = ds.drop_vars("time").rename({"step": "time"}).assign_coords(time=ds.valid_time.values).drop_vars(["gribfile_projection", "surface", "valid_time"])
        return ds
    except:
        print(f"Failed to download data for {date}")
        return None

# Temporal extent: 20000108-20191231
for year in range(2009, 2020):
    dsets = thread_map(
        download_GEFS,
        pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"),
        max_workers=32,
        desc=f"Downloading GEFS data for {year}",
        unit="day",
    )
    dsets = [ds for ds in dsets if ds is not None]
    dsets = xr.concat(dsets, dim="time")
    dsets.to_netcdf(f"NZ_GEFS_{year}.nc", mode="w")