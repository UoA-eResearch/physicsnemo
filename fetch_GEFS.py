#!/usr/bin/env python3

import xarray as xr
import pandas as pd
from herbie import Herbie
from tqdm.contrib.concurrent import thread_map

def download_GEFS(date):
    try:
        dsets = []
        for fxx in [0, 3, 6, 9, 12, 15, 18, 21]:
            H = Herbie(date, model="gefs", product="wave", member="p01", fxx=fxx, save_dir="./GEFS", verbose=True)
            ds = H.xarray(rf":(?:HTSGW|DIRPW|PERPW):surface:(?:anl|(?:3|6|9|12|15|18|21) hour fcst)", remove_grib=True, backend_kwargs=dict(decode_timedelta=False))
            ds = ds.sel(longitude=slice(165, 180), latitude=slice(-33,-48))
            ds = ds.drop_vars("time").rename({"step": "time"}).assign_coords(time=ds.valid_time.values).drop_vars(["gribfile_projection", "surface", "valid_time"])
            dsets.append(ds)
        dsets = xr.concat(dsets, dim="time")
        return dsets
    except:
        print(f"Failed to download data for {date}")
        return None

for year in range(2021, 2025):
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