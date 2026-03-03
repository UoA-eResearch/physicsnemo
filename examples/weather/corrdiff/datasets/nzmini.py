# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import glob
import json
import math
import os
import re
from typing import List, Tuple, Union

import numpy as np
import xarray as xr
from numba import jit, prange
from scipy.interpolate import griddata

from datasets.base import ChannelMetadata, DownscalingDataset
from helpers.train_helpers import _convert_datetime_to_cftime

# GEFS wave reforecast -> WHACS NZ

class NZMiniDataset(DownscalingDataset):
    """
    Reader for NZ dataset downscaling GEFS (0.25°) to WHACS (~0.0625°).
    
    Loads GEFS data from a single consolidated NetCDF file and WHACS irregular grid data
    from monthly NetCDF files, performing on-the-fly regridding to save disk space.
    
    GEFS is loaded in memory from NZ_GEFS.nc, with automatic fallback to
    yearly files (NZ_GEFS_YYYY.nc) for requested years not present in the
    consolidated file.
    WHACS remains lazy-loaded to control memory usage.
    """

    def __init__(
        self,
        gefs_dir: str,
        whacs_dir: str,
        stats_path: str,
        train_years: Union[List[int], None] = None,
        valid_years: Union[List[int], None] = None,
        train: bool = True,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        target_resolution: float = 0.0625,
    ):
        """
        Initialize NZ dataset with lazy loading.
        
        Parameters
        ----------
        gefs_dir : str
            Directory containing consolidated GEFS NetCDF file (NZ_GEFS.nc)
        whacs_dir : str
            Directory containing WHACS subdirectories (e.g., hs_NZ/, dir_NZ/)
        stats_path : str
            Path to JSON file with normalization statistics
        train_years : List[int], optional
            Years to use for training
        valid_years : List[int], optional
            Years to use for validation
        train : bool
            If True, use train_years; otherwise use valid_years
        input_variables : List[str], optional
            GEFS variables to use as input (default: swh, dirpw, perpw)
        output_variables : List[str], optional
            WHACS variables to use as output (default: hs, dir, t01)
        target_resolution : float
            Target grid resolution in degrees (default: 0.0625)
        """
        self.gefs_dir = gefs_dir
        self.whacs_dir = whacs_dir
        self.target_resolution = target_resolution
        self.upsample_factor = 4  # GEFS 0.25° to WHACS ~0.0625°
        
        # Set default variables
        if input_variables is None:
            input_variables = ["swh", "dirpw", "perpw"]
        if output_variables is None:
            output_variables = ["hs", "dir", "t01"]
        
        self.input_variables = input_variables
        self.output_variables = output_variables
        
        # Select years
        years = train_years if train else valid_years
        if years is None:
            raise ValueError("Must specify either train_years or valid_years")
        
        print(f"Loading {'training' if train else 'validation'} data for years: {years}")
        
        # Load consolidated GEFS data into memory and get dimensions
        gefs_nlat, gefs_nlon = self._load_gefs_data(years)
        
        # Load WHACS file paths and get grid info (lazy loading)
        self._load_whacs_file_paths(years)
        
        # Create target grid for WHACS data
        self._create_target_grid(gefs_nlat, gefs_nlon)
        
        # Build time index from file metadata and match timestamps
        self._build_time_index_and_match()
        
        # Load normalization stats
        self._load_stats(stats_path)
        
        print(f"Dataset initialized with {len(self.matched_times)} samples")

    def _load_gefs_data(self, years: List[int]) -> Tuple[int, int]:
        """Load GEFS data into memory using consolidated file + yearly fallback.
        
        Returns
        -------
        gefs_nlat : int
            Number of latitude points in GEFS grid
        gefs_nlon : int
            Number of longitude points in GEFS grid
        """
        requested_years = list(years)
        requested_year_set = set(requested_years)
        datasets = []
        loaded_sources = []
        covered_years = set()

        # Primary source: consolidated file if present.
        gefs_path = os.path.join(self.gefs_dir, "NZ_GEFS.nc")
        if os.path.exists(gefs_path):
            with xr.open_dataset(gefs_path) as ds:
                missing_input = [var for var in self.input_variables if var not in ds.data_vars]
                if missing_input:
                    raise ValueError(f"Missing GEFS variables in {gefs_path}: {missing_input}")

                year_values = ds["time"].dt.year.values
                selected_idx = np.where(np.isin(year_values, requested_years))[0]
                if len(selected_idx) > 0:
                    selected_years = set(np.unique(year_values[selected_idx]).astype(int).tolist())
                    covered_years.update(selected_years)
                    datasets.append(ds[self.input_variables].isel(time=selected_idx).load())
                    loaded_sources.append(f"{gefs_path} (years: {min(selected_years)}-{max(selected_years)})")

        # Fallback source: yearly files for requested years not covered above.
        missing_years = sorted(requested_year_set - covered_years)
        not_found_years = []
        for year in missing_years:
            year_path = os.path.join(self.gefs_dir, f"NZ_GEFS_{year}.nc")
            if not os.path.exists(year_path):
                not_found_years.append(year)
                continue

            with xr.open_dataset(year_path) as ds_year:
                missing_input = [var for var in self.input_variables if var not in ds_year.data_vars]
                if missing_input:
                    raise ValueError(f"Missing GEFS variables in {year_path}: {missing_input}")

                year_values = ds_year["time"].dt.year.values
                selected_idx = np.where(year_values == year)[0]
                if len(selected_idx) == 0:
                    continue

                datasets.append(ds_year[self.input_variables].isel(time=selected_idx).load())
                covered_years.add(year)
                loaded_sources.append(year_path)

        if not datasets:
            raise ValueError(
                f"No GEFS timesteps found for requested years {requested_years}. "
                f"Checked consolidated file ({gefs_path}) and yearly files in {self.gefs_dir}."
            )

        self.gefs_data = xr.concat(datasets, dim="time").sortby("time")

        # Handle different dimension names
        if 'lat' in self.gefs_data.sizes:
            gefs_nlat = self.gefs_data.sizes['lat']
            gefs_nlon = self.gefs_data.sizes['lon']
        elif 'latitude' in self.gefs_data.sizes:
            gefs_nlat = self.gefs_data.sizes['latitude']
            gefs_nlon = self.gefs_data.sizes['longitude']
        else:
            raise ValueError(f"Could not find lat/lon dimensions in loaded GEFS data")

        if not_found_years:
            print(f"  Warning: Missing GEFS yearly files for years: {not_found_years}")
        print("  GEFS sources:")
        for src in loaded_sources:
            print(f"    - {src}")
        print(f"  GEFS grid: {gefs_nlat}×{gefs_nlon}")
        print(f"  GEFS loaded timesteps: {self.gefs_data.sizes['time']}")

        return gefs_nlat, gefs_nlon

    def _load_whacs_file_paths(self, years: List[int]) -> None:
        """Load WHACS file paths and extract grid coordinates.
        
        Store file paths and load grid info (lon/lat) but not actual data.
        """
        self.whacs_files = {}  # {var: {(year, month): file_path}}
        self.whacs_lon = None
        self.whacs_lat = None
        
        for var in self.output_variables:
            var_dir = os.path.join(self.whacs_dir, f"{var}_NZ")
            if not os.path.exists(var_dir):
                raise ValueError(f"WHACS directory not found: {var_dir}")
            
            self.whacs_files[var] = {}
            
            # Collect all files, then parse timestamp range suffix
            all_files = sorted(glob.glob(os.path.join(var_dir, "*.nc")))
            year_set = set(years)
            
            for file in all_files:
                # Pattern suffix: ..._{YYYYMMDDHHMM}-{YYYYMMDDHHMM}.nc
                base = os.path.basename(file)
                match = re.search(r'_(\d{12})-(\d{12})\.nc$', base)
                if match is None:
                    continue

                start_ts = match.group(1)
                file_year = int(start_ts[:4])
                file_month = int(start_ts[4:6])

                if file_year in year_set:
                    self.whacs_files[var][(file_year, file_month)] = file
            
            if not self.whacs_files[var]:
                raise ValueError(f"No WHACS files found for {var} in years {years}")
        
        # Load grid coordinates from first file (same for all variables)
        first_var = self.output_variables[0]
        first_file = list(self.whacs_files[first_var].values())[0]
        with xr.open_dataset(first_file) as ds:
            self.whacs_lon = ds['longitude'].values.copy()
            self.whacs_lat = ds['latitude'].values.copy()
            print(f"  WHACS grid: {len(self.whacs_lon)} points")

    def _create_target_grid(self, gefs_nlat: int, gefs_nlon: int) -> None:
        """Create regular target grid matching upsampled GEFS dimensions.
        
        Parameters
        ----------
        gefs_nlat : int
            GEFS latitude dimension
        gefs_nlon : int
            GEFS longitude dimension
        """
        # Upsampled GEFS dimensions (4x upsampling)
        upsampled_nlat = gefs_nlat * self.upsample_factor
        upsampled_nlon = gefs_nlon * self.upsample_factor
        
        # Get WHACS bounds
        lon_min, lon_max = float(self.whacs_lon.min()), float(self.whacs_lon.max())
        lat_min, lat_max = float(self.whacs_lat.min()), float(self.whacs_lat.max())
        
        # Create grid with dimensions matching upsampled GEFS
        self.grid_lon = np.linspace(lon_min, lon_max, upsampled_nlon)
        self.grid_lat = np.linspace(lat_min, lat_max, upsampled_nlat)
        
        self.grid_x, self.grid_y = np.meshgrid(self.grid_lon, self.grid_lat)
        self.img_shape = self.grid_y.shape
        
        print(f"  Target grid: {self.img_shape[0]} x {self.img_shape[1]}")

    def _build_time_index_and_match(self, time_tolerance_hours: int = 1) -> None:
        """Build time index from file metadata and match GEFS/WHACS timestamps.
        
        This avoids loading all data into memory by extracting times from file headers.
        """
        # Extract times from in-memory GEFS data
        gefs_times = self.gefs_data['time'].values
        self.gefs_ref_local = np.arange(len(gefs_times), dtype=np.int32)
        
        # Extract times from WHACS files (from first variable, since all have same time coords)
        first_var = self.output_variables[0]
        whacs_times = []
        whacs_file_refs = []  # (year, month) tuples
        whacs_file_local = []
        
        for (year, month), file_path in sorted(self.whacs_files[first_var].items()):
            with xr.open_dataset(file_path) as ds:
                times = ds['time'].values
                whacs_times.extend(times)
                for local_idx in range(len(times)):
                    whacs_file_refs.append((year, month))
                    whacs_file_local.append(local_idx)
        
        whacs_times = np.array(whacs_times)
        self.whacs_file_refs = np.array(whacs_file_refs, dtype=object)
        self.whacs_file_local = np.array(whacs_file_local, dtype=np.int32)
        
        # Find matching times between GEFS and WHACS
        self.matched_times = []
        self.gefs_indices = []
        self.whacs_indices = []
        
        tolerance = np.timedelta64(time_tolerance_hours, 'h')
        
        for i, gefs_time in enumerate(gefs_times):
            # Find closest WHACS time
            time_diffs = np.abs(whacs_times - gefs_time)
            min_diff_idx = np.argmin(time_diffs)
            
            if time_diffs[min_diff_idx] <= tolerance:
                self.matched_times.append(gefs_time)
                self.gefs_indices.append(i)
                self.whacs_indices.append(min_diff_idx)
        
        self.matched_times = np.array(self.matched_times)
        self.gefs_indices = np.array(self.gefs_indices)
        self.whacs_indices = np.array(self.whacs_indices)
        
        # Store time arrays for later reference
        self.gefs_times = gefs_times
        self.whacs_times = whacs_times
        
        print(f"  Matched {len(self.matched_times)} timestamps (tolerance: {time_tolerance_hours}h)")
    
    def _load_stats(self, stats_path: str) -> None:
        """Load normalization statistics from JSON file."""
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        # Input stats (GEFS + invariants)
        input_means = []
        input_stds = []
        
        for var in self.input_variables:
            if var in stats.get('input', {}):
                input_means.append(stats['input'][var]['mean'])
                input_stds.append(stats['input'][var]['std'])
            else:
                print(f"  Warning: No stats for input variable {var}, using 0/1")
                input_means.append(0.0)
                input_stds.append(1.0)
        
        # Add invariant stats (lat/lon)
        for var in ['latitude', 'longitude']:
            if var in stats.get('invariant', {}):
                input_means.append(stats['invariant'][var]['mean'])
                input_stds.append(stats['invariant'][var]['std'])
            else:
                # Use grid stats if not in file
                if var == 'latitude':
                    input_means.append(float(self.grid_lat.mean()))
                    input_stds.append(float(self.grid_lat.std()))
                else:
                    input_means.append(float(self.grid_lon.mean()))
                    input_stds.append(float(self.grid_lon.std()))
        
        self.input_mean = np.array(input_means, dtype=np.float32)[:, None, None]
        self.input_std = np.array(input_stds, dtype=np.float32)[:, None, None]
        
        # Output stats (WHACS)
        output_means = []
        output_stds = []
        
        for var in self.output_variables:
            if var in stats.get('output', {}):
                output_means.append(stats['output'][var]['mean'])
                output_stds.append(stats['output'][var]['std'])
            else:
                print(f"  Warning: No stats for output variable {var}, using 0/1")
                output_means.append(0.0)
                output_stds.append(1.0)
        
        self.output_mean = np.array(output_means, dtype=np.float32)[:, None, None]
        self.output_std = np.array(output_stds, dtype=np.float32)[:, None, None]

    def _load_gefs_sample(self, gefs_global_idx: int) -> np.ndarray:
        """Load single GEFS sample on-demand.
        
        Parameters
        ----------
        gefs_global_idx : int
            Global index across all GEFS files
            
        Returns
        -------
        x : np.ndarray
            Shape (n_vars, nlat, nlon) with GEFS variables
        """
        local_idx = int(self.gefs_ref_local[gefs_global_idx])

        x_gefs = []
        for var in self.input_variables:
            val = self.gefs_data[var].isel(time=local_idx).values
            x_gefs.append(val)
        x_gefs = np.stack(x_gefs, axis=0).astype(np.float32)
        
        return x_gefs

    def _load_whacs_sample(self, whacs_global_idx: int) -> np.ndarray:
        """Load single WHACS sample on-demand and regrid to regular grid.
        
        Parameters
        ----------
        whacs_global_idx : int
            Global index across all WHACS files
            
        Returns
        -------
        y : np.ndarray
            Shape (n_vars, img_height, img_width) on regular grid
        """
        year, month = self.whacs_file_refs[whacs_global_idx]
        local_idx = int(self.whacs_file_local[whacs_global_idx])
        
        # Load and interpolate each output variable
        output_data = []
        
        for var in self.output_variables:
            file_path = self.whacs_files[var][(year, month)]
            with xr.open_dataset(file_path) as ds:
                values = ds[var].isel(time=local_idx).values
                
            # Flatten values to 1D
            values_flat = values.ravel()
            
            # Interpolate to regular grid using nearest neighbor
            points = np.column_stack((self.whacs_lon, self.whacs_lat))
            grid_z = griddata(
                points=points,
                values=values_flat,
                xi=(self.grid_x, self.grid_y),
                method='nearest'
            )
            output_data.append(grid_z)
        
        return np.stack(output_data, axis=0).astype(np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the data sample (output, input) at index idx.
        
        All data loading happens here (lazy loading).
        """
        gefs_idx = self.gefs_indices[idx]
        
        # Load GEFS input and upsample
        x_gefs = self._load_gefs_sample(gefs_idx)
        x = self.upsample(x_gefs)
        
        # Add invariants to input (lat/lon grids)
        lon_grid = self.grid_x.astype(np.float32)
        lat_grid = self.grid_y.astype(np.float32)
        inv = np.stack([lat_grid, lon_grid], axis=0)
        x = np.concatenate([x, inv], axis=0)
        
        # Load WHACS output and regrid on-the-fly
        whacs_idx = self.whacs_indices[idx]
        y = self._load_whacs_sample(whacs_idx)
        
        # Normalize
        x = self.normalize_input(x)
        y = self.normalize_output(y)
        
        return (y, x)

    def __len__(self):
        return len(self.matched_times)

    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        return self.grid_x

    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        return self.grid_y

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in ['latitude', 'longitude']
        ]
        return inputs + invariants

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        """Get time values from the dataset."""
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in self.matched_times
        )
        return [_convert_datetime_to_cftime(t) for t in datetimes]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean

    def upsample(self, x):
        """Extend x around edges with linear extrapolation."""
        y_shape = (
            x.shape[0],
            x.shape[1] * self.upsample_factor,
            x.shape[2] * self.upsample_factor,
        )
        y = np.empty(y_shape, dtype=np.float32)
        _zoom_extrapolate(x, y, self.upsample_factor)
        return y


@jit(nopython=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation.
    Use a numba function here because numpy/scipy options are rather slow.
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)
