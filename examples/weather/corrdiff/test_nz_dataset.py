#!/usr/bin/env python
"""
Test script to verify NZMiniDataset loads correctly.
Run from examples/weather/corrdiff directory:
    python test_nz_dataset.py
"""

import sys
sys.path.insert(0, '.')

from datasets.nzmini import NZMiniDataset

print("=" * 80)
print("Testing NZ GEFS → WHACS Dataset")
print("=" * 80)

# Test with a small subset of years for quick validation
print("\n1. Initializing dataset (train split, 2000-2001)...")
try:
    dataset = NZMiniDataset(
        gefs_dir="../../../NZ_GEFS",
        whacs_dir="../../../WHACS",
        stats_path="../../../GEFS_WHACS_stats.json",
        train_years=[2000, 2001],
        valid_years=[2021],
        train=True,
        input_variables=["swh", "dirpw", "perpw"],
        output_variables=["hs", "dir", "t01"],
        target_resolution=0.0625
    )
    print("✓ Dataset initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize dataset: {e}")
    sys.exit(1)

print(f"\n2. Dataset info:")
print(f"   Total samples: {len(dataset)}")
print(f"   Image shape: {dataset.image_shape()}")
print(f"   Input channels: {len(dataset.input_channels())}")
print(f"   Output channels: {len(dataset.output_channels())}")

print(f"\n3. Testing data loading (sample 0)...")
try:
    output, input_data = dataset[0]
    print(f"✓ Successfully loaded sample 0")
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output.shape}")
    
    import numpy as np
    # Use nanmin/nanmax to ignore NaN values (which represent land/invalid areas)
    input_valid = input_data[~np.isnan(input_data)]
    output_valid = output[~np.isnan(output)]
    
    print(f"   Input range: [{input_valid.min():.3f}, {input_valid.max():.3f}]")
    print(f"   Input NaN: {np.isnan(input_data).sum()}/{input_data.size} ({100*np.isnan(input_data).mean():.1f}%)")
    print(f"   Output range: [{output_valid.min():.3f}, {output_valid.max():.3f}]")
    print(f"   Output NaN: {np.isnan(output).sum()}/{output.size} ({100*np.isnan(output).mean():.1f}%)")
except Exception as e:
    print(f"✗ Failed to load sample: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n4. Testing multiple samples...")
try:
    for i in [0, len(dataset)//2, len(dataset)-1]:
        output, input_data = dataset[i]
    print(f"✓ Successfully loaded samples at indices 0, {len(dataset)//2}, {len(dataset)-1}")
except Exception as e:
    print(f"✗ Failed to load multiple samples: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed! Dataset is ready for training.")
print("=" * 80)
print("\nNext steps:")
print("  1. Review config: conf/config_train_gefs_WHACS_regression.yaml")
print("  2. Start training: python train.py --config-name=config_train_gefs_WHACS_regression")
print("=" * 80)
