"""
Generate NYC Taxi Trip Duration Sample Files for Assignment 2

This script processes the NYC Taxi Trip Duration dataset and creates stratified
samples for use in the assignment.

Prerequisites:
    Place train.csv from Kaggle in this same directory before running.

Usage:
    python generate_nyc_taxi_files.py

Output files (saved in current directory):
    - taxi_500.csv   : Small sample (n≈500)
    - taxi_5k.csv    : Medium sample (n≈5,000)
    - taxi_50k.csv   : Large sample (n≈50,000)
    - taxi_test.csv  : Test set (n=20,000)

Each file includes:
    - Original features: pickup/dropoff coordinates, datetime, passenger_count, etc.
    - distance_km: Pre-computed Euclidean distance between pickup and dropoff
    - pickup_hour: Hour of day (0-23) extracted from pickup_datetime

Expected runtime: ~2-3 minutes
"""

import pandas as pd
import numpy as np
import os
import sys

# Fixed random seed for reproducibility
prng = np.random.RandomState(20260317)

def euclidean_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate approximate distance in km using Euclidean distance.
    For NYC (small distances), this is sufficient and simpler than Haversine.

    Conversion: 1 degree latitude ≈ 111 km
                1 degree longitude ≈ 85 km at NYC's latitude (40°N)
    """
    lat_diff_km = (lat2 - lat1) * 111.0
    lon_diff_km = (lon2 - lon1) * 85.0
    distance = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
    return distance


def print_download_instructions():
    """Print instructions for downloading the dataset."""
    print("=" * 70)
    print("ERROR: train.csv not found")
    print("=" * 70)
    print()
    print("Please download the NYC Taxi Trip Duration dataset:")
    print()
    print("1. Go to: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data")
    print("2. Click 'Download All' or download 'train.csv' directly")
    print("3. You may need to:")
    print("   - Create a Kaggle account (free)")
    print("   - Accept the competition rules")
    print("4. If you downloaded a zip file, extract train.csv from it")
    print("5. Place train.csv in this directory: data/nyc_taxi/")
    print()
    print("The file is approximately 200 MB")
    print()
    print("After downloading, run this script again:")
    print("  python generate_nyc_taxi_files.py")
    print()
    print("Note: You can also use the Kaggle API for automatic download.")
    print("      See: https://github.com/Kaggle/kaggle-api")
    print()
    print("=" * 70)


def check_and_load_data():
    """Check if train.csv exists and load it."""
    train_csv = 'train.csv'

    if not os.path.exists(train_csv):
        print_download_instructions()
        sys.exit(1)

    print("Loading train.csv...")
    print("  (This may take 1-2 minutes for the full dataset)")

    try:
        taxi_full = pd.read_csv(train_csv, parse_dates=['pickup_datetime', 'dropoff_datetime'])
        print(f"  ✓ Loaded {len(taxi_full):,} trips")
        return taxi_full
    except Exception as e:
        print(f"  ✗ Error loading train.csv: {e}")
        print()
        print("Please ensure train.csv is a valid file from the Kaggle competition.")
        sys.exit(1)


def create_stratified_sample(df, n, random_state):
    """Create stratified sample by hour to ensure temporal representation."""
    samples_per_hour = n // 24
    sampled = pd.concat([
        group.sample(min(len(group), samples_per_hour), random_state=random_state)
        for _, group in df.groupby('pickup_hour')
    ])
    return sampled.reset_index(drop=True)


def main():
    print("=" * 70)
    print("NYC Taxi Trip Duration Data Generator")
    print("=" * 70)
    print()

    # Check and load the data
    print("Step 1: Loading NYC Taxi dataset...")
    taxi_full = check_and_load_data()
    print()

    # Process the data
    print("Step 2: Processing data...")

    # Calculate distance
    print("  Computing distances...")
    taxi_full['distance_km'] = euclidean_distance_km(
        taxi_full['pickup_latitude'],
        taxi_full['pickup_longitude'],
        taxi_full['dropoff_latitude'],
        taxi_full['dropoff_longitude']
    )

    # Extract hour
    print("  Extracting temporal features...")
    taxi_full['pickup_hour'] = taxi_full['pickup_datetime'].dt.hour

    print(f"  ✓ Processed {len(taxi_full):,} trips")
    print()

    # Create test set first (20% of full data, or 20k trips, whichever is smaller)
    print("Step 3: Creating test set...")
    test_size = min(20000, int(0.2 * len(taxi_full)))
    test_indices = prng.choice(taxi_full.index, size=test_size, replace=False)
    taxi_test = taxi_full.loc[test_indices].copy()
    print(f"  ✓ Test set: {len(taxi_test):,} trips")
    print()

    # Create training samples (excluding test indices)
    train_pool = taxi_full.loc[~taxi_full.index.isin(test_indices)].copy()

    print("Step 4: Creating stratified training samples...")
    print("  (Stratified by hour to ensure temporal representation)")

    taxi_500 = create_stratified_sample(train_pool, 500, prng)
    print(f"  ✓ Small sample: {len(taxi_500):,} trips")

    taxi_5k = create_stratified_sample(train_pool, 5000, prng)
    print(f"  ✓ Medium sample: {len(taxi_5k):,} trips")

    # For large sample, use smaller amount if dataset is small
    large_sample_size = min(50000, int(0.6 * len(train_pool)))
    taxi_50k = create_stratified_sample(train_pool, large_sample_size, prng)
    print(f"  ✓ Large sample: {len(taxi_50k):,} trips")
    print()

    # Save files
    print("Step 5: Saving CSV files...")

    files_to_save = [
        (taxi_500, 'taxi_500.csv'),
        (taxi_5k, 'taxi_5k.csv'),
        (taxi_50k, 'taxi_50k.csv'),
        (taxi_test, 'taxi_test.csv')
    ]

    for df, filename in files_to_save:
        df.to_csv(filename, index=False)
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  ✓ {filename:20s} ({len(df):6,} rows, {file_size_mb:.1f} MB)")

    print()
    print("=" * 70)
    print("✓ All files generated successfully!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  - taxi_500.csv   : Use for initial exploration and simple models")
    print("  - taxi_5k.csv    : Use for medium-scale experiments")
    print("  - taxi_50k.csv   : Use for large-scale model comparison")
    print("  - taxi_test.csv  : Use this as your test set for ALL models")
    print()
    print("Each file includes:")
    print("  - All original trip features")
    print("  - distance_km: Pre-computed distance")
    print("  - pickup_hour: Extracted hour (0-23)")
    print()
    print("You can now proceed with the assignment!")
    print()


if __name__ == "__main__":
    main()
