"""
Merge two CSV datasets together

This script merges two CSV files (e.g., real data and generated data) into a single
combined dataset. Assumes both CSVs have the same column structure.
"""

import pandas as pd
import argparse
from pathlib import Path


def merge_datasets(csv1, csv2, output_csv, shuffle=True, random_seed=42):
    """
    Merge two CSV datasets together

    Args:
        csv1 (str): Path to first CSV file
        csv2 (str): Path to second CSV file
        output_csv (str): Path to output merged CSV file
        shuffle (bool): Whether to shuffle the merged dataset (default: True)
        random_seed (int): Random seed for shuffling (default: 42)
    """
    print(f"{'='*70}")
    print(f"Merging Datasets")
    print(f"{'='*70}")

    # Read first CSV
    print(f"\nReading: {csv1}")
    df1 = pd.read_csv(csv1)
    print(f"  Rows: {len(df1)}")
    print(f"  Columns: {len(df1.columns)}")

    # Read second CSV
    print(f"\nReading: {csv2}")
    df2 = pd.read_csv(csv2)
    print(f"  Rows: {len(df2)}")
    print(f"  Columns: {len(df2.columns)}")

    # Check if columns match
    if list(df1.columns) != list(df2.columns):
        print("\nWarning: Column names don't match exactly!")
        print(f"CSV1 columns: {list(df1.columns)}")
        print(f"CSV2 columns: {list(df2.columns)}")

        # Find differences
        cols1_set = set(df1.columns)
        cols2_set = set(df2.columns)
        only_in_1 = cols1_set - cols2_set
        only_in_2 = cols2_set - cols1_set

        if only_in_1:
            print(f"Only in CSV1: {only_in_1}")
        if only_in_2:
            print(f"Only in CSV2: {only_in_2}")

        print("\nProceeding with merge (missing columns will be filled with NaN)...")

    # Merge datasets
    print(f"\nMerging datasets...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"  Total rows after merge: {len(merged_df)}")

    # Shuffle if requested
    if shuffle:
        print(f"\nShuffling merged dataset (random_seed={random_seed})...")
        merged_df = merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Save merged dataset
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print(f"Merge Complete!")
    print(f"{'='*70}")
    print(f"Output saved to: {output_csv}")
    print(f"Total rows: {len(merged_df)}")

    # Show distribution if 'Pleural Effusion' column exists
    if 'Pleural Effusion' in merged_df.columns:
        print(f"\nPleural Effusion distribution:")
        effusion_counts = merged_df['Pleural Effusion'].value_counts().sort_index()
        for value, count in effusion_counts.items():
            if pd.isna(value):
                print(f"  NaN/Missing: {count}")
            else:
                label = "Positive" if value == 1.0 else "Negative" if value == 0.0 else "Uncertain"
                print(f"  {value} ({label}): {count}")

    print(f"\nFirst few rows of merged dataset:")
    print(merged_df.head())

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description='Merge two CSV datasets together'
    )

    parser.add_argument('--csv1', type=str, required=True,
                       help='Path to first CSV file (e.g., real training data)')
    parser.add_argument('--csv2', type=str, required=True,
                       help='Path to second CSV file (e.g., generated data)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output merged CSV file')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='Do not shuffle the merged dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    merge_datasets(
        csv1=args.csv1,
        csv2=args.csv2,
        output_csv=args.output,
        shuffle=not args.no_shuffle,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
