#!/usr/bin/env python3
"""
Copy Images to Organized Directories

This script:
1. Reads CSV files from the filtered_splits directory (created by filter_and_split_data.py)
2. Creates separate directories for each dataset: classifier_dev_full, diffusion_finetune, test
3. Optionally processes CV split folders (fold_1_train, fold_1_val, etc.)
4. Copies images from the original dataset to the appropriate directory
5. Creates new CSV files with updated paths pointing to the new directory structure

Usage:
  First run filter_and_split_data.py to create the CSV splits, then run this script.
"""

import pandas as pd
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def copy_images_and_create_labels(
    csv_dir='data/filtered_splits',
    source_image_root='.',
    output_root='data/organized_images',
    copy_images=True,
    create_symlinks=False,
    include_cv_splits=False
):
    """
    Copy images to organized directories and create corresponding label CSVs

    Args:
        csv_dir: Directory containing the CSV files
        source_image_root: Root directory where original images are stored
        output_root: Root directory for organized output
        copy_images: If True, copy images; if False, only create directory structure
        create_symlinks: If True, create symlinks instead of copying (faster, saves space)
        include_cv_splits: If True, also process CV splits
    """

    print("="*80)
    print("ORGANIZING IMAGES INTO DIRECTORIES")
    print("="*80)

    # Define CSV paths for main datasets
    csv_files = {
        'classifier_dev_full': os.path.join(csv_dir, 'classifier_dev_full.csv'),
        'diffusion_finetune': os.path.join(csv_dir, 'diffusion_finetune.csv'),
        'test': os.path.join(csv_dir, 'test.csv')
    }

    # Add CV splits if requested
    if include_cv_splits:
        cv_splits_dir = os.path.join(csv_dir, 'cv_splits')
        if os.path.exists(cv_splits_dir):
            for fold_num in range(1, 6):
                csv_files[f'cv_fold_{fold_num}_train'] = os.path.join(cv_splits_dir, f'fold_{fold_num}_train.csv')
                csv_files[f'cv_fold_{fold_num}_val'] = os.path.join(cv_splits_dir, f'fold_{fold_num}_val.csv')
    
    # Check if all CSV files exist
    print(f"Checking for CSV files in {csv_dir}...")
    for name, path in csv_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}\n"
                                   f"Please ensure the data organization script has been run!")
        csv_name = os.path.basename(path)
        print(f"   ✓ Found {csv_name}")
    
    # Create output directories
    print(f"Creating output directory structure in {output_root}...")
    output_dirs = {}
    for dataset_name in csv_files.keys():
        dir_path = os.path.join(output_root, dataset_name)
        os.makedirs(dir_path, exist_ok=True)
        output_dirs[dataset_name] = dir_path
        print(f"   ✓ Created: {dir_path}")
    
    # Process each dataset
    total_copied = 0
    total_failed = 0
    
    for dataset_name, csv_path in csv_files.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"\n   Loaded {len(df):,} records from {os.path.basename(csv_path)}")
        
        # Create new dataframe for updated paths
        new_records = []
        copied_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process each image
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"   Processing {dataset_name}"):
            old_path = row['Path']
            
            # Construct source path
            source_path = os.path.join(source_image_root, old_path)
            
            # Create new filename (flattened path to avoid collisions)
            # e.g., CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
            # becomes patient00001_study1_view1_frontal.jpg
            path_parts = Path(old_path).parts
            
            # Find patient/study/view structure
            if 'patient' in old_path.lower():
                # Extract relevant parts after train/valid directory
                relevant_parts = [p for p in path_parts if 'patient' in p.lower() or 
                                'study' in p.lower() or 'view' in p.lower() or 
                                p.endswith('.jpg') or p.endswith('.png')]
                if relevant_parts:
                    new_filename = '_'.join(relevant_parts)
                else:
                    # Fallback: use last few parts
                    new_filename = '_'.join(path_parts[-3:]) if len(path_parts) >= 3 else Path(old_path).name
            else:
                new_filename = Path(old_path).name
            
            # Destination path
            dest_path = os.path.join(output_dirs[dataset_name], new_filename)
            
            # New relative path for CSV (relative to output_root)
            new_relative_path = os.path.join(dataset_name, new_filename)
            
            # Check if source file exists
            if not os.path.exists(source_path):
                print(f"\n   Source not found: {source_path}")
                failed_count += 1
                continue
            
            # Copy or symlink the file
            if copy_images:
                try:
                    if create_symlinks:
                        # Create symlink (faster, saves space)
                        if not os.path.exists(dest_path):
                            os.symlink(os.path.abspath(source_path), dest_path)
                            copied_count += 1
                        else:
                            skipped_count += 1
                    else:
                        # Copy file (slower but safer)
                        if not os.path.exists(dest_path):
                            shutil.copy2(source_path, dest_path)
                            copied_count += 1
                        else:
                            skipped_count += 1
                except Exception as e:
                    print(f"\n   Error copying {source_path}: {e}")
                    failed_count += 1
                    continue
            
            # Add to new records with updated path
            new_row = row.copy()
            new_row['Path'] = new_relative_path
            new_records.append(new_row)
        
        # Create new DataFrame with updated paths
        new_df = pd.DataFrame(new_records)
        
        # Save new CSV
        new_csv_path = os.path.join(output_root, f'{dataset_name}.csv')
        new_df.to_csv(new_csv_path, index=False)
        
        print(f"\n   Summary for {dataset_name}:")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Successfully processed: {len(new_records):,}")
        if copy_images:
            action = "Symlinked" if create_symlinks else "Copied"
            print(f"   • {action} images: {copied_count:,}")
            print(f"   • Skipped (already exist): {skipped_count:,}")
        print(f"   • Failed: {failed_count:,}")
        print(f"   • CSV saved: {new_csv_path}")
        
        total_copied += copied_count
        total_failed += failed_count
    
    # Final summary
    print(f"\n{'='*80}")
    print("IMAGE ORGANIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTotal images processed: {total_copied:,}")
    print(f"Total failures: {total_failed:,}")
    
    print(f"Output structure:")
    print(f"   {output_root}/")
    print(f"   ├── classifier_dev_full/")
    print(f"   │   └── [images]")
    print(f"   ├── classifier_dev_full.csv")
    print(f"   ├── diffusion_finetune/")
    print(f"   │   └── [images]")
    print(f"   ├── diffusion_finetune.csv")
    print(f"   ├── test/")
    print(f"   │   └── [images]")
    print(f"   └── test.csv")
    if include_cv_splits:
        print(f"   ├── cv_fold_1_train/, cv_fold_1_val/, ...")
        print(f"   └── cv_fold_1_train.csv, cv_fold_1_val.csv, ...")

    print(f"\nNext steps:")
    print(f"   1. Verify the image counts match your expectations")
    print(f"   2. Update your training scripts to use the new CSV files:")
    print(f"      - Classifier Dev: {output_root}/classifier_dev_full.csv")
    print(f"      - Diffusion: {output_root}/diffusion_finetune.csv")
    print(f"      - Test: {output_root}/test.csv")
    print(f"   3. Set image_root to: {output_root}")
    
    print("\n" + "="*80 + "\n")
    
    return output_dirs


def verify_organization(output_root='data/organized_images', include_cv_splits=False):
    """
    Verify that the organization was successful

    Args:
        output_root: Root directory for organized output
        include_cv_splits: If True, also verify CV splits
    """
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    csv_files = ['classifier_dev_full.csv', 'diffusion_finetune.csv', 'test.csv']

    # Add CV splits if requested
    if include_cv_splits:
        for fold_num in range(1, 6):
            csv_files.append(f'cv_fold_{fold_num}_train.csv')
            csv_files.append(f'cv_fold_{fold_num}_val.csv')

    for csv_file in csv_files:
        csv_path = os.path.join(output_root, csv_file)
        if not os.path.exists(csv_path):
            print(f"   CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        dataset_name = csv_file.replace('.csv', '')

        print(f"\n   {dataset_name}:")
        print(f"   • CSV records: {len(df):,}")

        # Check if images exist
        missing = 0
        for path in df['Path'].head(100):  # Check first 100
            full_path = os.path.join(output_root, path)
            if not os.path.exists(full_path):
                missing += 1

        if missing == 0:
            print(f"   • Image verification: All checked images exist")
        else:
            print(f"   • Image verification: {missing}/100 checked images missing")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Copy and organize images into separate directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (copy images after running filter_and_split_data.py)
  python copy_filtered_images.py

  # Use symlinks instead of copying (faster, saves disk space)
  python copy_filtered_images.py --symlinks

  # Include CV splits
  python copy_filtered_images.py --include_cv_splits

  # Only create directory structure (no copying)
  python copy_filtered_images.py --no_copy

  # Custom paths
  python copy_filtered_images.py \\
      --csv_dir data/filtered_splits \\
      --source_root . \\
      --output_root data/organized_images
        """
    )

    parser.add_argument('--csv_dir', type=str, default='data/filtered_splits',
                       help='Directory containing CSV files (default: data/filtered_splits)')
    parser.add_argument('--source_root', type=str, default='.',
                       help='Root directory of source images (default: current directory)')
    parser.add_argument('--output_root', type=str, default='data/organized_images',
                       help='Output root directory (default: data/organized_images)')
    parser.add_argument('--copy', dest='copy_images', action='store_true', default=True,
                       help='Copy images (default: True)')
    parser.add_argument('--no_copy', dest='copy_images', action='store_false',
                       help='Do not copy images, only create structure')
    parser.add_argument('--symlinks', action='store_true',
                       help='Create symlinks instead of copying (saves space)')
    parser.add_argument('--include_cv_splits', action='store_true',
                       help='Also process CV fold splits')
    parser.add_argument('--verify', action='store_true',
                       help='Verify organization after completion')
    
    args = parser.parse_args()
    
    # Check if CSV directory exists
    if not os.path.exists(args.csv_dir):
        print(f"ERROR: CSV directory not found: {args.csv_dir}")
        print(f"Please ensure the data organization script has been run!")
        return

    # Warn about symlinks on Windows
    if args.symlinks and os.name == 'nt':
        print("WARNING: Symlinks may require administrator privileges on Windows")
        print("Consider using --copy instead if you encounter permission errors\n")

    # Run the organization
    copy_images_and_create_labels(
        csv_dir=args.csv_dir,
        source_image_root=args.source_root,
        output_root=args.output_root,
        copy_images=args.copy_images,
        create_symlinks=args.symlinks,
        include_cv_splits=args.include_cv_splits
    )

    # Verify if requested
    if args.verify:
        verify_organization(args.output_root, include_cv_splits=args.include_cv_splits)


if __name__ == "__main__":
    main()
