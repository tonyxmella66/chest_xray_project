"""
Create CSV file for generated chest X-ray images

This script creates a CSV file in the format expected by the classifier training
scripts, using generated images from the LoRA inference script. All generated
images are labeled as positive Pleural Effusion samples.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


def create_generated_csv(
    images_dir,
    output_csv,
    relative_to=None,
    sex='Unknown',
    age=None,
    frontal_lateral='Frontal',
    ap_pa='AP'
):
    """
    Create a CSV file for generated images compatible with the classifier training script

    Args:
        images_dir (str): Directory containing generated images
        output_csv (str): Path to output CSV file
        relative_to (str): Directory to make paths relative to (if None, uses absolute paths)
        sex (str): Sex value to use for all images (default: 'Unknown')
        age (int): Age value to use for all images (default: None/empty)
        frontal_lateral (str): View type (default: 'Frontal')
        ap_pa (str): AP/PA designation (default: 'AP')
    """

    # Get list of image files
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")

    # Find all PNG and JPG images
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_files)} images in {images_dir}")

    # Create DataFrame with the same structure as classifier_dev_full.csv
    # Columns: Path, Sex, Age, Frontal/Lateral, AP/PA, and all disease labels

    disease_columns = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]

    # Create list of rows
    rows = []
    for img_file in sorted(image_files):
        # Create path (relative or absolute)
        if relative_to:
            relative_to_path = Path(relative_to)
            try:
                img_path = img_file.relative_to(relative_to_path)
            except ValueError:
                # If file is not relative to the specified path, use absolute
                img_path = img_file
        else:
            img_path = img_file

        row = {
            'Path': str(img_path),
            'Sex': sex,
            'Age': age if age is not None else '',
            'Frontal/Lateral': frontal_lateral,
            'AP/PA': ap_pa,
        }

        # Set all disease labels
        # Pleural Effusion = 1.0 (positive), all others = 0.0 (negative)
        for disease in disease_columns:
            if disease == 'Pleural Effusion':
                row[disease] = 1.0
            else:
                row[disease] = 0.0

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure column order matches the expected format
    column_order = ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'] + disease_columns
    df = df[column_order]

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nCSV file created: {output_csv}")
    print(f"Total images: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nAll images labeled with:")
    print(f"  Pleural Effusion: 1.0 (positive)")
    print(f"  All other conditions: 0.0 (negative)")

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV file for generated chest X-ray images'
    )

    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing generated images (e.g., generated_xrays)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Output CSV file path (default: auto-generated based on images_dir)')
    parser.add_argument('--relative_to', type=str, default=None,
                       help='Make paths relative to this directory (e.g., "data")')
    parser.add_argument('--sex', type=str, default='Unknown',
                       help='Sex value for all images (default: "Unknown")')
    parser.add_argument('--age', type=int, default=None,
                       help='Age value for all images (default: None/empty)')
    parser.add_argument('--frontal_lateral', type=str, default='Frontal',
                       choices=['Frontal', 'Lateral'],
                       help='View type (default: "Frontal")')
    parser.add_argument('--ap_pa', type=str, default='AP',
                       choices=['AP', 'PA'],
                       help='AP/PA designation (default: "AP")')

    args = parser.parse_args()

    # Auto-generate output CSV name if not provided
    if args.output_csv is None:
        images_dir_name = Path(args.images_dir).name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_csv = f'generated_data_{images_dir_name}_{timestamp}.csv'

    print(f"{'='*70}")
    print(f"Creating CSV for Generated Images")
    print(f"{'='*70}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Relative to: {args.relative_to if args.relative_to else 'None (absolute paths)'}")
    print(f"Sex: {args.sex}")
    print(f"Age: {args.age if args.age else 'Empty'}")
    print(f"Frontal/Lateral: {args.frontal_lateral}")
    print(f"AP/PA: {args.ap_pa}")
    print(f"{'='*70}\n")

    create_generated_csv(
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        relative_to=args.relative_to,
        sex=args.sex,
        age=args.age,
        frontal_lateral=args.frontal_lateral,
        ap_pa=args.ap_pa
    )

    print(f"\n{'='*70}")
    print(f"CSV Creation Complete!")
    print(f"{'='*70}")
    print(f"\nYou can now use this CSV with the classifier training script:")
    print(f"python src/train_classifier_densenet121.py \\")
    print(f"  --train_csv {args.output_csv} \\")
    print(f"  --target 'Pleural Effusion' \\")
    print(f"  --data_root {args.relative_to if args.relative_to else '.'}")


if __name__ == '__main__':
    main()
