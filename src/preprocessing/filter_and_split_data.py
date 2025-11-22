#!/usr/bin/env python3
"""
Filter and Split Dataset for Pleural Effusion Classification and Diffusion Fine-tuning

This script:
1. Filters train.csv and valid.csv to only include Pleural Effusion records (positive and negative)
2. Splits the training data into:
   - diffusion_finetune: For fine-tuning the diffusion model (positive cases only)
   - classifier_dev: For classifier development (remaining data)
3. Creates 5-fold cross-validation splits from classifier_dev
4. Keeps validation data as final test set (never touched during development)

Data Structure:
Original Training Data (after filtering)
│
├─ Diffusion Fine-tuning Set (50% of positive samples)
│  └─ Used ONLY for training the generative model
│  └─ Never seen by classifier
│
└─ Classifier Development Set (remaining samples)
   │
   ├─ 5-Fold Cross-Validation Splits
   │  ├─ Fold 1: Train / Validation
   │  ├─ Fold 2: Train / Validation
   │  ├─ Fold 3: Train / Validation
   │  ├─ Fold 4: Train / Validation
   │  └─ Fold 5: Train / Validation
   │
   └─ Full dataset for final model training

Final Test Set (from original valid.csv)
└─ Touched ONLY ONCE at the very end
"""

import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import StratifiedKFold


def filter_and_split_data(
    train_csv='train.csv',
    valid_csv='valid.csv',
    output_dir='data/filtered_splits',
    target_feature='Pleural Effusion',
    diffusion_ratio=0.5,
    balance_classifier=True,
    n_folds=5,
    random_state=42
):
    """
    Filter and split CheXpert data for classification and diffusion fine-tuning
    
    Args:
        train_csv: Path to original training CSV
        valid_csv: Path to original validation CSV (will become test set)
        output_dir: Directory to save filtered and split CSV files
        target_feature: Medical condition to focus on (default: 'Pleural Effusion')
        diffusion_ratio: Proportion of positive cases to use for diffusion fine-tuning (0.0-1.0)
        balance_classifier: Whether to balance classifier training data
        n_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    cv_dir = os.path.join(output_dir, 'cv_splits')
    os.makedirs(cv_dir, exist_ok=True)
    
    print("="*80)
    print(f"FILTERING AND SPLITTING DATA FOR {target_feature.upper()}")
    print("="*80)
    
    # Load original data
    print(f"Loading original datasets...")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    print(f"   Original training samples: {len(train_df):,}")
    print(f"   Original validation samples: {len(valid_df):,}")
    
    # Check if target feature exists
    if target_feature not in train_df.columns:
        available_features = [col for col in train_df.columns 
                            if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
        raise ValueError(f"Target feature '{target_feature}' not found. Available: {available_features}")
    
    # STEP 1: Filter for binary labels only (0.0 and 1.0)
    print(f"\nSTEP 1: Filtering for binary {target_feature} labels...")
    print(f"Training set label distribution:")
    print(train_df[target_feature].value_counts(dropna=False).sort_index())
    
    train_filtered = train_df[train_df[target_feature].isin([0.0, 1.0])].copy()
    test_filtered = valid_df[valid_df[target_feature].isin([0.0, 1.0])].copy()
    
    print(f"   After filtering:")
    print(f"   ✓ Training: {len(train_filtered):,} samples "
          f"(removed {len(train_df) - len(train_filtered):,})")
    print(f"   ✓ Test: {len(test_filtered):,} samples "
          f"(removed {len(valid_df) - len(test_filtered):,})")
    
    # Analyze filtered training distribution
    train_positive = train_filtered[train_filtered[target_feature] == 1.0]
    train_negative = train_filtered[train_filtered[target_feature] == 0.0]
    
    print(f"   Training set distribution after filtering:")
    print(f"   • Positive ({target_feature}=1.0): {len(train_positive):,} "
          f"({len(train_positive)/len(train_filtered)*100:.1f}%)")
    print(f"   • Negative ({target_feature}=0.0): {len(train_negative):,} "
          f"({len(train_negative)/len(train_filtered)*100:.1f}%)")
    
    # Analyze test distribution
    test_positive = test_filtered[test_filtered[target_feature] == 1.0]
    test_negative = test_filtered[test_filtered[target_feature] == 0.0]
    
    print(f"   Test set distribution after filtering:")
    print(f"   • Positive ({target_feature}=1.0): {len(test_positive):,} "
          f"({len(test_positive)/len(test_filtered)*100:.1f}%)")
    print(f"   • Negative ({target_feature}=0.0): {len(test_negative):,} "
          f"({len(test_negative)/len(test_filtered)*100:.1f}%)")
    
    # STEP 2: Split positive cases for diffusion fine-tuning
    print(f"\nSTEP 2: Splitting positive cases for diffusion fine-tuning...")
    print(f"   Using {diffusion_ratio*100:.0f}% of positive cases for diffusion fine-tuning")
    
    n_diffusion_samples = int(len(train_positive) * diffusion_ratio)
    
    # Randomly split positive cases
    diffusion_df = train_positive.sample(n=n_diffusion_samples, random_state=random_state)
    remaining_positive = train_positive.drop(diffusion_df.index)
    
    print(f"   ✓ Diffusion fine-tuning: {len(diffusion_df):,} positive samples")
    print(f"   ✓ Remaining for classifier: {len(remaining_positive):,} positive samples")
    
    # STEP 3: Create classifier development dataset
    print(f"\nSTEP 3: Creating classifier development dataset...")
    
    if balance_classifier:
        # Balance by matching the number of negative samples to positive samples
        n_positive = len(remaining_positive)
        n_negative = len(train_negative)
        
        print(f"   Balancing classifier data:")
        print(f"   • Available positive samples: {n_positive:,}")
        print(f"   • Available negative samples: {n_negative:,}")
        
        if n_negative > n_positive:
            # Undersample negatives to match positives
            sampled_negative = train_negative.sample(n=n_positive, random_state=random_state)
            classifier_dev = pd.concat([remaining_positive, sampled_negative])
            print(f"   → Undersampled negatives to {n_positive:,} samples")
        else:
            # Use all available data
            classifier_dev = pd.concat([remaining_positive, train_negative])
            print(f"   → Using all data (negatives ≤ positives)")
        
        # Shuffle the combined dataset
        classifier_dev = classifier_dev.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    else:
        # Use all remaining data without balancing
        classifier_dev = pd.concat([remaining_positive, train_negative])
        classifier_dev = classifier_dev.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"   Using unbalanced data (all available samples)")
    
    print(f"   Final classifier development set: {len(classifier_dev):,} samples")
    print(f"   • Positive: {len(classifier_dev[classifier_dev[target_feature]==1.0]):,} "
          f"({len(classifier_dev[classifier_dev[target_feature]==1.0])/len(classifier_dev)*100:.1f}%)")
    print(f"   • Negative: {len(classifier_dev[classifier_dev[target_feature]==0.0]):,} "
          f"({len(classifier_dev[classifier_dev[target_feature]==0.0])/len(classifier_dev)*100:.1f}%)")
    
    # STEP 4: Create 5-fold cross-validation splits
    print(f"\nSTEP 4: Creating {n_folds}-fold cross-validation splits...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    labels = classifier_dev[target_feature].values
    
    fold_stats = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(classifier_dev, labels)):
        fold_num = fold_idx + 1
        fold_train = classifier_dev.iloc[train_idx]
        fold_val = classifier_dev.iloc[val_idx]
        
        # Save fold data
        fold_train_path = os.path.join(cv_dir, f'fold_{fold_num}_train.csv')
        fold_val_path = os.path.join(cv_dir, f'fold_{fold_num}_val.csv')
        
        fold_train.to_csv(fold_train_path, index=False)
        fold_val.to_csv(fold_val_path, index=False)
        
        # Calculate statistics
        train_pos = (fold_train[target_feature] == 1.0).sum()
        train_neg = (fold_train[target_feature] == 0.0).sum()
        val_pos = (fold_val[target_feature] == 1.0).sum()
        val_neg = (fold_val[target_feature] == 0.0).sum()
        
        # Calculate confidence interval (worst case: 50% accuracy)
        val_size = len(fold_val)
        ci_margin = 1.96 * np.sqrt(0.25 / val_size) * 100
        
        print(f"   Fold {fold_num}:")
        print(f"      Train: {len(fold_train):,} samples ({train_pos:,} pos, {train_neg:,} neg)")
        print(f"      Val:   {len(fold_val):,} samples ({val_pos:,} pos, {val_neg:,} neg)")
        print(f"      Expected CI: ±{ci_margin:.2f}%")
        
        fold_stats.append({
            'Fold': fold_num,
            'Train_Total': len(fold_train),
            'Train_Positive': train_pos,
            'Train_Negative': train_neg,
            'Val_Total': len(fold_val),
            'Val_Positive': val_pos,
            'Val_Negative': val_neg,
            'CI_Margin': f"±{ci_margin:.2f}%"
        })
    
    # Save fold statistics
    fold_stats_df = pd.DataFrame(fold_stats)
    fold_stats_path = os.path.join(cv_dir, 'fold_statistics.csv')
    fold_stats_df.to_csv(fold_stats_path, index=False)
    print(f"   ✓ Fold statistics saved to: {fold_stats_path}")
    
    # STEP 5: Save main datasets
    print(f"\nSTEP 5: Saving main datasets...")
    
    # Save full classifier development set (for training final model)
    classifier_dev_path = os.path.join(output_dir, 'classifier_dev_full.csv')
    classifier_dev.to_csv(classifier_dev_path, index=False)
    print(f"   ✓ Full classifier development data → {classifier_dev_path}")
    
    # Save diffusion fine-tuning data
    diffusion_finetune_path = os.path.join(output_dir, 'diffusion_finetune.csv')
    diffusion_df.to_csv(diffusion_finetune_path, index=False)
    print(f"   ✓ Diffusion fine-tuning data → {diffusion_finetune_path}")
    
    # Save test set (renamed from validation)
    test_path = os.path.join(output_dir, 'test.csv')
    test_filtered.to_csv(test_path, index=False)
    print(f"   ✓ Final test set → {test_path}")
    
    # STEP 6: Create summary statistics
    print(f"\nSUMMARY STATISTICS")
    print("="*80)
    
    total_filtered = len(train_filtered)
    
    print(f"Original filtered training data: {total_filtered:,}")
    print(f"├─ Diffusion fine-tuning: {len(diffusion_df):,} "
          f"({len(diffusion_df)/total_filtered*100:.1f}%)")
    print(f"└─ Classifier development: {len(classifier_dev):,} "
          f"({len(classifier_dev)/total_filtered*100:.1f}%)")
    print(f"   ├─ Used for {n_folds}-fold CV during development")
    print(f"   └─ Full set used for final model training")
    print(f"\nFinal test set: {len(test_filtered):,}")
    print(f"├─ Positive: {len(test_positive):,} ({len(test_positive)/len(test_filtered)*100:.1f}%)")
    print(f"└─ Negative: {len(test_negative):,} ({len(test_negative)/len(test_filtered)*100:.1f}%)")
    
    # Calculate test set confidence interval
    test_size = len(test_filtered)
    test_ci = 1.96 * np.sqrt(0.25 / test_size) * 100
    
    print(f"\nConfidence Intervals (95%, worst-case):")
    print(f"├─ Per CV fold (~{len(fold_val):,} samples): ±{ci_margin:.2f}%")
    print(f"└─ Final test ({test_size} samples): ±{test_ci:.2f}%")
    
    # Create detailed summary file
    summary = {
        'Dataset': [
            'Diffusion Fine-tune',
            'Classifier Dev (Full)',
            'CV Fold Train (avg)',
            'CV Fold Val (avg)',
            'Final Test'
        ],
        'Total_Samples': [
            len(diffusion_df),
            len(classifier_dev),
            int(fold_stats_df['Train_Total'].mean()),
            int(fold_stats_df['Val_Total'].mean()),
            len(test_filtered)
        ],
        'Positive_Cases': [
            len(diffusion_df[diffusion_df[target_feature]==1.0]),
            len(classifier_dev[classifier_dev[target_feature]==1.0]),
            int(fold_stats_df['Train_Positive'].mean()),
            int(fold_stats_df['Val_Positive'].mean()),
            len(test_filtered[test_filtered[target_feature]==1.0])
        ],
        'Negative_Cases': [
            len(diffusion_df[diffusion_df[target_feature]==0.0]),
            len(classifier_dev[classifier_dev[target_feature]==0.0]),
            int(fold_stats_df['Train_Negative'].mean()),
            int(fold_stats_df['Val_Negative'].mean()),
            len(test_filtered[test_filtered[target_feature]==0.0])
        ],
        'Purpose': [
            'Diffusion model training only',
            'Final model training',
            'CV training',
            'CV validation',
            'Final evaluation (touch once!)'
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, 'dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\n   Summary saved to: {summary_path}")
    
    # Verify no data leakage
    print(f"\nVerifying no data leakage between datasets...")
    diffusion_paths = set(diffusion_df['Path'])
    classifier_paths = set(classifier_dev['Path'])
    test_paths = set(test_filtered['Path'])
    
    overlap_diff_class = diffusion_paths.intersection(classifier_paths)
    overlap_diff_test = diffusion_paths.intersection(test_paths)
    overlap_class_test = classifier_paths.intersection(test_paths)
    
    all_clean = True
    
    if len(overlap_diff_class) == 0:
        print(f"   ✓ No overlap between diffusion and classifier data")
    else:
        print(f"   ✗ WARNING: {len(overlap_diff_class)} samples overlap between diffusion and classifier!")
        all_clean = False
    
    if len(overlap_diff_test) == 0:
        print(f"   ✓ No overlap between diffusion and test data")
    else:
        print(f"   ✗ WARNING: {len(overlap_diff_test)} samples overlap between diffusion and test!")
        all_clean = False
        
    if len(overlap_class_test) == 0:
        print(f"   ✓ No overlap between classifier and test data")
    else:
        print(f"   ✗ WARNING: {len(overlap_class_test)} samples overlap between classifier and test!")
        all_clean = False
    
    if all_clean:
        print(f"   ✓ All datasets are independent - no data leakage detected!")
    
    print("\n" + "="*80)
    print("DATA FILTERING AND SPLITTING COMPLETE!")
    print("="*80)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├─ diffusion_finetune.csv       (for diffusion model)")
    print(f"  ├─ classifier_dev_full.csv      (for final model training)")
    print(f"  ├─ test.csv                     (for final evaluation)")
    print(f"  ├─ dataset_summary.csv          (summary statistics)")
    print(f"  └─ cv_splits/")
    print(f"      ├─ fold_1_train.csv         (CV fold 1 training)")
    print(f"      ├─ fold_1_val.csv           (CV fold 1 validation)")
    print(f"      ├─ fold_2_train.csv")
    print(f"      ├─ fold_2_val.csv")
    print(f"      ├─ ...")
    print(f"      ├─ fold_{n_folds}_train.csv")
    print(f"      ├─ fold_{n_folds}_val.csv")
    print(f"      └─ fold_statistics.csv      (CV fold statistics)")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune diffusion model using: diffusion_finetune.csv")
    print(f"  2. Train classifiers using CV folds: cv_splits/fold_*_train.csv")
    print(f"  3. Validate during development using: cv_splits/fold_*_val.csv")
    print(f"  4. Train final model using: classifier_dev_full.csv")
    print(f"  5. Final evaluation ONCE using: test.csv")
    print("="*80 + "\n")
    
    return classifier_dev, diffusion_df, test_filtered, fold_stats_df


def main():
    parser = argparse.ArgumentParser(
        description='Filter and split CheXpert data with 5-fold CV for Pleural Effusion classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 50% positive cases for diffusion, balanced classifier, 5-fold CV
  python filter_and_split_data_cv.py --train_csv train.csv --valid_csv valid.csv
  
  # Use 70% of positive cases for diffusion
  python filter_and_split_data_cv.py --diffusion_ratio 0.7
  
  # Use 10-fold cross-validation
  python filter_and_split_data_cv.py --n_folds 10
  
  # Keep classifier data unbalanced
  python filter_and_split_data_cv.py --no_balance
        """
    )
    
    parser.add_argument('--train_csv', type=str, default='train.csv',
                       help='Path to original training CSV (default: train.csv)')
    parser.add_argument('--valid_csv', type=str, default='valid.csv',
                       help='Path to original validation CSV - will become test set (default: valid.csv)')
    parser.add_argument('--output_dir', type=str, default='data/filtered_splits',
                       help='Output directory for split files (default: data/filtered_splits)')
    parser.add_argument('--target', type=str, default='Pleural Effusion',
                       help='Target medical condition (default: Pleural Effusion)')
    parser.add_argument('--diffusion_ratio', type=float, default=0.5,
                       help='Proportion of positive cases for diffusion (0.0-1.0, default: 0.5)')
    parser.add_argument('--balance_classifier', action='store_true', default=True,
                       help='Balance classifier training data (default: True)')
    parser.add_argument('--no_balance', dest='balance_classifier', action='store_false',
                       help='Do not balance classifier training data')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.diffusion_ratio < 1.0):
        print("ERROR: --diffusion_ratio must be between 0.0 and 1.0")
        return
    
    if args.n_folds < 2:
        print("ERROR: --n_folds must be at least 2")
        return
    
    # Check if input files exist
    if not os.path.exists(args.train_csv):
        print(f"ERROR: Training CSV not found: {args.train_csv}")
        return
    if not os.path.exists(args.valid_csv):
        print(f"ERROR: Validation CSV not found: {args.valid_csv}")
        return
    
    # Run the filtering and splitting
    filter_and_split_data(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        output_dir=args.output_dir,
        target_feature=args.target,
        diffusion_ratio=args.diffusion_ratio,
        balance_classifier=args.balance_classifier,
        n_folds=args.n_folds,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()