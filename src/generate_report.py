"""
Generate comprehensive report from chest X-ray classification simulation results.

This script analyzes results from baseline, augmented, and fine-tuned experiments,
aggregating metrics across seeds and dataset sizes, and tracking carbon emissions.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_all_results(results_dir):
    """
    Load all results.json files from the results directory.

    Args:
        results_dir (str): Path to results directory

    Returns:
        dict: Dictionary with keys 'baseline', 'augmented', 'finetuned' containing result lists
    """
    results = {
        'baseline': [],
        'augmented': [],
        'finetuned': []
    }

    results_path = Path(results_dir)

    # Find all results.json files
    for json_file in results_path.rglob('results.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Categorize the result
            if data.get('finetuned', False):
                results['finetuned'].append(data)
            elif data.get('augmented', False):
                results['augmented'].append(data)
            else:
                results['baseline'].append(data)

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def create_individual_runs_report(results, output_dir):
    """
    Create detailed report of all individual runs.

    Args:
        results (dict): Results dictionary from load_all_results
        output_dir (str): Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Baseline runs
    baseline_rows = []
    for r in results['baseline']:
        row = {
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'run_number': r['run_number'],
            'train_samples': r['train_samples'],
            'val_samples': r['val_samples'],
            'test_samples': r['test_samples'],
            'num_epochs_trained': r['num_epochs_trained'],
            'val_accuracy': r['final_metrics']['accuracy'],
            'val_precision': r['final_metrics']['precision'],
            'val_recall': r['final_metrics']['recall'],
            'val_f1': r['final_metrics']['f1'],
            'val_auc': r['final_metrics']['auc'],
            'test_accuracy': r['test_metrics']['accuracy'],
            'test_precision': r['test_metrics']['precision'],
            'test_recall': r['test_metrics']['recall'],
            'test_f1': r['test_metrics']['f1'],
            'test_auc': r['test_metrics']['auc'],
            'training_co2_kg': r['training_co2_kg'],
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['test_co2_kg']
        }
        baseline_rows.append(row)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df = baseline_df.sort_values(['data_size', 'random_seed'])
    baseline_df.to_csv(os.path.join(output_dir, 'baseline_individual_runs.csv'), index=False)

    # Augmented runs (non-finetuned)
    augmented_rows = []
    for r in results['augmented']:
        row = {
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'run_number': r['run_number'],
            'gen_ratio': r['gen_ratio'],
            'num_synthetic': r['num_synthetic'],
            'train_samples': r['train_samples'],
            'val_samples': r['val_samples'],
            'test_samples': r['test_samples'],
            'num_epochs_trained': r['num_epochs_trained'],
            'val_accuracy': r['final_metrics']['accuracy'],
            'val_precision': r['final_metrics']['precision'],
            'val_recall': r['final_metrics']['recall'],
            'val_f1': r['final_metrics']['f1'],
            'val_auc': r['final_metrics']['auc'],
            'test_accuracy': r['test_metrics']['accuracy'],
            'test_precision': r['test_metrics']['precision'],
            'test_recall': r['test_metrics']['recall'],
            'test_f1': r['test_metrics']['f1'],
            'test_auc': r['test_metrics']['auc'],
            'training_co2_kg': r['training_co2_kg'],
            'generation_co2_kg': r['generation_co2_kg'],
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['generation_co2_kg'] + r['test_co2_kg']
        }
        augmented_rows.append(row)

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.sort_values(['data_size', 'gen_ratio', 'random_seed'])
    augmented_df.to_csv(os.path.join(output_dir, 'augmented_individual_runs.csv'), index=False)

    # Fine-tuned runs
    finetuned_rows = []
    for r in results['finetuned']:
        row = {
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'run_number': r['run_number'],
            'gen_ratio': r['gen_ratio'],
            'num_synthetic': r['num_synthetic'],
            'finetune_samples': r['finetune_samples'],
            'train_samples': r['train_samples'],
            'val_samples': r['val_samples'],
            'test_samples': r['test_samples'],
            'num_epochs_trained': r['num_epochs_trained'],
            'val_accuracy': r['final_metrics']['accuracy'],
            'val_precision': r['final_metrics']['precision'],
            'val_recall': r['final_metrics']['recall'],
            'val_f1': r['final_metrics']['f1'],
            'val_auc': r['final_metrics']['auc'],
            'test_accuracy': r['test_metrics']['accuracy'],
            'test_precision': r['test_metrics']['precision'],
            'test_recall': r['test_metrics']['recall'],
            'test_f1': r['test_metrics']['f1'],
            'test_auc': r['test_metrics']['auc'],
            'training_co2_kg': r['training_co2_kg'],
            'generation_co2_kg': r['generation_co2_kg'],
            'finetune_co2_kg': r['finetune_co2_kg'],
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['generation_co2_kg'] + r['finetune_co2_kg'] + r['test_co2_kg']
        }
        finetuned_rows.append(row)

    finetuned_df = pd.DataFrame(finetuned_rows)
    finetuned_df = finetuned_df.sort_values(['data_size', 'gen_ratio', 'random_seed'])
    finetuned_df.to_csv(os.path.join(output_dir, 'finetuned_individual_runs.csv'), index=False)

    return baseline_df, augmented_df, finetuned_df


def aggregate_metrics_across_seeds(results, output_dir):
    """
    Aggregate metrics across different random seeds for each configuration.

    Args:
        results (dict): Results dictionary from load_all_results
        output_dir (str): Directory to save the report
    """
    # Baseline aggregation
    baseline_agg = defaultdict(list)
    for r in results['baseline']:
        key = r['data_size']
        baseline_agg[key].append(r)

    baseline_summary_rows = []
    for data_size, runs in sorted(baseline_agg.items()):
        row = {
            'data_size': data_size,
            'num_runs': len(runs),
            'val_accuracy_mean': np.mean([r['final_metrics']['accuracy'] for r in runs]),
            'val_accuracy_std': np.std([r['final_metrics']['accuracy'] for r in runs]),
            'val_auc_mean': np.mean([r['final_metrics']['auc'] for r in runs]),
            'val_auc_std': np.std([r['final_metrics']['auc'] for r in runs]),
            'val_f1_mean': np.mean([r['final_metrics']['f1'] for r in runs]),
            'val_f1_std': np.std([r['final_metrics']['f1'] for r in runs]),
            'test_accuracy_mean': np.mean([r['test_metrics']['accuracy'] for r in runs]),
            'test_accuracy_std': np.std([r['test_metrics']['accuracy'] for r in runs]),
            'test_auc_mean': np.mean([r['test_metrics']['auc'] for r in runs]),
            'test_auc_std': np.std([r['test_metrics']['auc'] for r in runs]),
            'test_f1_mean': np.mean([r['test_metrics']['f1'] for r in runs]),
            'test_f1_std': np.std([r['test_metrics']['f1'] for r in runs]),
            'training_co2_mean': np.mean([r['training_co2_kg'] for r in runs]),
            'training_co2_std': np.std([r['training_co2_kg'] for r in runs]),
            'total_co2_mean': np.mean([r['training_co2_kg'] + r['test_co2_kg'] for r in runs]),
            'total_co2_std': np.std([r['training_co2_kg'] + r['test_co2_kg'] for r in runs])
        }
        baseline_summary_rows.append(row)

    baseline_summary_df = pd.DataFrame(baseline_summary_rows)
    baseline_summary_df.to_csv(os.path.join(output_dir, 'baseline_aggregated.csv'), index=False)

    # Augmented aggregation
    augmented_agg = defaultdict(list)
    for r in results['augmented']:
        key = (r['data_size'], r['gen_ratio'])
        augmented_agg[key].append(r)

    augmented_summary_rows = []
    for (data_size, gen_ratio), runs in sorted(augmented_agg.items()):
        row = {
            'data_size': data_size,
            'gen_ratio': gen_ratio,
            'num_runs': len(runs),
            'val_accuracy_mean': np.mean([r['final_metrics']['accuracy'] for r in runs]),
            'val_accuracy_std': np.std([r['final_metrics']['accuracy'] for r in runs]),
            'val_auc_mean': np.mean([r['final_metrics']['auc'] for r in runs]),
            'val_auc_std': np.std([r['final_metrics']['auc'] for r in runs]),
            'val_f1_mean': np.mean([r['final_metrics']['f1'] for r in runs]),
            'val_f1_std': np.std([r['final_metrics']['f1'] for r in runs]),
            'test_accuracy_mean': np.mean([r['test_metrics']['accuracy'] for r in runs]),
            'test_accuracy_std': np.std([r['test_metrics']['accuracy'] for r in runs]),
            'test_auc_mean': np.mean([r['test_metrics']['auc'] for r in runs]),
            'test_auc_std': np.std([r['test_metrics']['auc'] for r in runs]),
            'test_f1_mean': np.mean([r['test_metrics']['f1'] for r in runs]),
            'test_f1_std': np.std([r['test_metrics']['f1'] for r in runs]),
            'training_co2_mean': np.mean([r['training_co2_kg'] for r in runs]),
            'training_co2_std': np.std([r['training_co2_kg'] for r in runs]),
            'generation_co2_mean': np.mean([r['generation_co2_kg'] for r in runs]),
            'generation_co2_std': np.std([r['generation_co2_kg'] for r in runs]),
            'total_co2_mean': np.mean([r['training_co2_kg'] + r['generation_co2_kg'] + r['test_co2_kg'] for r in runs]),
            'total_co2_std': np.std([r['training_co2_kg'] + r['generation_co2_kg'] + r['test_co2_kg'] for r in runs])
        }
        augmented_summary_rows.append(row)

    augmented_summary_df = pd.DataFrame(augmented_summary_rows)
    augmented_summary_df.to_csv(os.path.join(output_dir, 'augmented_aggregated.csv'), index=False)

    # Fine-tuned aggregation
    finetuned_agg = defaultdict(list)
    for r in results['finetuned']:
        key = (r['data_size'], r['gen_ratio'])
        finetuned_agg[key].append(r)

    finetuned_summary_rows = []
    for (data_size, gen_ratio), runs in sorted(finetuned_agg.items()):
        row = {
            'data_size': data_size,
            'gen_ratio': gen_ratio,
            'num_runs': len(runs),
            'val_accuracy_mean': np.mean([r['final_metrics']['accuracy'] for r in runs]),
            'val_accuracy_std': np.std([r['final_metrics']['accuracy'] for r in runs]),
            'val_auc_mean': np.mean([r['final_metrics']['auc'] for r in runs]),
            'val_auc_std': np.std([r['final_metrics']['auc'] for r in runs]),
            'val_f1_mean': np.mean([r['final_metrics']['f1'] for r in runs]),
            'val_f1_std': np.std([r['final_metrics']['f1'] for r in runs]),
            'test_accuracy_mean': np.mean([r['test_metrics']['accuracy'] for r in runs]),
            'test_accuracy_std': np.std([r['test_metrics']['accuracy'] for r in runs]),
            'test_auc_mean': np.mean([r['test_metrics']['auc'] for r in runs]),
            'test_auc_std': np.std([r['test_metrics']['auc'] for r in runs]),
            'test_f1_mean': np.mean([r['test_metrics']['f1'] for r in runs]),
            'test_f1_std': np.std([r['test_metrics']['f1'] for r in runs]),
            'training_co2_mean': np.mean([r['training_co2_kg'] for r in runs]),
            'training_co2_std': np.std([r['training_co2_kg'] for r in runs]),
            'generation_co2_mean': np.mean([r['generation_co2_kg'] for r in runs]),
            'generation_co2_std': np.std([r['generation_co2_kg'] for r in runs]),
            'finetune_co2_mean': np.mean([r['finetune_co2_kg'] for r in runs]),
            'finetune_co2_std': np.std([r['finetune_co2_kg'] for r in runs]),
            'total_co2_mean': np.mean([r['training_co2_kg'] + r['generation_co2_kg'] + r['finetune_co2_kg'] + r['test_co2_kg'] for r in runs]),
            'total_co2_std': np.std([r['training_co2_kg'] + r['generation_co2_kg'] + r['finetune_co2_kg'] + r['test_co2_kg'] for r in runs])
        }
        finetuned_summary_rows.append(row)

    finetuned_summary_df = pd.DataFrame(finetuned_summary_rows)
    finetuned_summary_df.to_csv(os.path.join(output_dir, 'finetuned_aggregated.csv'), index=False)

    return baseline_summary_df, augmented_summary_df, finetuned_summary_df


def generate_carbon_report(results, output_dir):
    """
    Generate detailed carbon emissions report.

    Args:
        results (dict): Results dictionary from load_all_results
        output_dir (str): Directory to save the report
    """
    carbon_rows = []

    # Baseline emissions
    for r in results['baseline']:
        row = {
            'experiment_type': 'baseline',
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'gen_ratio': None,
            'training_co2_kg': r['training_co2_kg'],
            'generation_co2_kg': 0.0,
            'finetune_co2_kg': 0.0,
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['test_co2_kg']
        }
        carbon_rows.append(row)

    # Augmented emissions
    for r in results['augmented']:
        row = {
            'experiment_type': 'augmented',
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'gen_ratio': r['gen_ratio'],
            'training_co2_kg': r['training_co2_kg'],
            'generation_co2_kg': r['generation_co2_kg'],
            'finetune_co2_kg': 0.0,
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['generation_co2_kg'] + r['test_co2_kg']
        }
        carbon_rows.append(row)

    # Fine-tuned emissions
    for r in results['finetuned']:
        row = {
            'experiment_type': 'finetuned',
            'data_size': r['data_size'],
            'random_seed': r['random_seed'],
            'gen_ratio': r['gen_ratio'],
            'training_co2_kg': r['training_co2_kg'],
            'generation_co2_kg': r['generation_co2_kg'],
            'finetune_co2_kg': r['finetune_co2_kg'],
            'test_co2_kg': r['test_co2_kg'],
            'total_co2_kg': r['training_co2_kg'] + r['generation_co2_kg'] + r['finetune_co2_kg'] + r['test_co2_kg']
        }
        carbon_rows.append(row)

    carbon_df = pd.DataFrame(carbon_rows)
    carbon_df = carbon_df.sort_values(['experiment_type', 'data_size', 'gen_ratio', 'random_seed'])
    carbon_df.to_csv(os.path.join(output_dir, 'carbon_emissions_detailed.csv'), index=False)

    # Summary statistics by experiment type
    carbon_summary_rows = []

    for exp_type in ['baseline', 'augmented', 'finetuned']:
        exp_data = carbon_df[carbon_df['experiment_type'] == exp_type]
        if len(exp_data) > 0:
            row = {
                'experiment_type': exp_type,
                'num_experiments': len(exp_data),
                'total_training_co2_kg': exp_data['training_co2_kg'].sum(),
                'total_generation_co2_kg': exp_data['generation_co2_kg'].sum(),
                'total_finetune_co2_kg': exp_data['finetune_co2_kg'].sum(),
                'total_test_co2_kg': exp_data['test_co2_kg'].sum(),
                'total_co2_kg': exp_data['total_co2_kg'].sum(),
                'mean_total_co2_kg': exp_data['total_co2_kg'].mean(),
                'std_total_co2_kg': exp_data['total_co2_kg'].std()
            }
            carbon_summary_rows.append(row)

    carbon_summary_df = pd.DataFrame(carbon_summary_rows)
    carbon_summary_df.to_csv(os.path.join(output_dir, 'carbon_emissions_summary.csv'), index=False)

    return carbon_df, carbon_summary_df


def generate_text_report(results, baseline_summary_df, augmented_summary_df,
                        finetuned_summary_df, carbon_summary_df, output_dir):
    """
    Generate human-readable text report.

    Args:
        results (dict): Results dictionary from load_all_results
        baseline_summary_df: Baseline aggregated metrics DataFrame
        augmented_summary_df: Augmented aggregated metrics DataFrame
        finetuned_summary_df: Fine-tuned aggregated metrics DataFrame
        carbon_summary_df: Carbon emissions summary DataFrame
        output_dir (str): Directory to save the report
    """
    report_path = os.path.join(output_dir, 'report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHEST X-RAY CLASSIFICATION SIMULATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Overview
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total baseline experiments: {len(results['baseline'])}\n")
        f.write(f"Total augmented experiments: {len(results['augmented'])}\n")
        f.write(f"Total fine-tuned experiments: {len(results['finetuned'])}\n")
        f.write(f"Total experiments: {len(results['baseline']) + len(results['augmented']) + len(results['finetuned'])}\n")
        f.write("\n")

        # Baseline results
        f.write("="*80 + "\n")
        f.write("BASELINE RESULTS (No Synthetic Data)\n")
        f.write("="*80 + "\n\n")

        for _, row in baseline_summary_df.iterrows():
            f.write(f"Dataset Size: {int(row['data_size'])}\n")
            f.write(f"  Number of runs: {int(row['num_runs'])}\n")
            f.write(f"  Validation Metrics:\n")
            f.write(f"    Accuracy: {row['val_accuracy_mean']:.4f} ± {row['val_accuracy_std']:.4f}\n")
            f.write(f"    AUC:      {row['val_auc_mean']:.4f} ± {row['val_auc_std']:.4f}\n")
            f.write(f"    F1 Score: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f}\n")
            f.write(f"  Test Metrics:\n")
            f.write(f"    Accuracy: {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}\n")
            f.write(f"    AUC:      {row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}\n")
            f.write(f"    F1 Score: {row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}\n")
            f.write(f"  Carbon Emissions:\n")
            f.write(f"    Total CO2: {row['total_co2_mean']:.6f} ± {row['total_co2_std']:.6f} kg\n")
            f.write("\n")

        # Augmented results
        if len(augmented_summary_df) > 0:
            f.write("="*80 + "\n")
            f.write("AUGMENTED RESULTS (Non-Finetuned Diffusion Model)\n")
            f.write("="*80 + "\n\n")

            for data_size in sorted(augmented_summary_df['data_size'].unique()):
                f.write(f"Dataset Size: {int(data_size)}\n")
                f.write("-"*80 + "\n")

                size_data = augmented_summary_df[augmented_summary_df['data_size'] == data_size]
                for _, row in size_data.iterrows():
                    f.write(f"  Generation Ratio: {row['gen_ratio']}\n")
                    f.write(f"    Validation Metrics:\n")
                    f.write(f"      Accuracy: {row['val_accuracy_mean']:.4f} ± {row['val_accuracy_std']:.4f}\n")
                    f.write(f"      AUC:      {row['val_auc_mean']:.4f} ± {row['val_auc_std']:.4f}\n")
                    f.write(f"      F1 Score: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f}\n")
                    f.write(f"    Test Metrics:\n")
                    f.write(f"      Accuracy: {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}\n")
                    f.write(f"      AUC:      {row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}\n")
                    f.write(f"      F1 Score: {row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}\n")
                    f.write(f"    Carbon Emissions:\n")
                    f.write(f"      Training:   {row['training_co2_mean']:.6f} ± {row['training_co2_std']:.6f} kg\n")
                    f.write(f"      Generation: {row['generation_co2_mean']:.6f} ± {row['generation_co2_std']:.6f} kg\n")
                    f.write(f"      Total:      {row['total_co2_mean']:.6f} ± {row['total_co2_std']:.6f} kg\n")
                    f.write("\n")

        # Fine-tuned results
        if len(finetuned_summary_df) > 0:
            f.write("="*80 + "\n")
            f.write("FINE-TUNED RESULTS (Fine-Tuned Diffusion Model)\n")
            f.write("="*80 + "\n\n")

            for data_size in sorted(finetuned_summary_df['data_size'].unique()):
                f.write(f"Dataset Size: {int(data_size)}\n")
                f.write("-"*80 + "\n")

                size_data = finetuned_summary_df[finetuned_summary_df['data_size'] == data_size]
                for _, row in size_data.iterrows():
                    f.write(f"  Generation Ratio: {row['gen_ratio']}\n")
                    f.write(f"    Validation Metrics:\n")
                    f.write(f"      Accuracy: {row['val_accuracy_mean']:.4f} ± {row['val_accuracy_std']:.4f}\n")
                    f.write(f"      AUC:      {row['val_auc_mean']:.4f} ± {row['val_auc_std']:.4f}\n")
                    f.write(f"      F1 Score: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f}\n")
                    f.write(f"    Test Metrics:\n")
                    f.write(f"      Accuracy: {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f}\n")
                    f.write(f"      AUC:      {row['test_auc_mean']:.4f} ± {row['test_auc_std']:.4f}\n")
                    f.write(f"      F1 Score: {row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}\n")
                    f.write(f"    Carbon Emissions:\n")
                    f.write(f"      Training:   {row['training_co2_mean']:.6f} ± {row['training_co2_std']:.6f} kg\n")
                    f.write(f"      Generation: {row['generation_co2_mean']:.6f} ± {row['generation_co2_std']:.6f} kg\n")
                    f.write(f"      Fine-tuning:{row['finetune_co2_mean']:.6f} ± {row['finetune_co2_std']:.6f} kg\n")
                    f.write(f"      Total:      {row['total_co2_mean']:.6f} ± {row['total_co2_std']:.6f} kg\n")
                    f.write("\n")

        # Carbon emissions summary
        f.write("="*80 + "\n")
        f.write("CARBON EMISSIONS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for _, row in carbon_summary_df.iterrows():
            f.write(f"{row['experiment_type'].upper()}\n")
            f.write(f"  Number of experiments: {int(row['num_experiments'])}\n")
            f.write(f"  Total training CO2:    {row['total_training_co2_kg']:.6f} kg\n")
            f.write(f"  Total generation CO2:  {row['total_generation_co2_kg']:.6f} kg\n")
            f.write(f"  Total fine-tuning CO2: {row['total_finetune_co2_kg']:.6f} kg\n")
            f.write(f"  Total test CO2:        {row['total_test_co2_kg']:.6f} kg\n")
            f.write(f"  Grand total CO2:       {row['total_co2_kg']:.6f} kg\n")
            f.write(f"  Mean CO2 per exp:      {row['mean_total_co2_kg']:.6f} ± {row['std_total_co2_kg']:.6f} kg\n")
            f.write("\n")

        total_co2 = carbon_summary_df['total_co2_kg'].sum()
        f.write("-"*80 + "\n")
        f.write(f"OVERALL TOTAL CO2 EMISSIONS: {total_co2:.6f} kg\n")
        f.write(f"OVERALL TOTAL CO2 EMISSIONS: {total_co2*1000:.3f} g\n")
        f.write("="*80 + "\n")

    print(f"Text report saved to: {report_path}")


def main():
    """Generate comprehensive reports from simulation results."""
    # Configuration
    results_dir = "./results"
    output_dir = "./reports"

    print("="*80)
    print("GENERATING CHEST X-RAY SIMULATION REPORTS")
    print("="*80)

    # Load all results
    print(f"\nLoading results from: {results_dir}")
    results = load_all_results(results_dir)

    print(f"  Baseline experiments: {len(results['baseline'])}")
    print(f"  Augmented experiments: {len(results['augmented'])}")
    print(f"  Fine-tuned experiments: {len(results['finetuned'])}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual runs report
    print(f"\nGenerating individual runs reports...")
    baseline_df, augmented_df, finetuned_df = create_individual_runs_report(results, output_dir)
    print(f"  Saved: baseline_individual_runs.csv")
    print(f"  Saved: augmented_individual_runs.csv")
    print(f"  Saved: finetuned_individual_runs.csv")

    # Generate aggregated metrics
    print(f"\nGenerating aggregated metrics across seeds...")
    baseline_summary_df, augmented_summary_df, finetuned_summary_df = aggregate_metrics_across_seeds(results, output_dir)
    print(f"  Saved: baseline_aggregated.csv")
    print(f"  Saved: augmented_aggregated.csv")
    print(f"  Saved: finetuned_aggregated.csv")

    # Generate carbon report
    print(f"\nGenerating carbon emissions reports...")
    carbon_df, carbon_summary_df = generate_carbon_report(results, output_dir)
    print(f"  Saved: carbon_emissions_detailed.csv")
    print(f"  Saved: carbon_emissions_summary.csv")

    # Generate text report
    print(f"\nGenerating human-readable text report...")
    generate_text_report(results, baseline_summary_df, augmented_summary_df,
                        finetuned_summary_df, carbon_summary_df, output_dir)

    print("\n" + "="*80)
    print(f"REPORT GENERATION COMPLETE")
    print(f"All reports saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
