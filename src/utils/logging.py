"""
Logging utilities for training and experiments.
"""
import os
import logging
from datetime import datetime


def setup_logger(name, log_dir='./log', log_file=None, level=logging.INFO):
    """
    Setup a logger that logs to both console and file

    Args:
        name (str): Name of the logger
        log_dir (str): Directory to save log files (default: './log')
        log_file (str): Specific log file name (if None, will auto-generate)
        level: Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'

    log_path = os.path.join(log_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (detailed format)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_path}")

    return logger


def get_logger(name):
    """
    Get an existing logger by name

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def create_summary_log(output_dir, config_dict, log_dir='./log'):
    """
    Create a summary log file for an experiment

    Args:
        output_dir (str): Directory to save the summary log
        config_dict (dict): Configuration dictionary to log
        log_dir (str): Directory for log files

    Returns:
        str: Path to the summary log file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary_log = os.path.join(
        log_dir,
        f'simulation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    with open(summary_log, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHEST X-RAY CLASSIFICATION SIMULATION\n")
        f.write("="*80 + "\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

        f.write("="*80 + "\n\n")

    return summary_log


def log_run_result(log_file, run_number, total_runs, data_size, random_seed,
                   results=None, error=None):
    """
    Log the result of a single run to the summary log

    Args:
        log_file (str): Path to the summary log file
        run_number (int): Current run number
        total_runs (int): Total number of runs
        data_size (int): Dataset size for this run
        random_seed (int): Random seed for this run
        results (dict, optional): Results dictionary if run succeeded
        error (Exception, optional): Error if run failed
    """
    with open(log_file, 'a') as f:
        f.write(f"\nRun {run_number}/{total_runs} - Size: {data_size}, Seed: {random_seed}\n")

        if results is not None:
            f.write(f"  Val Accuracy: {results['final_metrics']['accuracy']:.4f}\n")
            f.write(f"  Val AUC: {results['final_metrics']['auc']:.4f}\n")

            if 'test_metrics' in results:
                f.write(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}\n")
                f.write(f"  Test AUC: {results['test_metrics']['auc']:.4f}\n")

            f.write(f"  Training CO2: {results['training_co2_kg']:.6f} kg\n")

            if 'test_co2_kg' in results:
                f.write(f"  Test CO2: {results['test_co2_kg']:.6f} kg\n")

            f.write(f"  Status: SUCCESS\n")
        else:
            f.write(f"  Status: FAILED\n")
            f.write(f"  Error: {str(error)}\n")


def finalize_summary_log(log_file, num_completed, total_runs):
    """
    Write final statistics to the summary log

    Args:
        log_file (str): Path to the summary log file
        num_completed (int): Number of completed runs
        total_runs (int): Total number of runs
    """
    with open(log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completed runs: {num_completed}/{total_runs}\n")
        f.write("="*80 + "\n")
