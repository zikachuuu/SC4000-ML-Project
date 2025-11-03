import pandas as pd
import numpy as np
import os
from itertools import product
from pathlib import Path
from typing import Dict, Union, List, Tuple
from datetime import datetime
import json

class EnsembleLogger:
    """Logger class to document ensemble operations."""
    
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.log_file = os.path.join(output_folder, 'ensemble_log.txt')
        self.json_log_file = os.path.join(output_folder, 'ensemble_log.json')
        self.log_entries = []
        
        # Create output directory if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self._write_header()
    
    def _write_header(self):
        """Write header to log file."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENSEMBLE PREDICTIONS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Log created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_session_start(self, mode: str, file_paths: Dict[str, str]):
        """Log the start of an ensemble session."""
        message = f"\n{'='*80}\n"
        message += f"SESSION START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"{'='*80}\n"
        message += f"Mode: {mode}\n"
        message += f"Number of models: {len(file_paths)}\n\n"
        message += "Input files:\n"
        for name, path in file_paths.items():
            message += f"  - {name}: {path}\n"
        message += "\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message)
        
        print(message)
    
    def log_file_validation(self, name: str, path: str, num_rows: int, 
                           prediction_min: float, prediction_max: float,
                           prediction_mean: float):
        """Log validation information for each input file."""
        message = f"Validated {name}:\n"
        message += f"  Path: {path}\n"
        message += f"  Rows: {num_rows}\n"
        message += f"  Prediction range: [{prediction_min:.6f}, {prediction_max:.6f}]\n"
        message += f"  Prediction mean: {prediction_mean:.6f}\n\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def log_ensemble_creation(self, weights: Dict[str, float], 
                             original_weights: Dict[str, float],
                             output_path: str,
                             num_predictions: int,
                             prediction_min: float,
                             prediction_max: float,
                             prediction_mean: float,
                             was_normalized: bool):
        """Log information about a created ensemble."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"-" * 80 + "\n"
        message += f"Ensemble created: {timestamp}\n"
        message += f"-" * 80 + "\n"
        
        if was_normalized:
            message += "Original weights (before normalization):\n"
            for name, weight in original_weights.items():
                message += f"  {name}: {weight:.6f}\n"
            message += f"  Sum: {sum(original_weights.values()):.6f}\n\n"
            message += "Normalized weights:\n"
        else:
            message += "Weights:\n"
        
        for name, weight in weights.items():
            message += f"  {name}: {weight:.6f}\n"
        message += f"  Sum: {sum(weights.values()):.6f}\n\n"
        
        message += f"Output file: {os.path.basename(output_path)}\n"
        message += f"Number of predictions: {num_predictions}\n"
        message += f"Prediction range: [{prediction_min:.6f}, {prediction_max:.6f}]\n"
        message += f"Prediction mean: {prediction_mean:.6f}\n"
        message += "\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message)
        
        # Store for JSON log
        log_entry = {
            'timestamp': timestamp,
            'weights': weights,
            'original_weights': original_weights if was_normalized else None,
            'was_normalized': was_normalized,
            'output_file': os.path.basename(output_path),
            'num_predictions': num_predictions,
            'prediction_stats': {
                'min': float(prediction_min),
                'max': float(prediction_max),
                'mean': float(prediction_mean)
            }
        }
        self.log_entries.append(log_entry)
    
    def log_session_end(self, num_ensembles: int, total_time: float):
        """Log the end of an ensemble session."""
        message = f"\n{'='*80}\n"
        message += f"SESSION END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"{'='*80}\n"
        message += f"Total ensembles created: {num_ensembles}\n"
        message += f"Total time: {total_time:.2f} seconds\n"
        message += f"Average time per ensemble: {total_time/num_ensembles:.2f} seconds\n"
        message += f"{'='*80}\n\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message)
        
        print(message)
        
        # Save JSON log
        self._save_json_log()
    
    def log_error(self, error_message: str):
        """Log an error."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"\n!!! ERROR [{timestamp}] !!!\n{error_message}\n\n"
        
        with open(self.log_file, 'a') as f:
            f.write(message)
        
        print(message)
    
    def _save_json_log(self):
        """Save structured log in JSON format."""
        json_data = {
            'log_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_ensembles': len(self.log_entries),
            'ensembles': self.log_entries
        }
        
        with open(self.json_log_file, 'w') as f:
            json.dump(json_data, f, indent=2)

def validate_weights(weights: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    """Validate and normalize weights to sum to 1."""
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        normalized = {name: w / total for name, w in weights.items()}
        return normalized, True
    return weights, False

def ensemble_predictions(
    file_paths: Dict[str, str],
    weights: Dict[str, float],
    output_folder: str,
    logger: EnsembleLogger = None,
    filename_suffix: str = ""
) -> str:
    """
    Ensemble predictions from multiple CSV files.
    
    Args:
        file_paths: Dict mapping model names to file paths
        weights: Dict mapping model names to weights
        output_folder: Folder to save the output
        logger: EnsembleLogger instance for logging
        filename_suffix: Optional suffix for the output filename
    
    Returns:
        Path to the output file
    """
    try:
        # Store original weights
        original_weights = weights.copy()
        
        # Validate weights
        weights, was_normalized = validate_weights(weights)
        
        if was_normalized and logger:
            print(f"Warning: Weights sum to {sum(original_weights.values()):.4f}, normalizing to 1.0")
        
        # Read all prediction files
        predictions = {}
        customer_ids = None
        file_stats = {}
        
        for name, path in file_paths.items():
            df = pd.read_csv(path)
            
            # Validate columns
            if 'customer_ID' not in df.columns or 'prediction' not in df.columns:
                raise ValueError(f"File {path} must contain 'customer_ID' and 'prediction' columns")
            
            # Store predictions
            predictions[name] = df.set_index('customer_ID')['prediction']
            
            # Collect stats for logging
            file_stats[name] = {
                'path': path,
                'num_rows': len(df),
                'min': df['prediction'].min(),
                'max': df['prediction'].max(),
                'mean': df['prediction'].mean()
            }
            
            # Log file validation
            if logger:
                logger.log_file_validation(
                    name, path, len(df),
                    df['prediction'].min(),
                    df['prediction'].max(),
                    df['prediction'].mean()
                )
            
            # Validate customer IDs match across files
            if customer_ids is None:
                customer_ids = set(df['customer_ID'])
            else:
                if set(df['customer_ID']) != customer_ids:
                    raise ValueError(f"Customer IDs in {path} don't match other files")
        
        # Calculate weighted ensemble
        ensemble_pred = pd.Series(0.0, index=predictions[list(predictions.keys())[0]].index)
        
        for name, weight in weights.items():
            ensemble_pred += predictions[name] * weight
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'customer_ID': ensemble_pred.index,
            'prediction': ensemble_pred.values
        })
        
        # Create output filename
        weight_parts = [f"{name}-{weight:.4f}" for name, weight in weights.items()]
        output_filename = f"ensemble_{'_'.join(weight_parts)}"
        if filename_suffix:
            output_filename += f"_{filename_suffix}"
        output_filename += ".csv"
        
        # Create output directory if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Save output
        output_path = os.path.join(output_folder, output_filename)
        output_df.to_csv(output_path, index=False)
        
        # Log ensemble creation
        if logger:
            logger.log_ensemble_creation(
                weights, original_weights, output_path,
                len(output_df),
                ensemble_pred.min(),
                ensemble_pred.max(),
                ensemble_pred.mean(),
                was_normalized
            )
        
        print(f"Saved ensemble to: {output_path}")
        return output_path
    
    except Exception as e:
        if logger:
            logger.log_error(str(e))
        raise

def generate_weight_combinations(
    weight_ranges: Dict[str, Tuple[float, float, float]]
) -> List[Dict[str, float]]:
    """
    Generate all possible weight combinations from ranges.
    
    Args:
        weight_ranges: Dict mapping model names to (min, max, step) tuples
    
    Returns:
        List of weight dictionaries
    """
    names = list(weight_ranges.keys())
    ranges = []
    
    for name in names:
        min_w, max_w, step = weight_ranges[name]
        # Generate range with proper floating point handling
        weights = np.arange(min_w, max_w + step/2, step)
        weights = np.round(weights, decimals=10)  # Handle floating point errors
        ranges.append(weights)
    
    # Generate all combinations
    combinations = []
    for combo in product(*ranges):
        # Only keep combinations that sum to 1.0 (within a small tolerance)
        total = float(np.sum(combo))
        if not np.isclose(total, 1.0, atol=1e-8):
            # skip combos that don't exactly sum to 1. This avoids rescaling
            # and prevents generating too many combinations that would later
            # be normalized.
            continue

        weight_dict = dict(zip(names, combo))
        combinations.append(weight_dict)

    return combinations

def main():
    """
    Main function to run the ensemble script.
    
    Example usage:
    
    # Single weight configuration:
    file_paths = {
        'model1': 'predictions/model1.csv',
        'model2': 'predictions/model2.csv',
        'model3': 'predictions/model3.csv'
    }
    weights = {
        'model1': 0.5,
        'model2': 0.3,
        'model3': 0.2
    }
    output_folder = 'ensemble_results'
    
    logger = EnsembleLogger(output_folder)
    logger.log_session_start('Single Ensemble', file_paths)
    start_time = datetime.now()
    
    ensemble_predictions(file_paths, weights, output_folder, logger)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log_session_end(1, elapsed)
    
    # Multiple weight configurations with ranges:
    weight_ranges = {
        'model1': (0.3, 0.5, 0.1),  # min, max, step
        'model2': (0.2, 0.4, 0.1),
        'model3': (0.2, 0.4, 0.1)
    }
    
    logger = EnsembleLogger(output_folder)
    logger.log_session_start('Grid Search', file_paths)
    start_time = datetime.now()
    
    combinations = generate_weight_combinations(weight_ranges)
    print(f"Generating {len(combinations)} ensemble combinations...")
    
    for i, weights in enumerate(combinations):
        ensemble_predictions(file_paths, weights, output_folder, logger)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(combinations)}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log_session_end(len(combinations), elapsed)
    """
    
    # ========== CONFIGURATION SECTION - MODIFY THIS ==========
    
    # Define your file paths
    file_paths = {
        'run2': r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2 (lightGBM simple auc)\lightgbm_test_predictions.csv",
        'run3': r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 3 (lightGBM time auc)\lightgbm_test_predictions.csv",
        'run4': r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 4 GY (lightGBM_it_2_cv_2_amex_scorer)\lightgbm_amex_scorer_cv_2_it_2_results.csv",
        'nn': r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2 Ensembled\final_submission_nn.csv",
        'run 5': r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 5 GY (lightGBM_simple_summary_it10_cv5_amex_scorer)\lightgbm_test_predictions.csv",
    }

    # Public score
    # run 2 : 0.78831 (lgbm, simple, auc)
    # run 3 : 0.78313 (lgbm, time, auc)
    # run 4 : 0.78963 (lgbm, time, amex scorer) (gy)
    # run 5 : 0.78989 (lgbm, simple, amex scorer) (gy)
    # nn    : 0.78874 (simple, auc)
    
    # Output folder
    output_folder = r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Ensembled Run 2 3 4 5 NN"
    
    # Choose one of the following modes:
    
    # MODE 1: Single weight configuration
    # Uncomment and modify this section for single ensemble
    weights = {
        'run2': 0.15,
        'run3': 0.0,
        'run4': 0.3,
        'run 5': 0.25,
        'nn': 0.3
    }
    
    logger = EnsembleLogger(output_folder)
    logger.log_session_start('Single Ensemble', file_paths)
    start_time = datetime.now()
    
    ensemble_predictions(file_paths, weights, output_folder, logger)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log_session_end(1, elapsed)
    
    
    # MODE 2: Multiple weight configurations with ranges
    # Uncomment and modify this section for grid search
    # weight_ranges = {
    #     'run2': (0, 0.4, 0.05),  # (min, max, step)
    #     'run3': (0, 0.4, 0.05),
    #     'run4': (0.3, 0.6, 0.05),
    #     'run5': (0.3, 0.6, 0.05),
    #     'nn': (0.3, 0.6, 0.05)
    # }
    
    # logger = EnsembleLogger(output_folder)
    # logger.log_session_start('Grid Search', file_paths)
    # start_time = datetime.now()
    
    # combinations = generate_weight_combinations(weight_ranges)
    # print(f"Generating {len(combinations)} ensemble combinations...")
    
    # for i, weights in enumerate(combinations):
    #     ensemble_predictions(file_paths, weights, output_folder, logger)
    #     if (i + 1) % 10 == 0:
    #         print(f"Progress: {i + 1}/{len(combinations)}")
    
    # elapsed = (datetime.now() - start_time).total_seconds()
    # logger.log_session_end(len(combinations), elapsed)
    
    # print(f"\nCompleted! All {len(combinations)} ensembles saved to {output_folder}/")
    # print(f"Check {output_folder}/ensemble_log.txt for detailed logs")
    # print(f"Check {output_folder}/ensemble_log.json for structured logs")

    
    print("Please configure the script by uncommenting and modifying the appropriate mode section.")

if __name__ == "__main__":
    main()