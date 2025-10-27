import pandas as pd
import numpy as np
import os
from pathlib import Path

def ensemble_predictions(
    pred_file1: str,
    pred_file2: str,
    output_dir: str = 'ensemble_predictions',
    weight_step: float = 0.1,
    model1_name: str = 'model1',
    model2_name: str = 'model2'
):
    """
    Ensemble two prediction files with different weight combinations.
    
    Parameters:
    -----------
    pred_file1 : str
        Path to first predictions CSV file (columns: customer_ID, prediction)
    pred_file2 : str
        Path to second predictions CSV file (columns: customer_ID, prediction)
    output_dir : str
        Directory to save ensemble predictions
    weight_step : float
        Step size for weight increments (default: 0.1)
    model1_name : str
        Name for first model (used in output filename)
    model2_name : str
        Name for second model (used in output filename)
    
    Returns:
    --------
    dict : Dictionary mapping weights to output file paths
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load prediction files
    print(f"Loading predictions from {pred_file1}...")
    pred1 = pd.read_csv(pred_file1)
    print(f"  - Loaded {len(pred1)} predictions from {model1_name}")
    
    print(f"Loading predictions from {pred_file2}...")
    pred2 = pd.read_csv(pred_file2)
    print(f"  - Loaded {len(pred2)} predictions from {model2_name}")
    
    # Validate columns
    required_cols = ['customer_ID', 'prediction']
    if not all(col in pred1.columns for col in required_cols):
        raise ValueError(f"pred_file1 must have columns: {required_cols}")
    if not all(col in pred2.columns for col in required_cols):
        raise ValueError(f"pred_file2 must have columns: {required_cols}")
    
    # Merge on customer_ID
    print("\nMerging predictions on customer_ID...")
    merged = pred1.merge(
        pred2, 
        on='customer_ID', 
        how='inner',
        suffixes=('_model1', '_model2')
    )
    
    print(f"  - Merged dataset has {len(merged)} rows")
    
    # Check for missing customers
    missing_in_pred2 = set(pred1['customer_ID']) - set(pred2['customer_ID'])
    missing_in_pred1 = set(pred2['customer_ID']) - set(pred1['customer_ID'])
    
    if missing_in_pred2:
        print(f"  ⚠ Warning: {len(missing_in_pred2)} customers in {model1_name} not found in {model2_name}")
    if missing_in_pred1:
        print(f"  ⚠ Warning: {len(missing_in_pred1)} customers in {model2_name} not found in {model1_name}")
    
    # Generate weight combinations
    weights = np.arange(0.1, 1.0, weight_step)
    weights = np.round(weights, 1)  # Round to avoid floating point errors
    
    print(f"\nGenerating {len(weights)} ensemble combinations...")
    print("=" * 80)
    
    output_files = {}
    
    for weight1 in weights:
        weight2 = round(1.0 - weight1, 1)
        
        # Calculate weighted average
        ensemble_pred = (
            weight1 * merged['prediction_model1'] + 
            weight2 * merged['prediction_model2']
        )
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'customer_ID': merged['customer_ID'],
            'prediction': ensemble_pred
        })
        
        # Generate output filename
        output_filename = f'ensemble_{model1_name}_{weight1:.1f}_{model2_name}_{weight2:.1f}.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        # Store in dictionary
        output_files[f"{weight1:.1f}_{weight2:.1f}"] = output_path
        
        # Print summary
        print(f"Weight: {model1_name}={weight1:.1f}, {model2_name}={weight2:.1f}")
        print(f"  - Mean prediction: {ensemble_pred.mean():.4f}")
        print(f"  - Median prediction: {ensemble_pred.median():.4f}")
        print(f"  - Predictions > 0.5: {(ensemble_pred > 0.5).sum()} ({(ensemble_pred > 0.5).mean()*100:.1f}%)")
        print(f"  - Saved to: {output_path}")
        print("-" * 80)
    
    print("=" * 80)
    print(f"\n✓ Successfully created {len(output_files)} ensemble files in '{output_dir}/'")
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ENSEMBLE PREDICTIONS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model 1: {model1_name} ({pred_file1})\n")
        f.write(f"Model 2: {model2_name} ({pred_file2})\n")
        f.write(f"Total customers: {len(merged)}\n\n")
        f.write("Weight Combinations:\n")
        f.write("-" * 80 + "\n")
        
        for weight_combo, filepath in output_files.items():
            f.write(f"{weight_combo.replace('_', ' / ')}: {filepath}\n")
    
    print(f"Summary saved to: {summary_path}\n")
    
    return output_files


def ensemble_predictions_with_stats(
    pred_file1: str,
    pred_file2: str,
    output_dir: str = 'ensemble_predictions',
    weight_step: float = 0.1,
    model1_name: str = 'model1',
    model2_name: str = 'model2'
):
    """
    Enhanced version with detailed statistics for each ensemble.
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from {pred_file1}...")
    pred1 = pd.read_csv(pred_file1)
    print(f"  - Loaded {len(pred1)} predictions from {model1_name}")
    
    print(f"Loading predictions from {pred_file2}...")
    pred2 = pd.read_csv(pred_file2)
    print(f"  - Loaded {len(pred2)} predictions from {model2_name}")
    
    # Merge predictions
    print("\nMerging predictions on customer_ID...")
    merged = pred1.merge(
        pred2, 
        on='customer_ID', 
        how='inner',
        suffixes=('_model1', '_model2')
    )
    print(f"  - Merged dataset has {len(merged)} rows")
    
    # Generate weights
    weights = np.arange(0.1, 1.0, weight_step)
    weights = np.round(weights, 1)
    
    print(f"\nGenerating {len(weights)} ensemble combinations...")
    print("=" * 80)
    
    output_files = {}
    stats_list = []
    
    for weight1 in weights:
        weight2 = round(1.0 - weight1, 1)
        
        # Calculate ensemble
        ensemble_pred = (
            weight1 * merged['prediction_model1'] + 
            weight2 * merged['prediction_model2']
        )
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'customer_ID': merged['customer_ID'],
            'prediction': ensemble_pred
        })
        
        # Generate filename
        output_filename = f'ensemble_{model1_name}_{weight1:.1f}_{model2_name}_{weight2:.1f}.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        output_files[f"{weight1:.1f}_{weight2:.1f}"] = output_path
        
        # Calculate statistics
        stats = {
            'weight1': weight1,
            'weight2': weight2,
            'mean': ensemble_pred.mean(),
            'median': ensemble_pred.median(),
            'std': ensemble_pred.std(),
            'min': ensemble_pred.min(),
            'max': ensemble_pred.max(),
            'q25': ensemble_pred.quantile(0.25),
            'q75': ensemble_pred.quantile(0.75),
            'pred_above_0.5': (ensemble_pred > 0.5).sum(),
            'pct_above_0.5': (ensemble_pred > 0.5).mean() * 100,
            'filename': output_filename
        }
        stats_list.append(stats)
        
        # Print summary
        print(f"Weight: {model1_name}={weight1:.1f}, {model2_name}={weight2:.1f}")
        print(f"  - Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f} | Std: {stats['std']:.4f}")
        print(f"  - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  - Pred > 0.5: {stats['pred_above_0.5']} ({stats['pct_above_0.5']:.1f}%)")
        print(f"  - File: {output_path}")
        print("-" * 80)
    
    # Create statistics DataFrame
    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(output_dir, 'ensemble_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    
    print("=" * 80)
    print(f"\n✓ Created {len(output_files)} ensemble files in '{output_dir}/'")
    print(f"✓ Statistics saved to: {stats_csv_path}\n")
    
    return output_files, stats_df


# Example usage
if __name__ == "__main__":
    
    # Basic usage
    print("BASIC ENSEMBLE\n")
    output_files = ensemble_predictions(
        pred_file1=r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2\lightgbm_test_predictions.csv",
        pred_file2=r"C:\Users\leyan\Downloads\final_submission_nn.csv",
        output_dir=r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2 Ensembled",
        weight_step=0.1,
        model1_name='lightgbm',
        model2_name='nn'
    )
    
    # print("\n" + "=" * 80)
    # print("ENSEMBLE WITH DETAILED STATISTICS\n")
    
    # # Enhanced version with statistics
    # output_files, stats_df = ensemble_predictions_with_stats(
    #     pred_file1='lightgbm_test_predictions.csv',
    #     pred_file2='xgboost_test_predictions.csv',
    #     output_dir='ensemble_outputs_detailed',
    #     weight_step=0.1,
    #     model1_name='lightgbm',
    #     model2_name='xgboost'
    # )
    
    # Display statistics summary
    # print("\nSTATISTICS SUMMARY:")
    # print(stats_df.to_string(index=False))