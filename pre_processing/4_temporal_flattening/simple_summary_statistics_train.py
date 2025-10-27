"""
================================================================================
SIMPLE SUMMARY STATISTICS PREPROCESSING - MEMORY-EFFICIENT MULTI-FILE VERSION
================================================================================

Purpose: Aggregate temporal credit card data using basic statistics
Best for: Decision Trees, Random Forests, Neural Networks

Input:  Multiple split CSV files (customer_split_0.csv to customer_split_4.csv)
        Processes each file separately to avoid memory issues

Output: Aggregated dataset with one row per customer
        Features: last, mean, std, min, max for each numerical feature
                  last, mode for categorical features

Author: Credit Risk Team
Date: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple
import os
import gc
warnings.filterwarnings('ignore')


class SimpleSummaryPreprocessor:
    """
    Preprocess temporal credit card data using simple summary statistics.
    
    This class aggregates multiple statements per customer into a single row
    using basic statistical measures. It's fast, interpretable, and works well
    with tree-based models.
    
    Attributes:
        categorical_features (List[str]): List of categorical feature names
        numerical_features (List[str]): List of numerical feature names
        
    Methods:
        fit_transform: Process raw data and return aggregated features
        transform: Apply fitted preprocessing to new data
    """
    
    def __init__(self):
        """
        Initialize the preprocessor with known categorical features.
        
        These features are categorical based on the American Express dataset
        documentation. They represent discrete states or categories.
        """
        # Known categorical features from the dataset
        self.categorical_features = [
            'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 
            'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
        ]
        
        self.numerical_features = None
        self.feature_names = None
        
    def identify_feature_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features from the dataset.
        
        Args:
            data (pd.DataFrame): Raw temporal data
            
        Returns:
            Tuple[List[str], List[str]]: (numerical_features, categorical_features)
            
        Notes:
            - Excludes customer_ID, S_2 (date), and target from features
            - Treats specified columns as categorical
            - All others are treated as numerical
        """
        # Get all feature columns (exclude identifiers and target)
        all_columns = data.columns.tolist()
        exclude_columns = ['customer_ID', 'S_2', 'target']
        feature_columns = [col for col in all_columns if col not in exclude_columns]
        
        # Separate numerical and categorical
        numerical = [col for col in feature_columns 
                    if col not in self.categorical_features]
        categorical = [col for col in feature_columns 
                      if col in self.categorical_features]
        
        print(f"Identified {len(numerical)} numerical features")
        print(f"Identified {len(categorical)} categorical features")
        
        return numerical, categorical
    
    def aggregate_customer_simple(self, customer_group: pd.DataFrame, 
                                  numerical_features: List[str],
                                  categorical_features: List[str]) -> Dict:
        """
        Aggregate a single customer's temporal data using summary statistics.
        
        Args:
            customer_group (pd.DataFrame): All statements for one customer
            numerical_features (List[str]): List of numerical feature names
            categorical_features (List[str]): List of categorical feature names
            
        Returns:
            Dict: Dictionary of aggregated features for the customer
            
        Aggregation Strategy:
            Numerical features:
                - last: Most recent statement value (captures current state)
                - mean: Average across all statements (captures typical behavior)
                - std: Standard deviation (captures volatility/stability)
                - min: Minimum value (captures best case)
                - max: Maximum value (captures worst case)
                
            Categorical features:
                - last: Most recent category
                - mode: Most frequent category across all statements
                
            Meta features:
                - num_statements: Number of statements available for customer
        """
        agg_dict = {}
        
        # Customer identifier
        agg_dict['customer_ID'] = customer_group['customer_ID'].iloc[0]
        
        # Target variable (same across all rows for a customer)
        if 'target' in customer_group.columns:
            agg_dict['target'] = customer_group['target'].iloc[0]
        
        # Meta feature: number of statements
        # More statements = more data = potentially more reliable features
        agg_dict['num_statements'] = len(customer_group)
        
        # Aggregate numerical features
        for col in numerical_features:
            if col not in customer_group.columns:
                continue
                
            series = customer_group[col]
            
            # Last value: Most recent behavior (often most predictive)
            agg_dict[f'{col}_last'] = series.iloc[-1] if len(series) > 0 else np.nan
            
            # Mean: Historical average behavior
            agg_dict[f'{col}_mean'] = series.mean()
            
            # Standard deviation: Volatility measure
            # High std = erratic behavior (risky)
            # Low std = stable behavior (safer)
            agg_dict[f'{col}_std'] = series.std()
            
            # Minimum: Best historical behavior
            agg_dict[f'{col}_min'] = series.min()
            
            # Maximum: Worst historical behavior
            agg_dict[f'{col}_max'] = series.max()
        
        # Aggregate categorical features
        for col in categorical_features:
            if col not in customer_group.columns:
                continue
                
            series = customer_group[col]
            
            # Last value: Current category
            agg_dict[f'{col}_last'] = series.iloc[-1] if len(series) > 0 else np.nan
            
            # Mode: Most common category
            mode_values = series.mode()
            agg_dict[f'{col}_mode'] = mode_values.iloc[0] if len(mode_values) > 0 else np.nan
        
        return agg_dict
    
    def fit_transform_single_file(self, data: pd.DataFrame, file_index: int) -> pd.DataFrame:
        """
        Fit the preprocessor and transform data from a single file.
        
        Args:
            data (pd.DataFrame): Raw temporal data with multiple rows per customer
            file_index (int): Index of the file being processed (for display)
            
        Returns:
            pd.DataFrame: Aggregated data with one row per customer
        """
        print("="*80)
        print(f"PROCESSING FILE {file_index}")
        print("="*80)
        
        # Validate required columns
        required_cols = ['customer_ID']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column to datetime if present
        if 'S_2' in data.columns:
            print("\nConverting S_2 to datetime...")
            data['S_2'] = pd.to_datetime(data['S_2'], errors='coerce')
            
            # Sort by customer and date to ensure temporal order
            print("Sorting data by customer_ID and date...")
            data = data.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)
        else:
            print("\nWarning: No S_2 (date) column found. Data will not be sorted temporally.")
            data = data.sort_values('customer_ID').reset_index(drop=True)
        
        # Identify feature types (only on first file)
        if self.numerical_features is None:
            print("\nIdentifying feature types...")
            numerical, categorical = self.identify_feature_types(data)
            self.numerical_features = numerical
            self.categorical_features = categorical
        
        # Show data statistics
        print(f"\nData Statistics:")
        print(f"  Total rows: {len(data):,}")
        print(f"  Unique customers: {data['customer_ID'].nunique():,}")
        print(f"  Avg statements per customer: {len(data) / data['customer_ID'].nunique():.1f}")
        if 'target' in data.columns:
            default_rate = data.groupby('customer_ID')['target'].first().mean()
            print(f"  Default rate: {default_rate:.2%}")
        
        # Aggregate customer data
        print(f"\nAggregating {data['customer_ID'].nunique():,} customers...")
        
        aggregated_list = []
        
        # Group by customer and aggregate
        for i, (customer_id, group) in enumerate(data.groupby('customer_ID')):
            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} customers...")
            
            agg_dict = self.aggregate_customer_simple(
                group, 
                self.numerical_features,
                self.categorical_features
            )
            aggregated_list.append(agg_dict)
        
        # Create aggregated DataFrame
        agg_df = pd.DataFrame(aggregated_list)
        
        # Summary of results
        print(f"\n{'='*80}")
        print(f"FILE {file_index} COMPLETE")
        print(f"{'='*80}")
        print(f"Input shape: {data.shape}")
        print(f"Output shape: {agg_df.shape}")
        
        return agg_df
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature types.
        
        Args:
            data (pd.DataFrame): Raw temporal data
            
        Returns:
            pd.DataFrame: Aggregated data
            
        Note:
            Must call fit_transform before transform
        """
        if self.numerical_features is None or self.categorical_features is None:
            raise ValueError("Must call fit_transform_single_file before transform")
        
        print("Transforming new data...")
        
        # Sort data
        if 'S_2' in data.columns:
            data['S_2'] = pd.to_datetime(data['S_2'], errors='coerce')
            data = data.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)
        else:
            data = data.sort_values('customer_ID').reset_index(drop=True)
        
        # Aggregate
        aggregated_list = []
        for customer_id, group in data.groupby('customer_ID'):
            agg_dict = self.aggregate_customer_simple(
                group,
                self.numerical_features,
                self.categorical_features
            )
            aggregated_list.append(agg_dict)
        
        agg_df = pd.DataFrame(aggregated_list)
        print(f"Transformed shape: {agg_df.shape}")
        
        return agg_df


# ================================================================================
# USAGE EXAMPLE - MULTI-FILE PROCESSING
# ================================================================================

def process_multiple_files(file_pattern='customer_split', num_files=5, output_file='train_data_aggregated_simple.csv'):
    """
    Process multiple split files and combine aggregated results.
    
    Args:
        file_pattern (str): Base name of split files (default: 'customer_split')
        num_files (int): Number of split files to process (default: 5)
        output_file (str): Name of output file (default: 'train_data_aggregated_simple.csv')
    
    Returns:
        pd.DataFrame: Combined aggregated data
    """
    print("="*80)
    print("MULTI-FILE PREPROCESSING - MEMORY EFFICIENT")
    print("="*80)
    print(f"\nProcessing {num_files} files...")
    print(f"Pattern: {file_pattern}_X.csv")
    print()
    
    # Initialize preprocessor
    preprocessor = SimpleSummaryPreprocessor()
    
    # List to store aggregated results from each file
    all_aggregated = []
    
    # Process each file separately
    for i in range(num_files):
        file_name = f'{file_pattern}_{i}.csv'
        
        print(f"\n{'#'*80}")
        print(f"# LOADING FILE {i}: {file_name}")
        print(f"{'#'*80}\n")
        
        # Check if file exists
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} not found. Skipping...")
            continue
        
        # Load current file
        print(f"Loading {file_name}...")
        data = pd.read_csv(file_name)
        print(f"Loaded {len(data):,} rows, {len(data.columns)} columns")
        
        # Process and aggregate
        agg_df = preprocessor.fit_transform_single_file(data, i)
        
        # Append to results list
        all_aggregated.append(agg_df)
        
        # Free memory
        del data
        del agg_df
        gc.collect()
        
        print(f"\nMemory cleared. Ready for next file.\n")
    
    # Combine all aggregated DataFrames
    print("="*80)
    print("COMBINING ALL AGGREGATED DATA")
    print("="*80)
    
    if len(all_aggregated) == 0:
        raise ValueError("No files were successfully processed!")
    
    combined_df = pd.concat(all_aggregated, ignore_index=True)
    
    print(f"\nCombined shape: {combined_df.shape}")
    print(f"Total customers: {len(combined_df):,}")
    
    # Check for duplicates (shouldn't happen if split was done correctly)
    duplicates = combined_df['customer_ID'].duplicated().sum()
    if duplicates > 0:
        print(f"\nWarning: Found {duplicates} duplicate customer_IDs!")
        print("Removing duplicates...")
        combined_df = combined_df.drop_duplicates(subset=['customer_ID'], keep='first')
        print(f"New shape: {combined_df.shape}")
    
    # Save combined results
    print(f"\nSaving aggregated data to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    print("Save complete!")
    
    # Feature statistics
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Features created: {combined_df.shape[1] - 2}")  # Exclude customer_ID and target
    print(f"  - Numerical features: {len(preprocessor.numerical_features)} × 5 = {len(preprocessor.numerical_features) * 5}")
    print(f"  - Categorical features: {len(preprocessor.categorical_features)} × 2 = {len(preprocessor.categorical_features) * 2}")
    print(f"  - Meta features: 1 (num_statements)")
    
    # Check for missing values
    missing_pct = (combined_df.isnull().sum().sum() / (combined_df.shape[0] * combined_df.shape[1])) * 100
    print(f"\nMissing values: {missing_pct:.2f}% of total cells")
    
    if 'target' in combined_df.columns:
        default_rate = combined_df['target'].mean()
        print(f"Overall default rate: {default_rate:.2%}")
    
    # Display sample
    print(f"\nSample of final aggregated data:")
    print(combined_df.head())
    
    return combined_df


if __name__ == "__main__":
    # Process all 5 split files
    aggregated_data = process_multiple_files(
        file_pattern='customer_split',
        num_files=5,
        output_file='train_data_aggregated_simple.csv'
    )