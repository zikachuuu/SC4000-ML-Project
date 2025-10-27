"""
================================================================================
SIMPLE SUMMARY STATISTICS PREPROCESSING - MEMORY-EFFICIENT CHUNKED VERSION
================================================================================

Purpose: Aggregate temporal credit card data using basic statistics
Best for: Decision Trees, Random Forests, Neural Networks

Input:  Multiple split CSV files (customer_split_0.csv to customer_split_29.csv)
        Processes each file in chunks to avoid memory issues

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
    CHUNKED VERSION - Memory efficient for large files.
    """
    
    def __init__(self):
        """Initialize the preprocessor with known categorical features."""
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
        """
        agg_dict = {}
        
        # Customer identifier
        agg_dict['customer_ID'] = customer_group['customer_ID'].iloc[0]
        
        # Target variable (same across all rows for a customer)
        if 'target' in customer_group.columns:
            agg_dict['target'] = customer_group['target'].iloc[0]
        
        # Meta feature: number of statements
        agg_dict['num_statements'] = len(customer_group)
        
        # Aggregate numerical features
        for col in numerical_features:
            if col not in customer_group.columns:
                continue
                
            series = customer_group[col]
            
            # Last value: Most recent behavior
            agg_dict[f'{col}_last'] = series.iloc[-1] if len(series) > 0 else np.nan
            
            # Mean: Historical average behavior
            agg_dict[f'{col}_mean'] = series.mean()
            
            # Standard deviation: Volatility measure
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
    
    def fit_transform_single_file_chunked(self, file_name: str, file_index: int, 
                                         chunk_size: int = 50000) -> pd.DataFrame:
        """
        Fit the preprocessor and transform data from a single file using chunks.
        MEMORY EFFICIENT VERSION - doesn't load entire file at once.
        """
        print("="*80)
        print(f"PROCESSING FILE {file_index}: {file_name}")
        print("="*80)
        
        # Step 1: Identify feature types (from first chunk only)
        if self.numerical_features is None:
            print("\nIdentifying feature types from first chunk...")
            first_chunk = pd.read_csv(file_name, nrows=10000)
            numerical, categorical = self.identify_feature_types(first_chunk)
            self.numerical_features = numerical
            self.categorical_features = categorical
            del first_chunk
            gc.collect()
        
        # Step 2: Collect data per customer across all chunks
        print(f"\nReading file in chunks of {chunk_size:,} rows...")
        customer_data = {}  # Dictionary to accumulate rows per customer
        
        chunk_num = 0
        total_rows = 0
        
        for chunk in pd.read_csv(file_name, chunksize=chunk_size):
            chunk_num += 1
            total_rows += len(chunk)
            
            # Convert date if present
            if 'S_2' in chunk.columns:
                chunk['S_2'] = pd.to_datetime(chunk['S_2'], errors='coerce')
            
            # Sort chunk by customer and date
            chunk = chunk.sort_values(['customer_ID', 'S_2'] if 'S_2' in chunk.columns else ['customer_ID'])
            
            # Group by customer in this chunk
            for customer_id, group in chunk.groupby('customer_ID'):
                if customer_id not in customer_data:
                    customer_data[customer_id] = []
                customer_data[customer_id].append(group)
            
            print(f"  Processed chunk {chunk_num} ({total_rows:,} rows, {len(customer_data):,} customers so far)...", end='\r')
            
            # Free chunk memory
            del chunk
            gc.collect()
        
        print(f"\n\nTotal rows processed: {total_rows:,}")
        print(f"Unique customers: {len(customer_data):,}")
        
        # Step 3: Aggregate each customer's data
        print(f"\nAggregating {len(customer_data):,} customers...")
        aggregated_list = []
        
        for i, (customer_id, chunk_list) in enumerate(customer_data.items()):
            # Combine all chunks for this customer
            customer_df = pd.concat(chunk_list, ignore_index=True)
            
            # Sort by date to ensure temporal order
            if 'S_2' in customer_df.columns:
                customer_df = customer_df.sort_values('S_2').reset_index(drop=True)
            
            # Aggregate
            agg_dict = self.aggregate_customer_simple(
                customer_df,
                self.numerical_features,
                self.categorical_features
            )
            aggregated_list.append(agg_dict)
            
            # Progress indicator
            if (i + 1) % 5000 == 0:
                print(f"  Aggregated {i+1:,} customers...", end='\r')
            
            # Free memory periodically
            if (i + 1) % 10000 == 0:
                gc.collect()
        
        print(f"\n  Aggregated {len(aggregated_list):,} customers - COMPLETE")
        
        # Create aggregated DataFrame
        agg_df = pd.DataFrame(aggregated_list)
        
        # Free customer_data
        del customer_data
        gc.collect()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"FILE {file_index} COMPLETE")
        print(f"{'='*80}")
        print(f"Output shape: {agg_df.shape}")
        
        return agg_df


def process_multiple_files_chunked(file_pattern='customer_split', num_files=30, 
                                   output_file='train_data_aggregated_simple.csv',
                                   chunk_size=50000):
    """
    Process multiple split files using chunked reading.
    MEMORY EFFICIENT for large files.
    """
    print("="*80)
    print("MULTI-FILE PREPROCESSING - CHUNKED MEMORY EFFICIENT")
    print("="*80)
    print(f"\nProcessing {num_files} files...")
    print(f"Chunk size: {chunk_size:,} rows")
    print()
    
    preprocessor = SimpleSummaryPreprocessor()
    all_aggregated = []
    
    for i in range(num_files):
        file_name = f'{file_pattern}_{i}.csv'
        
        print(f"\n{'#'*80}")
        print(f"# FILE {i}: {file_name}")
        print(f"{'#'*80}\n")
        
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} not found. Skipping...")
            continue
        
        # Check file size
        file_size_gb = os.path.getsize(file_name) / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")
        
        # Process file in chunks
        agg_df = preprocessor.fit_transform_single_file_chunked(file_name, i, chunk_size)
        
        # Save intermediate result (optional but recommended)
        intermediate_file = f'aggregated_{file_pattern}_{i}.csv'
        print(f"\nSaving intermediate result: {intermediate_file}")
        agg_df.to_csv(intermediate_file, index=False)
        
        all_aggregated.append(agg_df)
        
        del agg_df
        gc.collect()
        
        print(f"\nFile {i} complete. Memory cleared.\n")
    
    # Combine results
    print("="*80)
    print("COMBINING ALL AGGREGATED DATA")
    print("="*80)
    
    combined_df = pd.concat(all_aggregated, ignore_index=True)
    
    print(f"\nCombined shape: {combined_df.shape}")
    print(f"Total customers: {len(combined_df):,}")
    
    # Check for duplicates
    duplicates = combined_df['customer_ID'].duplicated().sum()
    if duplicates > 0:
        print(f"\nWarning: Found {duplicates} duplicate customer_IDs!")
        combined_df = combined_df.drop_duplicates(subset=['customer_ID'], keep='first')
        print(f"New shape after deduplication: {combined_df.shape}")
    
    # Save final result
    print(f"\nSaving final aggregated data: {output_file}")
    combined_df.to_csv(output_file, index=False)
    print("Save complete!")
    
    # Feature statistics
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total features: {combined_df.shape[1]}")
    print(f"Total customers: {len(combined_df):,}")
    
    if 'target' in combined_df.columns:
        default_rate = combined_df['target'].mean()
        print(f"Overall default rate: {default_rate:.2%}")
    
    # Check for missing values
    missing_pct = (combined_df.isnull().sum().sum() / (combined_df.shape[0] * combined_df.shape[1])) * 100
    print(f"Missing values: {missing_pct:.2f}% of total cells")
    
    print(f"\nSample of final data:")
    print(combined_df.head())
    
    return combined_df


if __name__ == "__main__":
    # Process all 30 split files with chunked reading
    aggregated_data = process_multiple_files_chunked(
        file_pattern='customer_split',
        num_files=30,
        output_file='train_data_aggregated_simple.csv',
        chunk_size=50000  # Adjust based on your RAM: 50000 for 16GB, 25000 for 8GB
    )