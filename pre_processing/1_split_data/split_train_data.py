import pandas as pd
import numpy as np
from collections import defaultdict
import os

def split_customer_data(input_file, output_prefix='split', num_splits=5):
    """
    Split a large CSV file by customer ID into multiple files of approximately equal size.
    All rows for a single customer will be in the same output file.
    
    Parameters:
    - input_file: Path to the input CSV file
    - output_prefix: Prefix for output files (default: 'split')
    - num_splits: Number of output files to create (default: 5)
    """
    
    print("Starting data split process...")
    
    # Customer ID column name
    customer_col = 'customer_ID'
    
    # Step 1: Read the data in chunks and count rows per customer
    print("Step 1: Analyzing customer distribution...")
    chunk_size = 100000  # Adjust based on your RAM
    customer_row_counts = defaultdict(int)
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        counts = chunk[customer_col].value_counts()
        for customer, count in counts.items():
            customer_row_counts[customer] += count
    
    print(f"Found {len(customer_row_counts)} unique customers")
    print(f"Total rows: {sum(customer_row_counts.values())}")
    
    # Step 2: Assign customers to splits using a greedy bin packing approach
    print("\nStep 2: Distributing customers across splits...")
    
    # Sort customers by row count (descending) for better distribution
    sorted_customers = sorted(customer_row_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize splits with their current row counts
    split_assignments = {i: [] for i in range(num_splits)}
    split_sizes = {i: 0 for i in range(num_splits)}
    
    # Greedy assignment: assign each customer to the split with fewest rows
    for customer, row_count in sorted_customers:
        # Find the split with minimum size
        min_split = min(split_sizes.items(), key=lambda x: x[1])[0]
        split_assignments[min_split].append(customer)
        split_sizes[min_split] += row_count
    
    # Print distribution
    print("\nDistribution across splits:")
    for split_idx in range(num_splits):
        print(f"  Split {split_idx}: {split_sizes[split_idx]:,} rows, {len(split_assignments[split_idx])} customers")
    
    # Step 3: Read data again and write to appropriate split files
    print("\nStep 3: Writing split files...")
    
    # Create a mapping of customer to split index
    customer_to_split = {}
    for split_idx, customers in split_assignments.items():
        for customer in customers:
            customer_to_split[customer] = split_idx
    
    # Open all output files
    output_files = {}
    writers = {}
    headers_written = {}
    
    for split_idx in range(num_splits):
        output_file = f"{output_prefix}_{split_idx}.csv"
        output_files[split_idx] = open(output_file, 'w', newline='', encoding='utf-8')
        headers_written[split_idx] = False
    
    # Read and distribute data
    chunk_num = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_num += 1
        print(f"  Processing chunk {chunk_num}...", end='\r')
        
        # Group chunk by split assignment
        chunk['_split_idx'] = chunk[customer_col].map(customer_to_split)
        
        for split_idx in range(num_splits):
            split_data = chunk[chunk['_split_idx'] == split_idx].drop('_split_idx', axis=1)
            
            if len(split_data) > 0:
                # Write header only once
                write_header = not headers_written[split_idx]
                split_data.to_csv(output_files[split_idx], mode='a', header=write_header, index=False)
                headers_written[split_idx] = True
    
    # Close all files
    for f in output_files.values():
        f.close()
    
    print(f"\n\nCompleted! Created {num_splits} files:")
    for split_idx in range(num_splits):
        output_file = f"{output_prefix}_{split_idx}.csv"
        file_size = os.path.getsize(output_file) / (1024**3)  # Size in GB
        print(f"  {output_file}: {file_size:.2f} GB ({split_sizes[split_idx]:,} rows)")

if __name__ == "__main__":
    # Run the split
    split_customer_data(
        input_file='train_data.csv',
        output_prefix='customer_split',
        num_splits=5
    )