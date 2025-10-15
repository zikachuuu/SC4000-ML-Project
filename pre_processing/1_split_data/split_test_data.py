import pandas as pd
import numpy as np
from collections import defaultdict
import os
import gc

def split_customer_data_memory_efficient(input_file, output_prefix='split', num_splits=5, chunk_size=50000):
    """
    Split a large CSV file by customer ID into multiple files of approximately equal size.
    All rows for a single customer will be in the same output file.
    Memory-efficient version that processes data in smaller chunks.
    
    Parameters:
    - input_file: Path to the input CSV file
    - output_prefix: Prefix for output files (default: 'split')
    - num_splits: Number of output files to create (default: 5)
    - chunk_size: Number of rows to process at a time (default: 50000)
    """
    
    print("Starting memory-efficient data split process...")
    print(f"Chunk size: {chunk_size:,} rows")
    
    # Customer ID column name
    customer_col = 'customer_ID'
    
    # Step 1: Read the data in chunks and count rows per customer
    print("\nStep 1: Analyzing customer distribution...")
    customer_row_counts = defaultdict(int)
    total_rows = 0
    chunk_num = 0
    
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunk_num += 1
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # Count customers in this chunk
            counts = chunk[customer_col].value_counts()
            for customer, count in counts.items():
                customer_row_counts[customer] += count
            
            # Progress update
            print(f"  Processed chunk {chunk_num} ({total_rows:,} rows so far)...", end='\r')
            
            # Force garbage collection to free memory
            del chunk
            gc.collect()
    
    except Exception as e:
        print(f"\nError during Step 1: {e}")
        raise
    
    print(f"\n  Found {len(customer_row_counts):,} unique customers")
    print(f"  Total rows: {total_rows:,}")
    
    # Step 2: Assign customers to splits using a greedy bin packing approach
    print("\nStep 2: Distributing customers across splits...")
    
    # Sort customers by row count (descending) for better distribution
    sorted_customers = sorted(customer_row_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize splits with their current row counts
    split_assignments = {i: set() for i in range(num_splits)}  # Use set for faster lookup
    split_sizes = {i: 0 for i in range(num_splits)}
    
    # Greedy assignment: assign each customer to the split with fewest rows
    for customer, row_count in sorted_customers:
        # Find the split with minimum size
        min_split = min(split_sizes.items(), key=lambda x: x[1])[0]
        split_assignments[min_split].add(customer)
        split_sizes[min_split] += row_count
    
    # Print distribution
    print("\nDistribution across splits:")
    for split_idx in range(num_splits):
        print(f"  Split {split_idx}: {split_sizes[split_idx]:,} rows, {len(split_assignments[split_idx]):,} customers")
    
    # Free memory from customer_row_counts
    del customer_row_counts
    del sorted_customers
    gc.collect()
    
    # Step 3: Create customer to split mapping (more memory efficient than dict)
    print("\nStep 3: Creating customer-to-split mapping...")
    customer_to_split = {}
    for split_idx, customers in split_assignments.items():
        for customer in customers:
            customer_to_split[customer] = split_idx
    
    # Free split_assignments memory
    del split_assignments
    gc.collect()
    
    print(f"  Mapping created for {len(customer_to_split):,} customers")
    
    # Step 4: Read data again and write to appropriate split files
    print("\nStep 4: Writing split files...")
    
    # Open all output files
    output_files = {}
    headers_written = {}
    
    for split_idx in range(num_splits):
        output_file = f"{output_prefix}_{split_idx}.csv"
        output_files[split_idx] = open(output_file, 'w', newline='', encoding='utf-8')
        headers_written[split_idx] = False
    
    # Read and distribute data in chunks
    chunk_num = 0
    rows_written = {i: 0 for i in range(num_splits)}
    
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunk_num += 1
            print(f"  Processing chunk {chunk_num}...", end='\r')
            
            # Map customers to their split assignments
            chunk['_split_idx'] = chunk[customer_col].map(customer_to_split)
            
            # Write each split's portion
            for split_idx in range(num_splits):
                # Filter rows for this split
                mask = chunk['_split_idx'] == split_idx
                split_data = chunk[mask].drop('_split_idx', axis=1)
                
                if len(split_data) > 0:
                    # Write header only once
                    write_header = not headers_written[split_idx]
                    split_data.to_csv(
                        output_files[split_idx], 
                        mode='a', 
                        header=write_header, 
                        index=False
                    )
                    headers_written[split_idx] = True
                    rows_written[split_idx] += len(split_data)
                
                # Free memory
                del split_data
            
            # Free chunk memory
            del chunk
            gc.collect()
    
    except Exception as e:
        print(f"\nError during Step 4: {e}")
        # Close files before raising
        for f in output_files.values():
            f.close()
        raise
    
    # Close all files
    for f in output_files.values():
        f.close()
    
    print(f"\n\nCompleted! Created {num_splits} files:")
    for split_idx in range(num_splits):
        output_file = f"{output_prefix}_{split_idx}.csv"
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024**3)  # Size in GB
            print(f"  {output_file}: {file_size:.2f} GB ({rows_written[split_idx]:,} rows)")
        else:
            print(f"  {output_file}: NOT CREATED")
    
    # Verify totals
    total_written = sum(rows_written.values())
    print(f"\nTotal rows written: {total_written:,}")
    print(f"Expected total rows: {total_rows:,}")
    if total_written == total_rows:
        print("✓ All rows accounted for!")
    else:
        print(f"⚠ Warning: Mismatch of {abs(total_written - total_rows):,} rows")


if __name__ == "__main__":
    # Run the split with reduced chunk size for 36GB dataset
    split_customer_data_memory_efficient(
        input_file='test_data.csv',
        output_prefix='customer_split',
        num_splits=30,
        chunk_size=50000  # Reduced from 100k to handle larger dataset
    )