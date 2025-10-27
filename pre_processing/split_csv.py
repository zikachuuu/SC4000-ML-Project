"""
================================================================================
CSV FILE SPLITTER
================================================================================

Split large CSV files into smaller chunks while preserving all columns.
Useful for handling memory constraints and distributed processing.

Features:
- Split by number of rows per file
- Split by number of total files
- Preserve headers in all output files
- Memory-efficient chunk processing
- Progress tracking
- Automatic file naming

================================================================================
"""

import pandas as pd
import os
import math
from typing import Optional, List
import time


class CSVSplitter:
    """
    Split large CSV files into multiple smaller files.
    """
    
    def __init__(self, input_file: str):
        """
        Initialize CSV splitter.
        
        Args:
            input_file: Path to the large CSV file to split
        """
        self.input_file = input_file
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")
        
        # Get file info
        self.file_size = os.path.getsize(input_file) / (1024**2)  # MB
        print(f"\n{'='*60}")
        print(f"CSV Splitter Initialized")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"File size: {self.file_size:.2f} MB")
    
    # ========================================================================
    # METHOD 1: Split by rows per file
    # ========================================================================
    
    def split_by_rows(self, rows_per_file: int, 
                     output_prefix: str = None,
                     output_dir: str = None,
                     include_header: bool = True,
                     chunksize: int = 10000) -> List[str]:
        """
        Split CSV by specifying number of rows per output file.
        
        Args:
            rows_per_file: Number of rows in each output file
            output_prefix: Prefix for output files (default: input filename)
            output_dir: Directory for output files (default: same as input)
            include_header: Include header row in each file
            chunksize: Number of rows to read at once (for memory efficiency)
            
        Returns:
            List of created file paths
        """
        print(f"\n{'='*60}")
        print("SPLIT BY ROWS PER FILE")
        print(f"{'='*60}")
        print(f"Target rows per file: {rows_per_file:,}")
        
        # Setup output naming
        if output_prefix is None:
            output_prefix = os.path.splitext(os.path.basename(self.input_file))[0]
        
        if output_dir is None:
            output_dir = os.path.dirname(self.input_file) or '.'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Count total rows first
        print("\nCounting total rows...")
        total_rows = sum(1 for _ in open(self.input_file)) - 1  # Exclude header
        print(f"Total data rows: {total_rows:,}")
        
        # Calculate number of files needed
        num_files = math.ceil(total_rows / rows_per_file)
        print(f"Number of output files: {num_files}")
        
        # Process in chunks
        created_files = []
        file_index = 1
        rows_in_current_file = 0
        current_chunks = []
        
        print(f"\n{'='*60}")
        print("Processing...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for i, chunk in enumerate(pd.read_csv(self.input_file, chunksize=chunksize)):
            
            # Add chunk to current file
            current_chunks.append(chunk)
            rows_in_current_file += len(chunk)
            
            # Check if we've reached the target rows per file
            if rows_in_current_file >= rows_per_file:
                # Save current file
                output_file = os.path.join(
                    output_dir, 
                    f"{output_prefix}_{file_index}.csv"
                )
                
                # Combine chunks and save
                df_to_save = pd.concat(current_chunks, ignore_index=True)
                df_to_save.to_csv(output_file, index=False, header=include_header)
                
                created_files.append(output_file)
                
                print(f"✓ Created: {output_file} ({len(df_to_save):,} rows)")
                
                # Reset for next file
                file_index += 1
                rows_in_current_file = 0
                current_chunks = []
        
        # Save remaining rows (last file)
        if current_chunks:
            output_file = os.path.join(
                output_dir, 
                f"{output_prefix}_{file_index}.csv"
            )
            
            df_to_save = pd.concat(current_chunks, ignore_index=True)
            df_to_save.to_csv(output_file, index=False, header=include_header)
            
            created_files.append(output_file)
            print(f"✓ Created: {output_file} ({len(df_to_save):,} rows)")
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*60}")
        print("SPLIT COMPLETED!")
        print(f"{'='*60}")
        print(f"Total files created: {len(created_files)}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Output directory: {output_dir}")
        
        return created_files
    
    # ========================================================================
    # METHOD 2: Split into N files
    # ========================================================================
    
    def split_into_n_files(self, num_files: int,
                          output_prefix: str = None,
                          output_dir: str = None,
                          include_header: bool = True,
                          chunksize: int = 10000) -> List[str]:
        """
        Split CSV into exactly N files with approximately equal rows.
        
        Args:
            num_files: Number of output files to create
            output_prefix: Prefix for output files (default: input filename)
            output_dir: Directory for output files (default: same as input)
            include_header: Include header row in each file
            chunksize: Number of rows to read at once (for memory efficiency)
            
        Returns:
            List of created file paths
        """
        print(f"\n{'='*60}")
        print("SPLIT INTO N FILES")
        print(f"{'='*60}")
        print(f"Target number of files: {num_files}")
        
        # Count total rows first
        print("\nCounting total rows...")
        total_rows = sum(1 for _ in open(self.input_file)) - 1  # Exclude header
        print(f"Total data rows: {total_rows:,}")
        
        # Calculate rows per file
        rows_per_file = math.ceil(total_rows / num_files)
        print(f"Rows per file: ~{rows_per_file:,}")
        
        # Use split_by_rows method
        return self.split_by_rows(
            rows_per_file=rows_per_file,
            output_prefix=output_prefix,
            output_dir=output_dir,
            include_header=include_header,
            chunksize=chunksize
        )
    
    # ========================================================================
    # METHOD 3: Split by file size (MB)
    # ========================================================================
    
    def split_by_size(self, target_size_mb: float,
                     output_prefix: str = None,
                     output_dir: str = None,
                     include_header: bool = True,
                     chunksize: int = 10000) -> List[str]:
        """
        Split CSV into files of approximately target size in MB.
        
        Args:
            target_size_mb: Target size for each output file in MB
            output_prefix: Prefix for output files (default: input filename)
            output_dir: Directory for output files (default: same as input)
            include_header: Include header row in each file
            chunksize: Number of rows to read at once (for memory efficiency)
            
        Returns:
            List of created file paths
        """
        print(f"\n{'='*60}")
        print("SPLIT BY FILE SIZE")
        print(f"{'='*60}")
        print(f"Target size per file: {target_size_mb:.2f} MB")
        
        # Estimate rows per file based on average row size
        print("\nAnalyzing file structure...")
        
        # Read first chunk to estimate row size
        first_chunk = pd.read_csv(self.input_file, nrows=1000)
        
        # Estimate bytes per row
        bytes_per_row = self.file_size * (1024**2) / sum(1 for _ in open(self.input_file))
        
        # Calculate rows per file
        target_bytes = target_size_mb * (1024**2)
        rows_per_file = int(target_bytes / bytes_per_row)
        
        print(f"Estimated bytes per row: {bytes_per_row:.2f}")
        print(f"Estimated rows per file: ~{rows_per_file:,}")
        
        # Use split_by_rows method
        return self.split_by_rows(
            rows_per_file=rows_per_file,
            output_prefix=output_prefix,
            output_dir=output_dir,
            include_header=include_header,
            chunksize=chunksize
        )
    
    # ========================================================================
    # UTILITY: Get file info
    # ========================================================================
    
    def get_info(self) -> dict:
        """
        Get information about the CSV file.
        
        Returns:
            Dictionary with file information
        """
        print(f"\n{'='*60}")
        print("FILE ANALYSIS")
        print(f"{'='*60}")
        
        # Count rows
        print("Counting rows...")
        total_rows = sum(1 for _ in open(self.input_file)) - 1
        
        # Get columns
        print("Reading columns...")
        df_sample = pd.read_csv(self.input_file, nrows=5)
        columns = df_sample.columns.tolist()
        
        # Calculate average row size
        bytes_per_row = self.file_size * (1024**2) / (total_rows + 1)
        
        info = {
            'filename': os.path.basename(self.input_file),
            'file_size_mb': self.file_size,
            'total_rows': total_rows,
            'num_columns': len(columns),
            'columns': columns,
            'avg_bytes_per_row': bytes_per_row
        }
        
        print(f"\nFile: {info['filename']}")
        print(f"Size: {info['file_size_mb']:.2f} MB")
        print(f"Rows: {info['total_rows']:,}")
        print(f"Columns: {info['num_columns']}")
        print(f"Average bytes/row: {info['avg_bytes_per_row']:.2f}")
        print(f"\nColumns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
        
        return info


# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

def split_csv_by_rows(input_file: str, rows_per_file: int, 
                     output_prefix: str = None, output_dir: str = None) -> List[str]:
    """
    Quick function to split CSV by rows.
    
    Args:
        input_file: Path to input CSV file
        rows_per_file: Number of rows per output file
        output_prefix: Prefix for output files
        output_dir: Directory for output files
        
    Returns:
        List of created file paths
    """
    splitter = CSVSplitter(input_file)
    return splitter.split_by_rows(rows_per_file, output_prefix, output_dir)


def split_csv_into_n_files(input_file: str, num_files: int,
                           output_prefix: str = None, output_dir: str = None) -> List[str]:
    """
    Quick function to split CSV into N files.
    
    Args:
        input_file: Path to input CSV file
        num_files: Number of output files
        output_prefix: Prefix for output files
        output_dir: Directory for output files
        
    Returns:
        List of created file paths
    """
    splitter = CSVSplitter(input_file)
    return splitter.split_into_n_files(num_files, output_prefix, output_dir)


def split_csv_by_size(input_file: str, target_size_mb: float,
                     output_prefix: str = None, output_dir: str = None) -> List[str]:
    """
    Quick function to split CSV by file size.
    
    Args:
        input_file: Path to input CSV file
        target_size_mb: Target size per file in MB
        output_prefix: Prefix for output files
        output_dir: Directory for output files
        
    Returns:
        List of created file paths
    """
    splitter = CSVSplitter(input_file)
    return splitter.split_by_size(target_size_mb, output_prefix, output_dir)


# ================================================================================
# USAGE EXAMPLES
# ================================================================================

def example_usage():
    """
    Demonstrate all splitting methods.
    """
    
    print("="*80)
    print("CSV SPLITTER - USAGE EXAMPLES")
    print("="*80)
    
    # ========================================================================
    # Example 1: Split by rows per file
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Split by Rows Per File")
    print("="*60)
    
    # Create splitter
    splitter = CSVSplitter('large_file.csv')
    
    # Split into files with 100,000 rows each
    files = splitter.split_by_rows(
        rows_per_file=100000,
        output_prefix='split_data',
        output_dir='./split_files'
    )
    
    print(f"\nCreated files: {files}")
    
    # ========================================================================
    # Example 2: Split into exactly N files
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Split Into N Files")
    print("="*60)
    
    # Split into exactly 5 files
    files = splitter.split_into_n_files(
        num_files=5,
        output_prefix='part',
        output_dir='./parts'
    )
    
    print(f"\nCreated files: {files}")
    
    # ========================================================================
    # Example 3: Split by file size
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Split by File Size")
    print("="*60)
    
    # Split into files of ~50MB each
    files = splitter.split_by_size(
        target_size_mb=50,
        output_prefix='chunk',
        output_dir='./chunks'
    )
    
    print(f"\nCreated files: {files}")
    
    # ========================================================================
    # Example 4: Quick convenience functions
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Quick Convenience Functions")
    print("="*60)
    
    # Method 1: By rows
    files = split_csv_by_rows('large_file.csv', rows_per_file=50000)
    
    # Method 2: Into N files
    files = split_csv_into_n_files('large_file.csv', num_files=10)
    
    # Method 3: By size
    files = split_csv_by_size('large_file.csv', target_size_mb=100)
    
    # ========================================================================
    # Example 5: Get file info before splitting
    # ========================================================================
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Analyze File Before Splitting")
    print("="*60)
    
    info = splitter.get_info()
    
    # Decide how to split based on info
    if info['total_rows'] > 1000000:
        print("\nFile is large, splitting into 10 files...")
        files = splitter.split_into_n_files(10)
    else:
        print("\nFile is manageable, splitting into files of 100k rows...")
        files = splitter.split_by_rows(100000)


# ================================================================================
# SIMPLE COMMAND LINE USAGE
# ================================================================================

if __name__ == "__main__":
    
    # Simple example - modify these parameters
    INPUT_FILE = r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000-ML-Project\data\train_data_simple_summary_statistics\train_data_with_simple_summary_statistics.csv'
    
    # Choose ONE of these methods:
    
    # Method 1: Split by rows (e.g., 100,000 rows per file)
    # split_csv_by_rows(INPUT_FILE, rows_per_file=100000, output_dir='./output')
    
    # Method 2: Split into N files (e.g., 5 files)
    split_csv_into_n_files(INPUT_FILE, num_files=20, output_dir=r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000-ML-Project\data\train_data_simple_summary_statistics')
    
    # Method 3: Split by size (e.g., 50MB per file)
    # split_csv_by_size(INPUT_FILE, target_size_mb=50, output_dir='./output')