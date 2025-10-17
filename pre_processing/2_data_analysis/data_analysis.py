"""
================================================================================
CORRECTED DATA QUALITY ANALYSIS FOR SPLIT FILES
================================================================================

Purpose: Accurate analysis of temporal credit card data across split files
Author: Credit Risk Team
Date: 2025

ANALYSIS CHECKS:
1. Customer ID grouping/sorting (customer_changes == unique_customers)
2. Date column detection (check first value, not column name)
3. Monthly uniqueness (flag duplicates: same customer + same month)
4. Temporal window analysis (start/end month distributions + window length)
5. Missing gaps analysis (gaps per customer + temporal gap patterns)

Memory Management:
- Process one file at a time
- Aggregate statistics incrementally
- Free memory between files
- Single-pass processing where possible
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
import gc
import os
from collections import defaultdict

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class CorrectedDataQualityAnalyzer:
    """
    Corrected analyzer with accurate algorithms for each check.
    """
    
    def __init__(self, file_list: List[str]):
        """
        Initialize with list of split files.
        
        Args:
            file_list (List[str]): Paths to split CSV files
        """
        self.file_list = file_list
        self.num_files = len(file_list)
        
        # Aggregated results storage
        self.all_customer_ids = []
        self.total_rows = 0
        self.total_unique_customers = 0
        
        # For check 1: Grouping
        self.customer_changes_per_file = []
        self.unique_customers_per_file = []
        
        # For check 2: Date columns
        self.date_columns = None
        
        # For check 3: Monthly uniqueness
        self.duplicate_rows = []
        self.multiple_rows_same_month = []
        
        # For check 4: Temporal windows
        self.customer_windows = []  # {customer_ID, start_month, end_month, window_length}
        
        # For check 5: Row counts and gaps
        self.customer_row_counts = defaultdict(int)
        self.customer_gaps = []  # {customer_ID, total_gaps, gap_months}
        self.gap_months_count = defaultdict(int)  # Track which months have most gaps
        
        print("="*80)
        print("CORRECTED DATA QUALITY ANALYZER")
        print("="*80)
        print(f"\nFiles to process: {self.num_files}")
        for i, f in enumerate(file_list, 1):
            if os.path.exists(f):
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"  {i}. {f} ({size_mb:.0f} MB)")
            else:
                print(f"  {i}. {f} ❌ NOT FOUND")
    
    def clear_memory(self):
        """Force garbage collection."""
        gc.collect()
    
    # ========================================================================
    # CHECK 2: DETECT DATE COLUMNS (Check first value, not name)
    # ========================================================================
    
    def detect_date_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Detect date columns by checking if first value is a date.
        
        CORRECTED LOGIC:
        - Don't rely on column names
        - Check actual data values
        - Parse first non-null value of each column
        
        Args:
            data (pd.DataFrame): Data to check
            
        Returns:
            List[str]: Column names that contain dates
        """
        if self.date_columns is not None:
            return self.date_columns
        
        print("\n" + "="*80)
        print("CHECK 2: DATE COLUMN DETECTION (Checking actual values)")
        print("="*80)
        
        date_columns = []
        
        for col in data.columns:
            if col == 'customer_ID':
                continue
            
            # Get first non-null value
            first_value = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else None
            
            if first_value is None:
                continue
            
            # Try to parse as date
            try:
                parsed = pd.to_datetime(first_value, errors='coerce')
                if pd.notna(parsed):
                    date_columns.append(col)
                    print(f"  ✅ '{col}': Date column detected (first value: {first_value})")
            except:
                continue
        
        print(f"\nTotal date columns found: {len(date_columns)}")
        
        if len(date_columns) == 1:
            print(f"✅ PASS: Only one date column ({date_columns[0]})")
        elif len(date_columns) > 1:
            print(f"⚠️  WARNING: Multiple date columns found: {date_columns}")
        else:
            print(f"❌ ERROR: No date columns found!")
        
        self.date_columns = date_columns
        return date_columns
    
    # ========================================================================
    # PROCESS SINGLE FILE
    # ========================================================================
    
    def process_file(self, filepath: str, file_index: int) -> Dict:
        """
        Process a single file with all checks in one pass.
        
        Args:
            filepath (str): Path to CSV file
            file_index (int): Index of file (1-based)
            
        Returns:
            Dict: Results from this file
        """
        print("\n" + "="*80)
        print(f"PROCESSING FILE {file_index}/{self.num_files}: {filepath}")
        print("="*80)
        
        # Load data
        print(f"\nLoading data...")
        try:
            data = pd.read_csv(filepath)
            print(f"  Loaded: {len(data):,} rows × {len(data.columns)} columns")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            return {'error': str(e)}
        
        file_results = {}
        
        # ====================================================================
        # CHECK 1: CUSTOMER GROUPING
        # ====================================================================
        
        print(f"\n{'─'*80}")
        print("CHECK 1: Customer ID Grouping/Sorting")
        print(f"{'─'*80}")
        
        if 'customer_ID' not in data.columns:
            print("  ❌ ERROR: 'customer_ID' column not found")
            file_results['grouping_error'] = True
        else:
            customer_col = data['customer_ID']
            
            # Count customer changes
            customer_changes = (customer_col != customer_col.shift()).sum()
            unique_customers = customer_col.nunique()
            
            is_grouped = (customer_changes == unique_customers)
            
            print(f"  Unique customers: {unique_customers:,}")
            print(f"  Customer ID changes: {customer_changes:,}")
            print(f"  Expected if grouped: {unique_customers:,}")
            
            if is_grouped:
                print(f"  ✅ PASS: Data is properly grouped")
            else:
                print(f"  ❌ FAIL: Data is NOT grouped!")
                print(f"       Customer IDs appear in {customer_changes:,} separate blocks")
            
            # Store for aggregation
            self.customer_changes_per_file.append(customer_changes)
            self.unique_customers_per_file.append(unique_customers)
            self.total_rows += len(data)
            
            file_results['grouping'] = {
                'is_grouped': is_grouped,
                'customer_changes': customer_changes,
                'unique_customers': unique_customers,
                'total_rows': len(data)
            }
        
        # ====================================================================
        # CHECK 2: DATE COLUMNS (First file only)
        # ====================================================================
        
        if file_index == 1:
            self.detect_date_columns(data)
        
        # ====================================================================
        # CHECK 3, 4, 5: Process customer data
        # ====================================================================
        
        if self.date_columns and len(self.date_columns) > 0:
            date_col = self.date_columns[0]  # Use first date column
            
            print(f"\n{'─'*80}")
            print(f"CHECK 3, 4, 5: Processing customer temporal data (using '{date_col}')")
            print(f"{'─'*80}")
            
            # Parse dates
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
            # Add year-month column for easier analysis
            data['year_month'] = data[date_col].dt.to_period('M')
            
            # Process each customer
            duplicates_found = 0
            multiple_same_month_found = 0
            
            for customer_id, group in data.groupby('customer_ID'):
                
                # Count rows for this customer (across all files)
                self.customer_row_counts[customer_id] += len(group)
                
                # Get dates and year-months
                dates = group[[date_col, 'year_month']].dropna()
                
                if len(dates) == 0:
                    continue
                
                # ============================================================
                # CHECK 3: Monthly uniqueness
                # ============================================================
                
                # Check for exact duplicate dates (same customer + same date)
                exact_duplicates = dates[dates[date_col].duplicated(keep=False)]
                if len(exact_duplicates) > 0:
                    duplicates_found += len(exact_duplicates) - len(exact_duplicates[date_col].unique())
                    
                    for dup_date in exact_duplicates[date_col].unique():
                        self.duplicate_rows.append({
                            'customer_ID': customer_id,
                            'duplicate_date': dup_date,
                            'count': (dates[date_col] == dup_date).sum()
                        })
                
                # Check for multiple rows in same month (flag, don't remove)
                month_counts = dates['year_month'].value_counts()
                multiple_months = month_counts[month_counts > 1]
                
                if len(multiple_months) > 0:
                    multiple_same_month_found += len(multiple_months)
                    
                    for month, count in multiple_months.items():
                        self.multiple_rows_same_month.append({
                            'customer_ID': customer_id,
                            'year_month': str(month),
                            'row_count': count
                        })
                
                # ============================================================
                # CHECK 4: Temporal windows
                # ============================================================
                
                months = dates['year_month'].unique()
                if len(months) > 0:
                    start_month = min(months)
                    end_month = max(months)
                    
                    # Calculate window length in months
                    window_length = (end_month - start_month).n + 1
                    
                    self.customer_windows.append({
                        'customer_ID': customer_id,
                        'start_month': str(start_month),
                        'end_month': str(end_month),
                        'window_length_months': window_length,
                        'actual_rows': len(months)
                    })
                
                # ============================================================
                # CHECK 5: Gap detection
                # ============================================================
                
                if len(months) > 1:
                    # Generate expected months
                    start = min(months)
                    end = max(months)
                    expected_months = pd.period_range(start=start, end=end, freq='M')
                    
                    # Find missing months
                    missing_months = set(expected_months) - set(months)
                    
                    if len(missing_months) > 0:
                        self.customer_gaps.append({
                            'customer_ID': customer_id,
                            'total_gaps': len(missing_months),
                            'expected_months': len(expected_months),
                            'actual_months': len(months)
                        })
                        
                        # Track which months/years have gaps
                        for gap_month in missing_months:
                            month_key = f"{gap_month.year}-{gap_month.month:02d}"
                            self.gap_months_count[month_key] += 1
            
            print(f"\n  Processing complete:")
            print(f"    Exact duplicate date rows: {duplicates_found}")
            print(f"    Customers with multiple rows in same month: {multiple_same_month_found}")
            print(f"    Customer windows tracked: {len(self.customer_windows)}")
        
        # Clear memory
        del data
        self.clear_memory()
        
        print(f"\n✅ File {file_index} processing complete")
        
        return file_results
    
    # ========================================================================
    # GENERATE REPORTS AND VISUALIZATIONS
    # ========================================================================
    
    def generate_results(self):
        """
        Generate comprehensive results and visualizations.
        """
        print("\n" + "="*80)
        print("GENERATING FINAL RESULTS")
        print("="*80)
        
        # ====================================================================
        # RESULT 1: Customer Grouping Summary
        # ====================================================================
        
        print("\n" + "="*80)
        print("RESULT 1: CUSTOMER ID GROUPING")
        print("="*80)
        
        total_customer_changes = sum(self.customer_changes_per_file)
        total_unique_customers = len(self.customer_row_counts)
        
        print(f"\nAcross all {self.num_files} files:")
        print(f"  Total unique customers: {total_unique_customers:,}")
        print(f"  Total rows: {self.total_rows:,}")
        print(f"  Total customer ID changes: {total_customer_changes:,}")
        
        all_grouped = all(
            changes == unique 
            for changes, unique in zip(self.customer_changes_per_file, 
                                      self.unique_customers_per_file)
        )
        
        if all_grouped:
            print(f"\n✅ PASS: All files are properly grouped")
            print(f"   (Each file: customer_changes == unique_customers)")
        else:
            print(f"\n❌ FAIL: Some files are NOT properly grouped")
            for i, (changes, unique) in enumerate(zip(self.customer_changes_per_file,
                                                     self.unique_customers_per_file), 1):
                status = "✅" if changes == unique else "❌"
                print(f"   File {i}: {status} (changes={changes:,}, unique={unique:,})")
        
        # ====================================================================
        # RESULT 2: Date Columns
        # ====================================================================
        
        print("\n" + "="*80)
        print("RESULT 2: DATE COLUMNS")
        print("="*80)
        
        if self.date_columns:
            print(f"\nDate columns detected: {len(self.date_columns)}")
            for col in self.date_columns:
                print(f"  - {col}")
            
            if len(self.date_columns) == 1:
                print(f"\n✅ PASS: Only one date column found")
            else:
                print(f"\n⚠️  WARNING: Multiple date columns found")
        
        # ====================================================================
        # RESULT 3: Monthly Uniqueness
        # ====================================================================
        
        print("\n" + "="*80)
        print("RESULT 3: MONTHLY UNIQUENESS CHECK")
        print("="*80)
        
        # Exact duplicates (same customer + same date)
        if len(self.duplicate_rows) > 0:
            dup_df = pd.DataFrame(self.duplicate_rows)
            total_dup_rows = dup_df['count'].sum() - len(dup_df)
            
            print(f"\n⚠️  EXACT DUPLICATES FOUND:")
            print(f"   Total duplicate row occurrences: {total_dup_rows:,}")
            print(f"   Unique customer-date pairs affected: {len(dup_df):,}")
            print(f"\n   Top 10 examples:")
            print(dup_df.head(10).to_string(index=False))
            
            dup_df.to_csv('exact_duplicate_rows.csv', index=False)
            print(f"\n   ❌ ACTION REQUIRED: Remove these duplicate rows!")
            print(f"   Full list saved to: exact_duplicate_rows.csv")
        else:
            print(f"\n✅ PASS: No exact duplicate rows found")
        
        # Multiple rows same month (flagged, not removed)
        if len(self.multiple_rows_same_month) > 0:
            multi_df = pd.DataFrame(self.multiple_rows_same_month)
            
            print(f"\nℹ️  MULTIPLE ROWS IN SAME MONTH:")
            print(f"   Customer-month pairs with multiple rows: {len(multi_df):,}")
            print(f"   (Different dates within same month - this is OK)")
            print(f"\n   Top 10 examples:")
            print(multi_df.head(10).to_string(index=False))
            
            multi_df.to_csv('multiple_rows_same_month.csv', index=False)
            print(f"\n   ℹ️  INFO: Flagged for awareness (not an error)")
            print(f"   Full list saved to: multiple_rows_same_month.csv")
        else:
            print(f"\nℹ️  INFO: Each customer has at most one row per month")
        
        # ====================================================================
        # RESULT 4: Temporal Windows
        # ====================================================================
        
        print("\n" + "="*80)
        print("RESULT 4: TEMPORAL WINDOW ANALYSIS")
        print("="*80)
        
        if len(self.customer_windows) > 0:
            windows_df = pd.DataFrame(self.customer_windows)
            
            print(f"\nCustomers analyzed: {len(windows_df):,}")
            
            # Start month distribution
            start_months = windows_df['start_month'].value_counts().sort_index()
            print(f"\nStart month distribution:")
            print(f"  Earliest start: {start_months.index[0]}")
            print(f"  Latest start: {start_months.index[-1]}")
            print(f"  Most common: {start_months.idxmax()} ({start_months.max():,} customers)")
            
            # End month distribution
            end_months = windows_df['end_month'].value_counts().sort_index()
            print(f"\nEnd month distribution:")
            print(f"  Earliest end: {end_months.index[0]}")
            print(f"  Latest end: {end_months.index[-1]}")
            print(f"  Most common: {end_months.idxmax()} ({end_months.max():,} customers)")
            
            # Window length distribution
            print(f"\nWindow length (months):")
            print(f"  Min: {windows_df['window_length_months'].min()}")
            print(f"  Median: {windows_df['window_length_months'].median():.0f}")
            print(f"  Mean: {windows_df['window_length_months'].mean():.1f}")
            print(f"  Max: {windows_df['window_length_months'].max()}")
            
            windows_df.to_csv('customer_temporal_windows.csv', index=False)
            print(f"\nFull data saved to: customer_temporal_windows.csv")
        
        # ====================================================================
        # RESULT 5: Row Counts and Gap Analysis
        # ====================================================================
        
        print("\n" + "="*80)
        print("RESULT 5: ROW COUNTS AND GAP ANALYSIS")
        print("="*80)
        
        # Row counts per customer
        row_counts = pd.Series(self.customer_row_counts)
        
        print(f"\nRows per customer:")
        print(f"  Total customers: {len(row_counts):,}")
        print(f"  Min rows: {row_counts.min()}")
        print(f"  Median rows: {row_counts.median():.0f}")
        print(f"  Mean rows: {row_counts.mean():.1f}")
        print(f"  Max rows: {row_counts.max()}")
        
        # Gap analysis
        if len(self.customer_gaps) > 0:
            gaps_df = pd.DataFrame(self.customer_gaps)
            
            print(f"\nGap analysis:")
            print(f"  Customers with gaps: {len(gaps_df):,} "
                  f"({len(gaps_df)/len(row_counts)*100:.1f}%)")
            print(f"  Mean gaps per customer: {gaps_df['total_gaps'].mean():.1f}")
            print(f"  Median gaps: {gaps_df['total_gaps'].median():.0f}")
            print(f"  Max gaps: {gaps_df['total_gaps'].max()}")
            
            # Calculate completeness
            gaps_df['completeness_pct'] = (
                gaps_df['actual_months'] / gaps_df['expected_months'] * 100
            )
            
            print(f"\nData completeness:")
            print(f"  Mean completeness: {gaps_df['completeness_pct'].mean():.1f}%")
            print(f"  Median completeness: {gaps_df['completeness_pct'].median():.1f}%")
            
            incomplete = gaps_df[gaps_df['completeness_pct'] < 80]
            print(f"  Customers <80% complete: {len(incomplete):,} "
                  f"({len(incomplete)/len(gaps_df)*100:.1f}% of those with gaps)")
            
            gaps_df.to_csv('customer_gaps_analysis.csv', index=False)
            print(f"\nGaps data saved to: customer_gaps_analysis.csv")
            
            # Temporal gap patterns
            if len(self.gap_months_count) > 0:
                print(f"\nTemporal gap patterns:")
                gap_months_series = pd.Series(self.gap_months_count).sort_values(ascending=False)
                
                print(f"  Top 10 months with most gaps:")
                for month, count in gap_months_series.head(10).items():
                    print(f"    {month}: {count:,} customers missing data")
                
                # Save
                gap_months_df = pd.DataFrame({
                    'year_month': gap_months_series.index,
                    'customers_with_gap': gap_months_series.values
                })
                gap_months_df.to_csv('gap_temporal_patterns.csv', index=False)
                print(f"\n  Full temporal gap patterns saved to: gap_temporal_patterns.csv")
        else:
            print(f"\n✅ EXCELLENT: No customers have missing months!")
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        
        self.create_visualizations(windows_df if len(self.customer_windows) > 0 else None,
                                   gaps_df if len(self.customer_gaps) > 0 else None,
                                   row_counts)
    
    def create_visualizations(self, windows_df, gaps_df, row_counts):
        """
        Create comprehensive visualizations.
        
        Args:
            windows_df: DataFrame with window information
            gaps_df: DataFrame with gap information  
            row_counts: Series with row counts per customer
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Determine number of plots
        n_plots = 5 if gaps_df is not None else 4
        
        fig = plt.figure(figsize=(20, 12))
        
        # ====================================================================
        # PLOT 1: Start Month Distribution
        # ====================================================================
        
        if windows_df is not None:
            ax1 = plt.subplot(2, 3, 1)
            
            start_months = windows_df['start_month'].value_counts().sort_index()
            
            ax1.bar(range(len(start_months)), start_months.values, 
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Time', fontsize=11)
            ax1.set_ylabel('Number of Customers', fontsize=11)
            ax1.set_title('Distribution of Start Months', fontsize=12, fontweight='bold')
            
            # Show every nth label to avoid crowding
            step = max(1, len(start_months) // 12)
            ax1.set_xticks(range(0, len(start_months), step))
            ax1.set_xticklabels(start_months.index[::step], rotation=45, ha='right', fontsize=8)
            ax1.grid(axis='y', alpha=0.3)
        
        # ====================================================================
        # PLOT 2: End Month Distribution
        # ====================================================================
        
        if windows_df is not None:
            ax2 = plt.subplot(2, 3, 2)
            
            end_months = windows_df['end_month'].value_counts().sort_index()
            
            ax2.bar(range(len(end_months)), end_months.values,
                   color='coral', edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('Number of Customers', fontsize=11)
            ax2.set_title('Distribution of End Months', fontsize=12, fontweight='bold')
            
            step = max(1, len(end_months) // 12)
            ax2.set_xticks(range(0, len(end_months), step))
            ax2.set_xticklabels(end_months.index[::step], rotation=45, ha='right', fontsize=8)
            ax2.grid(axis='y', alpha=0.3)
        
        # ====================================================================
        # PLOT 3: Window Length Distribution
        # ====================================================================
        
        if windows_df is not None:
            ax3 = plt.subplot(2, 3, 3)
            
            ax3.hist(windows_df['window_length_months'], bins=30, 
                    color='green', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Window Length (months)', fontsize=11)
            ax3.set_ylabel('Number of Customers', fontsize=11)
            ax3.set_title('Distribution of Window Lengths', fontsize=12, fontweight='bold')
            ax3.axvline(windows_df['window_length_months'].median(), 
                       color='red', linestyle='--', linewidth=2,
                       label=f"Median: {windows_df['window_length_months'].median():.0f} months")
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # ====================================================================
        # PLOT 4: Rows per Customer Distribution
        # ====================================================================
        
        ax4 = plt.subplot(2, 3, 4)
        
        ax4.hist(row_counts, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Number of Rows', fontsize=11)
        ax4.set_ylabel('Number of Customers', fontsize=11)
        ax4.set_title('Distribution of Rows per Customer', fontsize=12, fontweight='bold')
        ax4.axvline(row_counts.median(), color='red', linestyle='--', linewidth=2,
                   label=f"Median: {row_counts.median():.0f} rows")
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # ====================================================================
        # PLOT 5: Missing Gaps per Customer
        # ====================================================================
        
        if gaps_df is not None:
            ax5 = plt.subplot(2, 3, 5)
            
            ax5.hist(gaps_df['total_gaps'], bins=30, 
                    color='crimson', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Number of Missing Months', fontsize=11)
            ax5.set_ylabel('Number of Customers', fontsize=11)
            ax5.set_title('Distribution of Missing Gaps per Customer\n(Only customers with gaps)', 
                         fontsize=12, fontweight='bold')
            ax5.axvline(gaps_df['total_gaps'].median(), 
                       color='yellow', linestyle='--', linewidth=2,
                       label=f"Median: {gaps_df['total_gaps'].median():.0f} gaps")
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # ====================================================================
        # PLOT 6: Temporal Gap Pattern
        # ====================================================================
        
        if gaps_df is not None and len(self.gap_months_count) > 0:
            ax6 = plt.subplot(2, 3, 6)
            
            gap_months_series = pd.Series(self.gap_months_count).sort_index()
            
            ax6.plot(range(len(gap_months_series)), gap_months_series.values,
                    marker='o', color='orange', linewidth=2, markersize=4)
            ax6.set_xlabel('Time', fontsize=11)
            ax6.set_ylabel('Number of Customers with Gap', fontsize=11)
            ax6.set_title('Temporal Pattern: Which Months Have Most Gaps?', 
                         fontsize=12, fontweight='bold')
            
            # Show every nth label
            step = max(1, len(gap_months_series) // 12)
            ax6.set_xticks(range(0, len(gap_months_series), step))
            ax6.set_xticklabels(gap_months_series.index[::step], 
                              rotation=45, ha='right', fontsize=8)
            ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_quality_analysis_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: data_quality_analysis_results.png")
        plt.show()
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_analysis(self):
        """
        Run complete analysis pipeline.
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE DATA QUALITY ANALYSIS")
        print("="*80)
        
        # Process each file
        for i, filepath in enumerate(self.file_list, 1):
            if not os.path.exists(filepath):
                print(f"\n⚠️  Skipping {filepath} - file not found")
                continue
            
            self.process_file(filepath, i)
        
        # Generate results
        self.generate_results()
        
        # Summary report
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
    
    def generate_summary_report(self):
        """
        Generate text summary report.
        """
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("DATA QUALITY ANALYSIS - SUMMARY REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Files analyzed: {self.num_files}")
        
        # Check 1
        report.append("\n" + "-"*80)
        report.append("1. CUSTOMER ID GROUPING")
        report.append("-"*80)
        report.append(f"Total unique customers: {len(self.customer_row_counts):,}")
        report.append(f"Total rows: {self.total_rows:,}")
        
        all_grouped = all(
            changes == unique 
            for changes, unique in zip(self.customer_changes_per_file, 
                                    self.unique_customers_per_file)
        )
        report.append(f"All files grouped: {'YES' if all_grouped else 'NO'}")
        
        # Check 2
        report.append("\n" + "-"*80)
        report.append("2. DATE COLUMNS")
        report.append("-"*80)
        if self.date_columns:
            report.append(f"Date columns found: {', '.join(self.date_columns)}")
            report.append(f"Status: {'PASS' if len(self.date_columns) == 1 else 'MULTIPLE'}")
        
        # Check 3
        report.append("\n" + "-"*80)
        report.append("3. MONTHLY UNIQUENESS")
        report.append("-"*80)
        report.append(f"Exact duplicate rows: {len(self.duplicate_rows):,}")
        report.append(f"Multiple rows same month: {len(self.multiple_rows_same_month):,}")
        
        if len(self.duplicate_rows) > 0:
            report.append(f"ACTION: Remove {len(self.duplicate_rows):,} duplicate rows")
        else:
            report.append(f"PASS: No exact duplicates")
        
        # Check 4
        report.append("\n" + "-"*80)
        report.append("4. TEMPORAL WINDOWS")
        report.append("-"*80)
        if len(self.customer_windows) > 0:
            windows_df = pd.DataFrame(self.customer_windows)
            report.append(f"Mean window length: {windows_df['window_length_months'].mean():.1f} months")
            report.append(f"Median window length: {windows_df['window_length_months'].median():.0f} months")
        
        # Check 5
        report.append("\n" + "-"*80)
        report.append("5. GAPS AND COMPLETENESS")
        report.append("-"*80)
        
        row_counts = pd.Series(self.customer_row_counts)
        report.append(f"Mean rows per customer: {row_counts.mean():.1f}")
        report.append(f"Median rows per customer: {row_counts.median():.0f}")
        
        if len(self.customer_gaps) > 0:
            gaps_df = pd.DataFrame(self.customer_gaps)
            
            # Calculate completeness_pct HERE before using it
            gaps_df['completeness_pct'] = (
                gaps_df['actual_months'] / gaps_df['expected_months'] * 100
            )
            
            report.append(f"\nCustomers with gaps: {len(gaps_df):,} "
                        f"({len(gaps_df)/len(row_counts)*100:.1f}%)")
            report.append(f"Mean completeness: {gaps_df['completeness_pct'].mean():.1f}%")
            
            if gaps_df['completeness_pct'].mean() >= 80:
                report.append(f"GOOD: Most customers have complete data")
            else:
                report.append(f"WARNING: Significant missing data detected")
        else:
            report.append(f"EXCELLENT: No gaps detected")
        
        # Overall conclusion
        report.append("\n" + "="*80)
        report.append("OVERALL ASSESSMENT")
        report.append("="*80)
        
        issues = []
        if not all_grouped:
            issues.append("Data not properly grouped")
        if len(self.duplicate_rows) > 0:
            issues.append(f"{len(self.duplicate_rows):,} duplicate rows to remove")
        if len(self.customer_gaps) > 0:
            gaps_df = pd.DataFrame(self.customer_gaps)
            
            # Calculate completeness_pct again (or reuse from above)
            gaps_df['completeness_pct'] = (
                gaps_df['actual_months'] / gaps_df['expected_months'] * 100
            )
            
            if gaps_df['completeness_pct'].mean() < 80:
                issues.append("Significant missing data")
        
        if len(issues) == 0:
            report.append("\nDATA QUALITY: EXCELLENT")
            report.append("   All checks passed successfully")
        else:
            report.append("\nDATA QUALITY: ISSUES FOUND")
            for issue in issues:
                report.append(f"   {issue}")
        
        report.append("\n" + "="*80)
        report.append("OUTPUT FILES GENERATED")
        report.append("="*80)
        report.append("- data_quality_analysis_results.png (visualizations)")
        report.append("- customer_temporal_windows.csv (window analysis)")
        if len(self.duplicate_rows) > 0:
            report.append("- exact_duplicate_rows.csv (duplicates to remove)")
        if len(self.multiple_rows_same_month) > 0:
            report.append("- multiple_rows_same_month.csv (flagged for info)")
        if len(self.customer_gaps) > 0:
            report.append("- customer_gaps_analysis.csv (gap analysis)")
            report.append("- gap_temporal_patterns.csv (temporal gap patterns)")
        report.append("- data_quality_summary.txt (this report)")
        
        report_text = "\n".join(report)
        
        # Save
        with open('data_quality_summary.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nSummary saved to: data_quality_summary.txt")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main execution function.
    """
    
    # Define split files
    file_list = [
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000-ML-Project\data\customer_split_0.csv",
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000-ML-Project\data\customer_split_1.csv"
    ]
    
    # Alternative: Auto-detect
    # import glob
    # file_list = sorted(glob.glob('train_data_part*.csv'))
    
    # Initialize analyzer
    analyzer = CorrectedDataQualityAnalyzer(file_list)
    
    # Run analysis
    analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("\nReview generated files for detailed analysis.")


if __name__ == "__main__":
    main()