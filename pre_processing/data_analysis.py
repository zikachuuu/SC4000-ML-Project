"""
================================================================================
MEMORY-EFFICIENT DATA QUALITY ANALYSIS FOR SPLIT FILES
================================================================================

Purpose: Analyze large credit card dataset split into multiple files
Approach: Process each file separately, then combine results

Input: Multiple CSV files (e.g., train_data_part1.csv, train_data_part2.csv, ...)
Output: Combined analysis results across all files

Memory Management:
- Process one file at a time
- Aggregate statistics incrementally
- Clear memory between files
- Generate final combined report

Usage:
    Option 1: With Manually Specified Files
        # Edit the file list in main()
        file_list = [
            'train_data_part1.csv',
            'train_data_part2.csv',
            'train_data_part3.csv',
            'train_data_part4.csv',
            'train_data_part5.csv'
        ]

    Option 2: Auto-detect Files
        # Uncomment in main()
        import glob
        file_list = sorted(glob.glob('train_data_part*.csv'))
    
    Run: python memory_efficient_analysis.py

Author: Credit Risk Team
Date: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import gc
import os

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class MemoryEfficientDataQualityAnalyzer:
    """
    Memory-efficient analyzer for large split files.
    
    Processes files one at a time and aggregates results.
    """
    
    def __init__(self, file_list: List[str]):
        """
        Initialize analyzer with list of file paths.
        
        Args:
            file_list (List[str]): List of CSV file paths to analyze
        """
        print("="*80)
        print("MEMORY-EFFICIENT DATA QUALITY ANALYSIS")
        print("="*80)
        
        self.file_list = file_list
        self.num_files = len(file_list)
        
        print(f"\nFiles to process: {self.num_files}")
        for i, filepath in enumerate(file_list, 1):
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"  {i}. {filepath} ({size_mb:.1f} MB)")
            else:
                print(f"  {i}. {filepath} ❌ NOT FOUND")
        
        # Storage for aggregated results
        self.aggregated_results = {
            'customer_grouping': {},
            'date_columns': {},
            'monthly_intervals': {},
            'temporal_coverage': {},
            'missing_periods': {}
        }
        
        # Lists to store per-file results
        self.per_file_results = []
        
        # Combined data structures
        self.all_customers = set()
        self.all_intervals = []
        self.all_coverage = []
        self.all_gaps = []
        
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def clear_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        print("  [Memory cleared]")
    
    def print_memory_usage(self):
        """Print current memory usage (if psutil available)."""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024**2)
            print(f"  [Memory usage: {mem_mb:.1f} MB]")
        except ImportError:
            pass
    
    # ========================================================================
    # SINGLE FILE ANALYSIS
    # ========================================================================
    
    def analyze_single_file(self, filepath: str, file_index: int) -> Dict:
        """
        Analyze a single file and return results.
        
        Args:
            filepath (str): Path to CSV file
            file_index (int): Index of this file (1-based)
            
        Returns:
            Dict: Analysis results for this file
        """
        print("\n" + "="*80)
        print(f"ANALYZING FILE {file_index}/{self.num_files}: {filepath}")
        print("="*80)
        
        # Load data in chunks to save memory
        print(f"\nLoading data...")
        
        try:
            # Read file
            data = pd.read_csv(filepath)
            
            print(f"  Loaded: {len(data):,} rows, {len(data.columns)} columns")
            self.print_memory_usage()
            
        except Exception as e:
            print(f"  ❌ ERROR loading file: {e}")
            return {'error': str(e)}
        
        file_results = {}
        
        # ====================================================================
        # CHECK 1: Customer Grouping
        # ====================================================================
        
        print(f"\n[Check 1/{self.num_files}] Customer ID Grouping...")
        
        if 'customer_ID' not in data.columns:
            print("  ❌ ERROR: 'customer_ID' column not found!")
            file_results['customer_grouping'] = {'error': 'Missing customer_ID'}
        else:
            customer_col = data['customer_ID']
            
            unique_customers = customer_col.nunique()
            total_rows = len(data)
            
            # Check grouping
            customer_changes = (customer_col != customer_col.shift()).sum()
            is_grouped = customer_changes == unique_customers
            
            # Rows per customer
            rows_per_customer = customer_col.value_counts()
            
            file_results['customer_grouping'] = {
                'unique_customers': unique_customers,
                'total_rows': total_rows,
                'is_grouped': is_grouped,
                'customer_changes': customer_changes,
                'rows_per_customer_stats': {
                    'min': int(rows_per_customer.min()),
                    'median': float(rows_per_customer.median()),
                    'mean': float(rows_per_customer.mean()),
                    'max': int(rows_per_customer.max()),
                    'std': float(rows_per_customer.std())
                }
            }
            
            # Track all customers across files
            self.all_customers.update(customer_col.unique())
            
            print(f"  Customers in this file: {unique_customers:,}")
            print(f"  Rows in this file: {total_rows:,}")
            print(f"  Grouped: {'✅ YES' if is_grouped else '⚠️ NO'}")
        
        # ====================================================================
        # CHECK 2: Date Columns
        # ====================================================================
        
        print(f"\n[Check 2/{self.num_files}] Date Columns...")
        
        date_columns = []
        
        for col in data.columns:
            if col == 'customer_ID':
                continue
            
            try:
                sample = data[col].dropna().head(100)
                
                if any(keyword in col.lower() for keyword in ['date', 'time', 's_2']):
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() / len(sample) > 0.8:
                        date_columns.append(col)
            except:
                continue
        
        file_results['date_columns'] = {
            'columns_found': date_columns,
            'count': len(date_columns)
        }
        
        print(f"  Date columns found: {date_columns}")
        
        # ====================================================================
        # CHECK 3: Monthly Intervals
        # ====================================================================
        
        print(f"\n[Check 3/{self.num_files}] Monthly Intervals...")
        
        date_column = 'S_2'
        
        if date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            data_sorted = data.sort_values(['customer_ID', date_column])
            
            intervals = []
            
            # Sample customers to save memory
            sample_size = min(10000, data['customer_ID'].nunique())
            sampled_customers = data['customer_ID'].unique()[:sample_size]
            
            for customer_id in sampled_customers:
                customer_data = data_sorted[data_sorted['customer_ID'] == customer_id]
                dates = customer_data[date_column].dropna().sort_values()
                
                if len(dates) < 2:
                    continue
                
                for i in range(1, len(dates)):
                    interval_days = (dates.iloc[i] - dates.iloc[i-1]).days
                    intervals.append(interval_days)
            
            if len(intervals) > 0:
                intervals = np.array(intervals)
                
                monthly_min = 27
                monthly_max = 32
                monthly_count = ((intervals >= monthly_min) & (intervals <= monthly_max)).sum()
                monthly_pct = monthly_count / len(intervals) * 100
                
                file_results['monthly_intervals'] = {
                    'total_intervals': len(intervals),
                    'mean_days': float(intervals.mean()),
                    'median_days': float(np.median(intervals)),
                    'std_days': float(intervals.std()),
                    'monthly_percentage': float(monthly_pct)
                }
                
                # Store intervals for later aggregation
                self.all_intervals.extend(intervals.tolist())
                
                print(f"  Intervals analyzed: {len(intervals):,}")
                print(f"  Mean interval: {intervals.mean():.1f} days")
                print(f"  Monthly percentage: {monthly_pct:.1f}%")
            else:
                print(f"  ⚠️ No intervals calculated")
        
        # ====================================================================
        # CHECK 4: Temporal Coverage
        # ====================================================================
        
        print(f"\n[Check 4/{self.num_files}] Temporal Coverage...")
        
        if date_column in data.columns:
            # Sample for memory efficiency
            sample_size = min(10000, data['customer_ID'].nunique())
            sampled_customers = data['customer_ID'].unique()[:sample_size]
            
            coverage_list = []
            
            for customer_id in sampled_customers:
                customer_data = data[data['customer_ID'] == customer_id]
                dates = customer_data[date_column].dropna().sort_values()
                
                if len(dates) == 0:
                    continue
                
                start_date = dates.min()
                end_date = dates.max()
                span_days = (end_date - start_date).days
                span_months = span_days / 30.44
                num_statements = len(dates)
                
                expected_statements = int(span_months) + 1
                completeness = num_statements / expected_statements if expected_statements > 0 else 0
                
                coverage_list.append({
                    'customer_ID': customer_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'span_months': span_months,
                    'num_statements': num_statements,
                    'expected_statements': expected_statements,
                    'completeness_pct': completeness * 100
                })
            
            if len(coverage_list) > 0:
                coverage_df = pd.DataFrame(coverage_list)
                
                file_results['temporal_coverage'] = {
                    'mean_span_months': float(coverage_df['span_months'].mean()),
                    'mean_completeness': float(coverage_df['completeness_pct'].mean()),
                    'customers_analyzed': len(coverage_df)
                }
                
                # Store for aggregation
                self.all_coverage.extend(coverage_list)
                
                print(f"  Customers analyzed: {len(coverage_df):,}")
                print(f"  Mean span: {coverage_df['span_months'].mean():.1f} months")
                print(f"  Mean completeness: {coverage_df['completeness_pct'].mean():.1f}%")
        
        # ====================================================================
        # CHECK 5: Missing Periods
        # ====================================================================
        
        print(f"\n[Check 5/{self.num_files}] Missing Periods...")
        
        if date_column in data.columns:
            # Sample for memory
            sample_size = min(5000, data['customer_ID'].nunique())
            sampled_customers = data['customer_ID'].unique()[:sample_size]
            
            customers_with_gaps = 0
            
            for customer_id in sampled_customers:
                customer_data = data[data['customer_ID'] == customer_id]
                dates = customer_data[date_column].dropna().sort_values()
                
                if len(dates) < 2:
                    continue
                
                start_date = dates.min()
                end_date = dates.max()
                
                expected_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
                actual_months = pd.to_datetime(dates.dt.to_period('M').dt.to_timestamp())
                
                missing_months = set(expected_dates) - set(actual_months)
                
                if len(missing_months) > 0:
                    customers_with_gaps += 1
                    
                    self.all_gaps.append({
                        'customer_ID': customer_id,
                        'expected_statements': len(expected_dates),
                        'actual_statements': len(actual_months.unique()),
                        'missing_statements': len(missing_months)
                    })
            
            file_results['missing_periods'] = {
                'customers_analyzed': sample_size,
                'customers_with_gaps': customers_with_gaps,
                'gap_percentage': customers_with_gaps / sample_size * 100 if sample_size > 0 else 0
            }
            
            print(f"  Customers analyzed: {sample_size:,}")
            print(f"  Customers with gaps: {customers_with_gaps:,} ({customers_with_gaps/sample_size*100:.1f}%)")
        
        # Store file results
        self.per_file_results.append({
            'file': filepath,
            'index': file_index,
            'results': file_results
        })
        
        # Clear data from memory
        del data
        if 'data_sorted' in locals():
            del data_sorted
        
        self.clear_memory()
        
        print(f"\n✅ File {file_index} analysis complete")
        
        return file_results
    
    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    
    def aggregate_results(self) -> Dict:
        """
        Aggregate results from all files into combined statistics.
        
        Returns:
            Dict: Combined analysis results
        """
        print("\n" + "="*80)
        print("AGGREGATING RESULTS FROM ALL FILES")
        print("="*80)
        
        combined = {}
        
        # ====================================================================
        # AGGREGATE: Customer Grouping
        # ====================================================================
        
        print("\nAggregating customer grouping statistics...")
        
        total_customers = len(self.all_customers)
        total_rows = sum(r['results']['customer_grouping']['total_rows'] 
                        for r in self.per_file_results 
                        if 'customer_grouping' in r['results'])
        
        all_grouped = all(r['results']['customer_grouping'].get('is_grouped', False)
                         for r in self.per_file_results
                         if 'customer_grouping' in r['results'])
        
        combined['customer_grouping'] = {
            'total_customers_across_files': total_customers,
            'total_rows_across_files': total_rows,
            'avg_rows_per_customer': total_rows / total_customers if total_customers > 0 else 0,
            'all_files_grouped': all_grouped
        }
        
        print(f"  Total unique customers: {total_customers:,}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Average rows per customer: {total_rows/total_customers:.2f}")
        
        # ====================================================================
        # AGGREGATE: Date Columns
        # ====================================================================
        
        print("\nAggregating date column information...")
        
        all_date_columns = set()
        for r in self.per_file_results:
            if 'date_columns' in r['results']:
                all_date_columns.update(r['results']['date_columns']['columns_found'])
        
        combined['date_columns'] = {
            'columns_found': list(all_date_columns),
            'count': len(all_date_columns)
        }
        
        print(f"  Date columns found: {list(all_date_columns)}")
        
        # ====================================================================
        # AGGREGATE: Monthly Intervals
        # ====================================================================
        
        print("\nAggregating monthly interval statistics...")
        
        if len(self.all_intervals) > 0:
            intervals_array = np.array(self.all_intervals)
            
            monthly_min = 27
            monthly_max = 32
            monthly_count = ((intervals_array >= monthly_min) & (intervals_array <= monthly_max)).sum()
            monthly_pct = monthly_count / len(intervals_array) * 100
            
            combined['monthly_intervals'] = {
                'total_intervals': len(intervals_array),
                'mean_days': float(intervals_array.mean()),
                'median_days': float(np.median(intervals_array)),
                'std_days': float(intervals_array.std()),
                'monthly_percentage': float(monthly_pct)
            }
            
            print(f"  Total intervals: {len(intervals_array):,}")
            print(f"  Mean: {intervals_array.mean():.1f} days")
            print(f"  Monthly percentage: {monthly_pct:.1f}%")
        
        # ====================================================================
        # AGGREGATE: Temporal Coverage
        # ====================================================================
        
        print("\nAggregating temporal coverage...")
        
        if len(self.all_coverage) > 0:
            coverage_df = pd.DataFrame(self.all_coverage)
            
            combined['temporal_coverage'] = {
                'customers_analyzed': len(coverage_df),
                'mean_span_months': float(coverage_df['span_months'].mean()),
                'mean_completeness': float(coverage_df['completeness_pct'].mean()),
                'incomplete_customers': len(coverage_df[coverage_df['completeness_pct'] < 80])
            }
            
            print(f"  Customers analyzed: {len(coverage_df):,}")
            print(f"  Mean span: {coverage_df['span_months'].mean():.1f} months")
            print(f"  Mean completeness: {coverage_df['completeness_pct'].mean():.1f}%")
            
            # Save coverage report
            coverage_df.to_csv('combined_temporal_coverage_report.csv', index=False)
            print(f"  Saved: combined_temporal_coverage_report.csv")
        
        # ====================================================================
        # AGGREGATE: Missing Periods
        # ====================================================================
        
        print("\nAggregating missing period statistics...")
        
        if len(self.all_gaps) > 0:
            gaps_df = pd.DataFrame(self.all_gaps)
            
            total_analyzed = sum(r['results']['missing_periods']['customers_analyzed']
                               for r in self.per_file_results
                               if 'missing_periods' in r['results'])
            
            combined['missing_periods'] = {
                'customers_analyzed': total_analyzed,
                'customers_with_gaps': len(gaps_df),
                'gap_percentage': len(gaps_df) / total_analyzed * 100 if total_analyzed > 0 else 0,
                'avg_missing_pct': float(gaps_df['missing_statements'] / gaps_df['expected_statements'] * 100).mean()
            }
            
            print(f"  Customers analyzed: {total_analyzed:,}")
            print(f"  Customers with gaps: {len(gaps_df):,}")
            print(f"  Gap percentage: {len(gaps_df)/total_analyzed*100:.1f}%")
            
            # Save gaps report
            gaps_df.to_csv('combined_customers_with_missing_periods.csv', index=False)
            print(f"  Saved: combined_customers_with_missing_periods.csv")
        
        self.aggregated_results = combined
        
        return combined
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    def create_combined_visualizations(self):
        """
        Create visualizations from aggregated results.
        """
        print("\n" + "="*80)
        print("GENERATING COMBINED VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Interval distribution
        if len(self.all_intervals) > 0:
            ax1 = plt.subplot(2, 3, 1)
            intervals = np.array(self.all_intervals)
            # Filter outliers for better visualization
            intervals_plot = intervals[intervals < np.percentile(intervals, 99)]
            
            ax1.hist(intervals_plot, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax1.axvline(30.44, color='green', linestyle='--', linewidth=2, 
                       label='Expected (30.44 days)')
            ax1.set_xlabel('Days Between Statements', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('Combined: Distribution of Time Intervals', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # 2. Coverage span distribution
        if len(self.all_coverage) > 0:
            ax2 = plt.subplot(2, 3, 2)
            coverage_df = pd.DataFrame(self.all_coverage)
            
            ax2.hist(coverage_df['span_months'], bins=30, color='purple', 
                    edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Coverage Span (months)', fontsize=11)
            ax2.set_ylabel('Number of Customers', fontsize=11)
            ax2.set_title('Combined: Customer Coverage Span', fontsize=12, fontweight='bold')
            ax2.axvline(coverage_df['span_months'].median(), color='red', linestyle='--',
                       label=f"Median: {coverage_df['span_months'].median():.1f} months")
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Completeness distribution
        if len(self.all_coverage) > 0:
            ax3 = plt.subplot(2, 3, 3)
            coverage_df = pd.DataFrame(self.all_coverage)
            
            ax3.hist(coverage_df['completeness_pct'], bins=20, color='teal', 
                    edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Data Completeness (%)', fontsize=11)
            ax3.set_ylabel('Number of Customers', fontsize=11)
            ax3.set_title('Combined: Data Completeness', fontsize=12, fontweight='bold')
            ax3.axvline(80, color='red', linestyle='--', linewidth=2, label='80% threshold')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # 4. Missing statements distribution
        if len(self.all_gaps) > 0:
            ax4 = plt.subplot(2, 3, 4)
            gaps_df = pd.DataFrame(self.all_gaps)
            
            ax4.hist(gaps_df['missing_statements'], bins=20, color='crimson', 
                    edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Number of Missing Statements', fontsize=11)
            ax4.set_ylabel('Number of Customers', fontsize=11)
            ax4.set_title('Combined: Missing Statements Distribution', fontsize=12, fontweight='bold')
            ax4.grid(alpha=0.3)
        
        # 5. Per-file comparison
        ax5 = plt.subplot(2, 3, 5)
        file_names = [f"File {r['index']}" for r in self.per_file_results]
        file_customers = [r['results']['customer_grouping'].get('unique_customers', 0) 
                         for r in self.per_file_results]
        
        ax5.bar(range(len(file_names)), file_customers, color='steelblue', edgecolor='black')
        ax5.set_xticks(range(len(file_names)))
        ax5.set_xticklabels(file_names)
        ax5.set_ylabel('Number of Customers', fontsize=11)
        ax5.set_title('Customers per File', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Gap percentage per file
        if any('missing_periods' in r['results'] for r in self.per_file_results):
            ax6 = plt.subplot(2, 3, 6)
            gap_pcts = [r['results']['missing_periods'].get('gap_percentage', 0)
                       for r in self.per_file_results
                       if 'missing_periods' in r['results']]
            
            ax6.bar(range(len(gap_pcts)), gap_pcts, color='orange', edgecolor='black')
            ax6.set_xticks(range(len(gap_pcts)))
            ax6.set_xticklabels([f"File {i+1}" for i in range(len(gap_pcts))])
            ax6.set_ylabel('Gap Percentage (%)', fontsize=11)
            ax6.set_title('Missing Periods per File', fontsize=12, fontweight='bold')
            ax6.axhline(20, color='red', linestyle='--', label='20% threshold')
            ax6.legend()
            ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('combined_data_quality_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: combined_data_quality_analysis.png")
        plt.show()
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    
    def generate_combined_summary(self) -> str:
        """
        Generate comprehensive summary report.
        
        Returns:
            str: Summary report text
        """
        print("\n" + "="*80)
        print("GENERATING COMBINED SUMMARY REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMBINED DATA QUALITY ANALYSIS - SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Files Analyzed: {self.num_files}")
        
        for i, file_info in enumerate(self.per_file_results, 1):
            report_lines.append(f"  {i}. {file_info['file']}")
        
        # Overall statistics
        if 'customer_grouping' in self.aggregated_results:
            r = self.aggregated_results['customer_grouping']
            report_lines.append("\n" + "-"*80)
            report_lines.append("1. CUSTOMER ID GROUPING (COMBINED)")
            report_lines.append("-"*80)
            report_lines.append(f"Total Unique Customers: {r['total_customers_across_files']:,}")
            report_lines.append(f"Total Rows: {r['total_rows_across_files']:,}")
            report_lines.append(f"Average Rows per Customer: {r['avg_rows_per_customer']:.2f}")
            report_lines.append(f"All Files Grouped: {'✅ YES' if r['all_files_grouped'] else '⚠️ NO'}")
        
        if 'date_columns' in self.aggregated_results:
            r = self.aggregated_results['date_columns']
            report_lines.append("\n" + "-"*80)
            report_lines.append("2. DATE COLUMNS (COMBINED)")
            report_lines.append("-"*80)
            report_lines.append(f"Date Columns Found: {r['count']}")
            report_lines.append(f"Columns: {', '.join(r['columns_found'])}")
            
            if r['count'] == 1 and 'S_2' in r['columns_found']:
                report_lines.append("Status: ✅ PASS (Only S_2 found as expected)")
            elif r['count'] > 1:
                report_lines.append("Status: ⚠️ WARNING (Multiple date columns)")
        
        if 'monthly_intervals' in self.aggregated_results:
            r = self.aggregated_results['monthly_intervals']
            report_lines.append("\n" + "-"*80)
            report_lines.append("3. MONTHLY INTERVAL VALIDATION (COMBINED)")
            report_lines.append("-"*80)
            report_lines.append(f"Total Intervals Analyzed: {r['total_intervals']:,}")
            report_lines.append(f"Mean Interval: {r['mean_days']:.1f} days ({r['mean_days']/30.44:.2f} months)")
            report_lines.append(f"Median Interval: {r['median_days']:.1f} days")
            report_lines.append(f"Monthly Intervals: {r['monthly_percentage']:.1f}%")
            
            if r['monthly_percentage'] >= 90:
                status = "✅ EXCELLENT"
            elif r['monthly_percentage'] >= 75:
                status = "✅ GOOD"
            elif r['monthly_percentage'] >= 50:
                status = "⚠️ WARNING"
            else:
                status = "❌ POOR"
            report_lines.append(f"Status: {status}")
        
        if 'temporal_coverage' in self.aggregated_results:
            r = self.aggregated_results['temporal_coverage']
            report_lines.append("\n" + "-"*80)
            report_lines.append("4. TEMPORAL COVERAGE (COMBINED)")
            report_lines.append("-"*80)
            report_lines.append(f"Customers Analyzed: {r['customers_analyzed']:,}")
            report_lines.append(f"Mean Coverage Span: {r['mean_span_months']:.1f} months")
            report_lines.append(f"Mean Completeness: {r['mean_completeness']:.1f}%")
            report_lines.append(f"Customers with <80% completeness: {r['incomplete_customers']:,}")
            
            if r['incomplete_customers'] == 0:
                report_lines.append("Status: ✅ EXCELLENT")
            elif r['incomplete_customers'] < r['customers_analyzed'] * 0.1:
                report_lines.append("Status: ✅ GOOD")
            else:
                report_lines.append("Status: ⚠️ WARNING")
        
        if 'missing_periods' in self.aggregated_results:
            r = self.aggregated_results['missing_periods']
            report_lines.append("\n" + "-"*80)
            report_lines.append("5. MISSING TIME PERIODS (⭐ CRITICAL - COMBINED)")
            report_lines.append("-"*80)
            report_lines.append(f"Customers Analyzed: {r['customers_analyzed']:,}")
            report_lines.append(f"Customers with Complete Data: {r['customers_analyzed'] - r['customers_with_gaps']:,}")
            report_lines.append(f"Customers with Missing Periods: {r['customers_with_gaps']:,} ({r['gap_percentage']:.1f}%)")
            
            if r['customers_with_gaps'] == 0:
                report_lines.append("\nStatus: ✅ EXCELLENT - Perfect for time series decomposition!")
                recommendation = "PROCEED with time series decomposition"
            else:
                avg_missing = r.get('avg_missing_pct', 0)
                if avg_missing < 10:
                    status = "⚠️ LOW Severity"
                    recommendation = "PROCEED with time series decomposition (use imputation)"
                elif avg_missing < 25:
                    status = "⚠️ MODERATE Severity"
                    recommendation = "PROCEED WITH CAUTION (consider filtering)"
                else:
                    status = "❌ HIGH Severity"
                    recommendation = "NOT RECOMMENDED for time series decomposition"
                
                report_lines.append(f"\nStatus: {status}")
                report_lines.append(f"Average Missing: {avg_missing:.1f}%")
                report_lines.append(f"\nRecommendation: {recommendation}")
        
        # Overall conclusion
        report_lines.append("\n" + "="*80)
        report_lines.append("OVERALL CONCLUSION (ACROSS ALL FILES)")
        report_lines.append("="*80)
        
        if 'missing_periods' in self.aggregated_results:
            gap_pct = self.aggregated_results['missing_periods']['gap_percentage']
            
            if gap_pct == 0:
                report_lines.append("\n✅ DATA IS EXCELLENT FOR TIME SERIES ANALYSIS")
                report_lines.append("   - All customers have complete monthly statements")
                report_lines.append("   - Perfect for seasonal decomposition and autocorrelation")
            elif gap_pct < 20:
                report_lines.append("\n✅ DATA IS GOOD FOR TIME SERIES ANALYSIS")
                report_lines.append("   - Most customers have complete data")
                report_lines.append("   - Minor gaps can be handled with imputation")
            else:
                report_lines.append("\n⚠️ DATA HAS LIMITATIONS FOR TIME SERIES ANALYSIS")
                report_lines.append(f"   - {gap_pct:.1f}% of customers have missing periods")
                report_lines.append("   - RECOMMENDED: Use simple aggregation features instead")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("FILES GENERATED")
        report_lines.append("="*80)
        report_lines.append("- combined_temporal_coverage_report.csv")
        report_lines.append("- combined_customers_with_missing_periods.csv")
        report_lines.append("- combined_data_quality_analysis.png")
        report_lines.append("- combined_data_quality_summary.txt")
        
        report_text = "\n".join(report_lines)
        
        # Save
        with open('combined_data_quality_summary.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nSummary saved to: combined_data_quality_summary.txt")
        
        return report_text
    
    # ========================================================================
    # RUN ALL
    # ========================================================================
    
    def run_all_checks(self):
        """
        Run complete analysis pipeline on all files.
        """
        print("\n" + "="*80)
        print("STARTING MEMORY-EFFICIENT ANALYSIS")
        print("="*80)
        
        # Analyze each file
        for i, filepath in enumerate(self.file_list, 1):
            if not os.path.exists(filepath):
                print(f"\n⚠️ Skipping {filepath} - file not found")
                continue
            
            self.analyze_single_file(filepath, i)
            
            print(f"\n{'='*80}")
            print(f"Progress: {i}/{self.num_files} files completed")
            print(f"{'='*80}")
        
        # Aggregate results
        print("\n")
        self.aggregate_results()
        
        # Create visualizations
        self.create_combined_visualizations()
        
        # Generate summary
        self.generate_combined_summary()
        
        print("\n" + "="*80)
        print("COMPLETE ANALYSIS FINISHED!")
        print("="*80)


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main function to run memory-efficient analysis on split files.
    """
    
    # Define your file list
    file_list = [
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\data\train_data_split_0.csv",
        r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\data\train_data_split_1.csv',
        r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\data\train_data_split_2.csv',
        r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\data\train_data_split_3.csv',
        r'C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\data\train_data_split_4.csv'
    ]
    
    # Alternative: Auto-detect files matching pattern
    # import glob
    # file_list = sorted(glob.glob('train_data_part*.csv'))
    
    print("="*80)
    print("MEMORY-EFFICIENT DATA QUALITY ANALYZER FOR SPLIT FILES")
    print("="*80)
    
    # Initialize analyzer
    analyzer = MemoryEfficientDataQualityAnalyzer(file_list)
    
    # Run analysis
    analyzer.run_all_checks()
    
    print("\n" + "="*80)
    print("ALL DONE! Review the generated files:")
    print("  - combined_data_quality_summary.txt")
    print("  - combined_data_quality_analysis.png")
    print("  - combined_temporal_coverage_report.csv")
    print("  - combined_customers_with_missing_periods.csv")
    print("="*80)


if __name__ == "__main__":
    main()