"""
================================================================================
DATA QUALITY ANALYSIS FOR CREDIT CARD TIME SERIES DATA
================================================================================

Purpose: Comprehensive data quality checks for temporal credit card data
Checks:
    1. Customer ID grouping and sorting
    2. Date column identification and interpretation
    3. Monthly interval validation
    4. Temporal coverage analysis
    5. Missing time period detection (critical for time series)

Output: 
    - Console report with statistics
    - CSV files with detailed findings
    - Visualizations (histograms, distributions, timelines)

Usage:
    # Run with default filename (train_data.csv)
    python data_quality_analysis.py

    # Or specify custom filename
    python data_quality_analysis.py your_data.csv

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

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class CreditDataQualityAnalyzer:
    """
    Comprehensive data quality analysis for temporal credit card data.
    
    This class performs critical checks to ensure data is suitable for
    time series analysis, particularly for seasonal decomposition and
    autocorrelation calculations.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize analyzer and load data.
        
        Args:
            filepath (str): Path to raw credit card data CSV
        """
        print("="*80)
        print("DATA QUALITY ANALYSIS FOR TIME SERIES")
        print("="*80)
        
        print(f"\nLoading data from: {filepath}")
        self.data = pd.read_csv(filepath)
        
        print(f"Data loaded successfully!")
        print(f"  Shape: {self.data.shape}")
        print(f"  Rows: {len(self.data):,}")
        print(f"  Columns: {len(self.data.columns)}")
        
        # Store analysis results
        self.results = {}
        
    # ========================================================================
    # CHECK 1: CUSTOMER ID GROUPING AND SORTING
    # ========================================================================
    
    def check_customer_grouping(self) -> Dict:
        """
        Check if customer IDs are grouped together and sorted.
        
        For efficient time series processing, customer data should be:
        1. Grouped: All rows for a customer are together
        2. Sorted: Customers appear in order
        
        Returns:
            Dict: Analysis results with grouping statistics
        """
        print("\n" + "="*80)
        print("CHECK 1: CUSTOMER ID GROUPING AND SORTING")
        print("="*80)
        
        if 'customer_ID' not in self.data.columns:
            print("❌ ERROR: 'customer_ID' column not found!")
            return {'error': 'customer_ID column missing'}
        
        customer_col = self.data['customer_ID']
        
        # Count unique customers
        unique_customers = customer_col.nunique()
        total_rows = len(self.data)
        avg_rows_per_customer = total_rows / unique_customers
        
        print(f"\nBasic Statistics:")
        print(f"  Total customers: {unique_customers:,}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Average rows per customer: {avg_rows_per_customer:.2f}")
        
        # Check if data is grouped (all rows for each customer are together)
        customer_changes = (customer_col != customer_col.shift()).sum()
        is_grouped = customer_changes == unique_customers
        
        print(f"\nGrouping Check:")
        print(f"  Number of customer changes: {customer_changes:,}")
        print(f"  Expected if grouped: {unique_customers:,}")
        
        if is_grouped:
            print(f"  ✅ PASS: Data is properly grouped (all rows for each customer are together)")
        else:
            print(f"  ⚠️  WARNING: Data is NOT grouped!")
            print(f"     Customer IDs appear in {customer_changes:,} separate blocks")
            print(f"     This means some customers have scattered rows")
            
            # Find customers with scattered rows
            customer_first_occurrence = {}
            customer_last_occurrence = {}
            
            for idx, cust_id in enumerate(customer_col):
                if cust_id not in customer_first_occurrence:
                    customer_first_occurrence[cust_id] = idx
                customer_last_occurrence[cust_id] = idx
            
            scattered_customers = []
            for cust_id in customer_first_occurrence.keys():
                first_idx = customer_first_occurrence[cust_id]
                last_idx = customer_last_occurrence[cust_id]
                expected_rows = customer_col[first_idx:last_idx+1].value_counts()[cust_id]
                actual_rows = (customer_col == cust_id).sum()
                
                if expected_rows != actual_rows:
                    scattered_customers.append({
                        'customer_ID': cust_id,
                        'first_row': first_idx,
                        'last_row': last_idx,
                        'rows_in_range': expected_rows,
                        'total_rows': actual_rows
                    })
            
            if scattered_customers:
                print(f"\n     Found {len(scattered_customers)} customers with scattered rows")
                print(f"     Example scattered customers:")
                for i, cust in enumerate(scattered_customers[:5]):
                    print(f"       {i+1}. Customer {cust['customer_ID']}: "
                          f"appears in rows {cust['first_row']}-{cust['last_row']} "
                          f"but has gaps")
        
        # Check if customers are sorted
        customer_ids_list = customer_col.unique()
        is_sorted = all(customer_ids_list[i] <= customer_ids_list[i+1] 
                       for i in range(len(customer_ids_list)-1))
        
        print(f"\nSorting Check:")
        if is_sorted:
            print(f"  ✅ PASS: Customer IDs are sorted in ascending order")
        else:
            print(f"  ℹ️  INFO: Customer IDs are not sorted (this is OK)")
        
        # Distribution of rows per customer
        rows_per_customer = customer_col.value_counts()
        
        print(f"\nRows per Customer Distribution:")
        print(f"  Min: {rows_per_customer.min()}")
        print(f"  25th percentile: {rows_per_customer.quantile(0.25):.0f}")
        print(f"  Median: {rows_per_customer.median():.0f}")
        print(f"  75th percentile: {rows_per_customer.quantile(0.75):.0f}")
        print(f"  Max: {rows_per_customer.max()}")
        print(f"  Std Dev: {rows_per_customer.std():.2f}")
        
        # Identify customers with unusual row counts
        mean_rows = rows_per_customer.mean()
        std_rows = rows_per_customer.std()
        
        few_rows = rows_per_customer[rows_per_customer < mean_rows - 2*std_rows]
        many_rows = rows_per_customer[rows_per_customer > mean_rows + 2*std_rows]
        
        if len(few_rows) > 0:
            print(f"\n  ⚠️  {len(few_rows)} customers have unusually FEW rows (< {mean_rows - 2*std_rows:.0f})")
            print(f"     Min rows: {few_rows.min()}")
        
        if len(many_rows) > 0:
            print(f"  ⚠️  {len(many_rows)} customers have unusually MANY rows (> {mean_rows + 2*std_rows:.0f})")
            print(f"     Max rows: {many_rows.max()}")
        
        # Store results
        results = {
            'unique_customers': unique_customers,
            'total_rows': total_rows,
            'avg_rows_per_customer': avg_rows_per_customer,
            'is_grouped': is_grouped,
            'customer_changes': customer_changes,
            'is_sorted': is_sorted,
            'rows_per_customer_stats': {
                'min': int(rows_per_customer.min()),
                'median': float(rows_per_customer.median()),
                'max': int(rows_per_customer.max()),
                'std': float(rows_per_customer.std())
            }
        }
        
        self.results['customer_grouping'] = results
        
        return results
    
    # ========================================================================
    # CHECK 2: DATE COLUMN IDENTIFICATION
    # ========================================================================
    
    def check_date_columns(self) -> Dict:
        """
        Identify all date columns and interpret their meaning.
        
        Returns:
            Dict: Information about date columns found
        """
        print("\n" + "="*80)
        print("CHECK 2: DATE COLUMN IDENTIFICATION")
        print("="*80)
        
        date_columns = []
        
        # Check each column for date-like content
        print(f"\nScanning {len(self.data.columns)} columns for dates...")
        
        for col in self.data.columns:
            # Skip customer_ID
            if col == 'customer_ID':
                continue
            
            # Try to parse as date
            try:
                sample = self.data[col].dropna().head(100)
                
                # Check if column name suggests date
                if any(keyword in col.lower() for keyword in ['date', 'time', 'dt', 'day', 'month', 'year']):
                    pd.to_datetime(sample, errors='coerce')
                    date_columns.append(col)
                    continue
                
                # Try parsing as date
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:  # >80% successfully parsed
                    date_columns.append(col)
            except:
                continue
        
        print(f"\nFound {len(date_columns)} date column(s):")
        
        if len(date_columns) == 0:
            print("  ❌ ERROR: No date columns found!")
            print("     Cannot perform temporal analysis without dates")
            return {'error': 'No date columns found'}
        
        # Analyze each date column
        date_info = {}
        
        for col in date_columns:
            print(f"\n  Column: '{col}'")
            
            # Parse dates
            dates = pd.to_datetime(self.data[col], errors='coerce')
            
            valid_dates = dates.notna().sum()
            invalid_dates = dates.isna().sum()
            
            print(f"    Valid dates: {valid_dates:,} ({valid_dates/len(dates)*100:.2f}%)")
            if invalid_dates > 0:
                print(f"    Invalid/Missing: {invalid_dates:,} ({invalid_dates/len(dates)*100:.2f}%)")
            
            if valid_dates > 0:
                date_range = dates.dropna()
                print(f"    Date range: {date_range.min().date()} to {date_range.max().date()}")
                print(f"    Span: {(date_range.max() - date_range.min()).days} days")
                
                # Check if this is the statement date (S_2)
                if col == 'S_2':
                    print(f"    ✅ This is the STATEMENT DATE column (S_2)")
                    print(f"       Each row represents a monthly statement for a customer")
                else:
                    print(f"    ⚠️  Additional date column detected!")
                    print(f"       Need to interpret what this date represents")
                
                date_info[col] = {
                    'valid_dates': valid_dates,
                    'invalid_dates': invalid_dates,
                    'min_date': date_range.min(),
                    'max_date': date_range.max(),
                    'span_days': (date_range.max() - date_range.min()).days
                }
        
        # Conclusion
        print(f"\nConclusion:")
        if len(date_columns) == 1 and 'S_2' in date_columns:
            print(f"  ✅ PASS: Only one date column (S_2) found as expected")
            print(f"     This represents the statement date for each monthly record")
        elif len(date_columns) > 1:
            print(f"  ⚠️  WARNING: Multiple date columns found!")
            print(f"     Columns: {', '.join(date_columns)}")
            print(f"     Need to understand the meaning of each date column:")
            for col in date_columns:
                if col == 'S_2':
                    print(f"       - {col}: Statement date (monthly billing cycle)")
                else:
                    print(f"       - {col}: Unknown - needs investigation")
        
        self.results['date_columns'] = {
            'columns_found': date_columns,
            'count': len(date_columns),
            'details': date_info
        }
        
        return self.results['date_columns']
    
    # ========================================================================
    # CHECK 3: MONTHLY INTERVAL VALIDATION
    # ========================================================================
    
    def check_monthly_intervals(self, date_column: str = 'S_2') -> Dict:
        """
        Validate that each row represents approximately one month.
        
        Checks:
        1. Time gaps between consecutive statements per customer
        2. Duplicate dates (same customer, same date = potential duplicate)
        3. Non-monthly intervals
        
        Args:
            date_column (str): Name of the date column to analyze
            
        Returns:
            Dict: Validation results
        """
        print("\n" + "="*80)
        print("CHECK 3: MONTHLY INTERVAL VALIDATION")
        print("="*80)
        
        if date_column not in self.data.columns:
            print(f"❌ ERROR: Date column '{date_column}' not found!")
            return {'error': f'Column {date_column} not found'}
        
        # Parse dates
        print(f"\nAnalyzing temporal intervals in column '{date_column}'...")
        data_copy = self.data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column], errors='coerce')
        
        # Sort by customer and date
        data_copy = data_copy.sort_values(['customer_ID', date_column])
        
        # Calculate intervals between consecutive statements for each customer
        print("\nCalculating time intervals between consecutive statements...")
        
        intervals = []
        duplicate_dates = []
        interval_details = []
        
        for customer_id, group in data_copy.groupby('customer_ID'):
            dates = group[date_column].dropna().sort_values()
            
            if len(dates) < 2:
                continue
            
            # Check for duplicate dates
            duplicates = dates[dates.duplicated(keep=False)]
            if len(duplicates) > 0:
                duplicate_dates.append({
                    'customer_ID': customer_id,
                    'duplicate_dates': duplicates.unique().tolist(),
                    'count': len(duplicates)
                })
            
            # Calculate intervals
            for i in range(1, len(dates)):
                interval_days = (dates.iloc[i] - dates.iloc[i-1]).days
                intervals.append(interval_days)
                
                interval_details.append({
                    'customer_ID': customer_id,
                    'date_from': dates.iloc[i-1],
                    'date_to': dates.iloc[i],
                    'interval_days': interval_days,
                    'interval_months': interval_days / 30.44  # Average days per month
                })
        
        if len(intervals) == 0:
            print("  ❌ ERROR: No intervals could be calculated!")
            return {'error': 'No intervals calculated'}
        
        intervals = np.array(intervals)
        
        # Statistics
        print(f"\nInterval Statistics (in days):")
        print(f"  Total intervals analyzed: {len(intervals):,}")
        print(f"  Mean: {intervals.mean():.1f} days ({intervals.mean()/30.44:.2f} months)")
        print(f"  Median: {np.median(intervals):.1f} days ({np.median(intervals)/30.44:.2f} months)")
        print(f"  Std Dev: {intervals.std():.1f} days")
        print(f"  Min: {intervals.min()} days")
        print(f"  Max: {intervals.max()} days")
        
        # Expected monthly interval: 28-32 days
        monthly_min = 27
        monthly_max = 32
        
        monthly_intervals = ((intervals >= monthly_min) & (intervals <= monthly_max)).sum()
        monthly_pct = monthly_intervals / len(intervals) * 100
        
        print(f"\nMonthly Interval Analysis:")
        print(f"  Intervals within monthly range ({monthly_min}-{monthly_max} days): "
              f"{monthly_intervals:,} ({monthly_pct:.2f}%)")
        
        if monthly_pct >= 90:
            print(f"  ✅ EXCELLENT: >90% of intervals are approximately monthly")
        elif monthly_pct >= 75:
            print(f"  ✅ GOOD: >75% of intervals are approximately monthly")
        elif monthly_pct >= 50:
            print(f"  ⚠️  WARNING: Only {monthly_pct:.1f}% of intervals are monthly")
        else:
            print(f"  ❌ POOR: Only {monthly_pct:.1f}% of intervals are monthly")
        
        # Check for problematic intervals
        very_short = (intervals < 7).sum()  # Less than a week
        very_long = (intervals > 60).sum()  # More than 2 months
        
        if very_short > 0:
            print(f"\n  ⚠️  {very_short:,} intervals are VERY SHORT (< 7 days)")
            print(f"     This could indicate duplicate or erroneous records")
        
        if very_long > 0:
            print(f"  ⚠️  {very_long:,} intervals are VERY LONG (> 60 days)")
            print(f"     This indicates missing monthly statements")
        
        # Duplicate dates check
        print(f"\nDuplicate Date Check:")
        if len(duplicate_dates) > 0:
            print(f"  ⚠️  WARNING: Found {len(duplicate_dates)} customers with duplicate dates")
            print(f"     These could be duplicate records!")
            print(f"\n     Examples:")
            for i, dup in enumerate(duplicate_dates[:5]):
                print(f"       {i+1}. Customer {dup['customer_ID']}: "
                      f"{dup['count']} rows with duplicate dates")
            
            # Save to CSV
            dup_df = pd.DataFrame(duplicate_dates)
            dup_df.to_csv('duplicate_dates_report.csv', index=False)
            print(f"\n     Full report saved to: duplicate_dates_report.csv")
        else:
            print(f"  ✅ PASS: No duplicate dates found")
        
        # Interval distribution
        interval_df = pd.DataFrame(interval_details)
        
        results = {
            'total_intervals': len(intervals),
            'mean_days': float(intervals.mean()),
            'median_days': float(np.median(intervals)),
            'std_days': float(intervals.std()),
            'monthly_percentage': float(monthly_pct),
            'very_short_intervals': int(very_short),
            'very_long_intervals': int(very_long),
            'duplicate_dates_customers': len(duplicate_dates),
            'interval_distribution': interval_df
        }
        
        self.results['monthly_intervals'] = results
        
        return results
    
    # ========================================================================
    # CHECK 4: TEMPORAL COVERAGE ANALYSIS
    # ========================================================================
    
    def analyze_temporal_coverage(self, date_column: str = 'S_2') -> Dict:
        """
        Analyze start/end dates and temporal coverage for each customer.
        
        Args:
            date_column (str): Name of the date column
            
        Returns:
            Dict: Coverage analysis results
        """
        print("\n" + "="*80)
        print("CHECK 4: TEMPORAL COVERAGE ANALYSIS")
        print("="*80)
        
        if date_column not in self.data.columns:
            print(f"❌ ERROR: Date column '{date_column}' not found!")
            return {'error': f'Column {date_column} not found'}
        
        print(f"\nAnalyzing temporal coverage per customer...")
        
        # Parse dates
        data_copy = self.data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column], errors='coerce')
        
        # Calculate coverage for each customer
        coverage_list = []
        
        for customer_id, group in data_copy.groupby('customer_ID'):
            dates = group[date_column].dropna().sort_values()
            
            if len(dates) == 0:
                continue
            
            start_date = dates.min()
            end_date = dates.max()
            span_days = (end_date - start_date).days
            span_months = span_days / 30.44
            num_statements = len(dates)
            
            # Expected statements based on time span
            expected_statements = int(span_months) + 1
            missing_statements = max(0, expected_statements - num_statements)
            completeness = num_statements / expected_statements if expected_statements > 0 else 0
            
            coverage_list.append({
                'customer_ID': customer_id,
                'start_date': start_date,
                'end_date': end_date,
                'start_month': start_date.strftime('%Y-%m'),
                'end_month': end_date.strftime('%Y-%m'),
                'span_days': span_days,
                'span_months': span_months,
                'num_statements': num_statements,
                'expected_statements': expected_statements,
                'missing_statements': missing_statements,
                'completeness_pct': completeness * 100
            })
        
        coverage_df = pd.DataFrame(coverage_list)
        
        # Statistics
        print(f"\nTemporal Coverage Statistics:")
        print(f"  Customers analyzed: {len(coverage_df):,}")
        
        print(f"\n  Coverage Span (months):")
        print(f"    Min: {coverage_df['span_months'].min():.1f} months")
        print(f"    Median: {coverage_df['span_months'].median():.1f} months")
        print(f"    Mean: {coverage_df['span_months'].mean():.1f} months")
        print(f"    Max: {coverage_df['span_months'].max():.1f} months")
        
        print(f"\n  Number of Statements:")
        print(f"    Min: {coverage_df['num_statements'].min()}")
        print(f"    Median: {coverage_df['num_statements'].median():.0f}")
        print(f"    Mean: {coverage_df['num_statements'].mean():.1f}")
        print(f"    Max: {coverage_df['num_statements'].max()}")
        
        print(f"\n  Data Completeness:")
        print(f"    Mean completeness: {coverage_df['completeness_pct'].mean():.1f}%")
        print(f"    Median completeness: {coverage_df['completeness_pct'].median():.1f}%")
        
        # Identify customers with poor completeness
        incomplete = coverage_df[coverage_df['completeness_pct'] < 80]
        
        if len(incomplete) > 0:
            print(f"\n  ⚠️  WARNING: {len(incomplete):,} customers ({len(incomplete)/len(coverage_df)*100:.1f}%) "
                  f"have <80% completeness")
            print(f"     These customers are missing monthly statements")
            print(f"     This will negatively impact time series analysis!")
        else:
            print(f"\n  ✅ EXCELLENT: All customers have ≥80% complete data")
        
        # Start date distribution
        print(f"\nStart Date Distribution:")
        start_months = coverage_df['start_month'].value_counts().sort_index()
        print(f"  Earliest start: {start_months.index[0]}")
        print(f"  Latest start: {start_months.index[-1]}")
        print(f"  Most common start month: {start_months.idxmax()} ({start_months.max():,} customers)")
        
        # End date distribution
        print(f"\nEnd Date Distribution:")
        end_months = coverage_df['end_month'].value_counts().sort_index()
        print(f"  Earliest end: {end_months.index[0]}")
        print(f"  Latest end: {end_months.index[-1]}")
        print(f"  Most common end month: {end_months.idxmax()} ({end_months.max():,} customers)")
        
        # Save detailed report
        coverage_df.to_csv('temporal_coverage_report.csv', index=False)
        print(f"\nDetailed coverage report saved to: temporal_coverage_report.csv")
        
        self.results['temporal_coverage'] = {
            'coverage_df': coverage_df,
            'mean_span_months': float(coverage_df['span_months'].mean()),
            'mean_completeness': float(coverage_df['completeness_pct'].mean()),
            'incomplete_customers': len(incomplete)
        }
        
        return self.results['temporal_coverage']
    
    # ========================================================================
    # CHECK 5: MISSING TIME PERIODS DETECTION (CRITICAL)
    # ========================================================================
    
    def detect_missing_periods(self, date_column: str = 'S_2') -> Dict:
        """
        Critical check: Detect customers with missing time periods.
        
        This is the most important check for time series decomposition.
        Missing periods will break seasonality and autocorrelation calculations.
        
        Args:
            date_column (str): Name of the date column
            
        Returns:
            Dict: Missing period analysis
        """
        print("\n" + "="*80)
        print("CHECK 5: MISSING TIME PERIODS DETECTION (CRITICAL)")
        print("="*80)
        print("\nThis is the MOST IMPORTANT check for time series analysis!")
        print("Missing monthly statements will break seasonality and autocorrelation.")
        
        if date_column not in self.data.columns:
            print(f"❌ ERROR: Date column '{date_column}' not found!")
            return {'error': f'Column {date_column} not found'}
        
        # Parse dates
        data_copy = self.data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column], errors='coerce')
        data_copy = data_copy.sort_values(['customer_ID', date_column])
        
        print(f"\nAnalyzing {data_copy['customer_ID'].nunique():,} customers for missing periods...")
        
        customers_with_gaps = []
        gap_details = []
        
        for customer_id, group in data_copy.groupby('customer_ID'):
            dates = group[date_column].dropna().sort_values()
            
            if len(dates) < 2:
                continue
            
            # Generate expected monthly sequence
            start_date = dates.min()
            end_date = dates.max()
            
            # Create expected monthly dates
            expected_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='MS'  # Month start frequency
            )
            
            # Convert actual dates to month-start for comparison
            actual_months = pd.to_datetime(dates.dt.to_period('M').dt.to_timestamp())
            
            # Find missing months
            missing_months = set(expected_dates) - set(actual_months)
            
            if len(missing_months) > 0:
                customers_with_gaps.append({
                    'customer_ID': customer_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'expected_statements': len(expected_dates),
                    'actual_statements': len(actual_months.unique()),
                    'missing_statements': len(missing_months),
                    'missing_months': sorted(missing_months)
                })
                
                for missing_month in missing_months:
                    gap_details.append({
                        'customer_ID': customer_id,
                        'missing_month': missing_month.strftime('%Y-%m'),
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d')
                    })
        
        # Summary
        total_customers = data_copy['customer_ID'].nunique()
        customers_with_gaps_count = len(customers_with_gaps)
        customers_complete = total_customers - customers_with_gaps_count
        
        print(f"\nMissing Period Summary:")
        print(f"  Total customers: {total_customers:,}")
        print(f"  Customers with complete data: {customers_complete:,} "
              f"({customers_complete/total_customers*100:.1f}%)")
        print(f"  Customers with missing periods: {customers_with_gaps_count:,} "
              f"({customers_with_gaps_count/total_customers*100:.1f}%)")
        
        if customers_with_gaps_count > 0:
            gaps_df = pd.DataFrame(customers_with_gaps)
            
            print(f"\n  ⚠️  CRITICAL WARNING: {customers_with_gaps_count:,} customers have missing statements!")
            print(f"\n  Missing Statements Statistics:")
            print(f"    Mean missing: {gaps_df['missing_statements'].mean():.1f}")
            print(f"    Median missing: {gaps_df['missing_statements'].median():.0f}")
            print(f"    Max missing: {gaps_df['missing_statements'].max()}")
            
            # Show examples
            print(f"\n  Examples of customers with missing periods:")
            for i, row in gaps_df.head(10).iterrows():
                print(f"    {i+1}. Customer {row['customer_ID']}: "
                      f"Missing {row['missing_statements']} out of {row['expected_statements']} statements "
                      f"({row['start_date'].strftime('%Y-%m')} to {row['end_date'].strftime('%Y-%m')})")
            
            # Impact assessment
            print(f"\n  IMPACT ON TIME SERIES ANALYSIS:")
            print(f"  ❌ Seasonality: Irregular spacing will distort seasonal patterns")
            print(f"  ❌ Autocorrelation: Missing lags will produce incorrect ACF values")
            print(f"  ❌ Trend: Gaps may be interpreted as trend changes")
            
            print(f"\n  RECOMMENDATIONS:")
            
            # Check severity
            avg_missing_pct = (gaps_df['missing_statements'] / gaps_df['expected_statements'] * 100).mean()
            
            if avg_missing_pct < 10:
                print(f"  ⚠️  Severity: LOW (average {avg_missing_pct:.1f}% missing)")
                print(f"     → Proceed with time series decomposition")
                print(f"     → Impute missing values using forward-fill or interpolation")
            elif avg_missing_pct < 25:
                print(f"  ⚠️  Severity: MODERATE (average {avg_missing_pct:.1f}% missing)")
                print(f"     → Proceed with caution")
                print(f"     → Consider excluding customers with >20% missing data")
                print(f"     → Use robust imputation methods")
            else:
                print(f"  ❌ Severity: HIGH (average {avg_missing_pct:.1f}% missing)")
                print(f"     → Time series decomposition NOT RECOMMENDED")
                print(f"     → Use simple aggregation features instead")
                print(f"     → Or filter to customers with complete data only")
            
            # Save detailed reports
            gaps_df.to_csv('customers_with_missing_periods.csv', index=False)
            print(f"\n  Detailed gap report saved to: customers_with_missing_periods.csv")
            
            gap_details_df = pd.DataFrame(gap_details)
            gap_details_df.to_csv('missing_periods_detailed.csv', index=False)
            print(f"  All missing periods saved to: missing_periods_detailed.csv")
            
        else:
            print(f"\n  ✅ EXCELLENT: All customers have complete monthly statements!")
            print(f"     Time series decomposition will work perfectly!")
        
        self.results['missing_periods'] = {
            'total_customers': total_customers,
            'customers_complete': customers_complete,
            'customers_with_gaps': customers_with_gaps_count,
            'gap_percentage': customers_with_gaps_count / total_customers * 100 if total_customers > 0 else 0
        }
        
        if customers_with_gaps_count > 0:
            self.results['missing_periods']['gaps_df'] = gaps_df
            self.results['missing_periods']['avg_missing_pct'] = avg_missing_pct
        
        return self.results['missing_periods']
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def create_visualizations(self, date_column: str = 'S_2'):
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            date_column (str): Name of the date column
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 12))
        
        # Parse dates
        data_copy = self.data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column], errors='coerce')
        
        # 1. Rows per customer distribution
        ax1 = plt.subplot(2, 3, 1)
        rows_per_customer = data_copy['customer_ID'].value_counts()
        ax1.hist(rows_per_customer, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Rows per Customer', fontsize=11)
        ax1.set_ylabel('Number of Customers', fontsize=11)
        ax1.set_title('Distribution of Rows per Customer', fontsize=12, fontweight='bold')
        ax1.axvline(rows_per_customer.median(), color='red', linestyle='--', 
                   label=f'Median: {rows_per_customer.median():.0f}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Temporal interval distribution
        if 'monthly_intervals' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            interval_df = self.results['monthly_intervals']['interval_distribution']
            
            # Filter extreme outliers for better visualization
            intervals_plot = interval_df['interval_days']
            intervals_plot = intervals_plot[intervals_plot < intervals_plot.quantile(0.99)]
            
            ax2.hist(intervals_plot, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax2.axvline(30.44, color='green', linestyle='--', linewidth=2, 
                       label='Expected (30.44 days)')
            ax2.set_xlabel('Days Between Statements', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title('Distribution of Time Intervals', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Start date distribution
        if 'temporal_coverage' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            coverage_df = self.results['temporal_coverage']['coverage_df']
            
            start_months = coverage_df['start_month'].value_counts().sort_index()
            ax3.plot(range(len(start_months)), start_months.values, marker='o', 
                    color='darkgreen', linewidth=2)
            ax3.set_xlabel('Time', fontsize=11)
            ax3.set_ylabel('Number of Customers Starting', fontsize=11)
            ax3.set_title('Customer Start Date Distribution', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # Show only some x-labels to avoid crowding
            step = max(1, len(start_months) // 10)
            ax3.set_xticks(range(0, len(start_months), step))
            ax3.set_xticklabels(start_months.index[::step], rotation=45, ha='right')
        
        # 4. Coverage span distribution
        if 'temporal_coverage' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            coverage_df = self.results['temporal_coverage']['coverage_df']
            
            ax4.hist(coverage_df['span_months'], bins=30, color='purple', 
                    edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Coverage Span (months)', fontsize=11)
            ax4.set_ylabel('Number of Customers', fontsize=11)
            ax4.set_title('Distribution of Customer Coverage Span', fontsize=12, fontweight='bold')
            ax4.axvline(coverage_df['span_months'].median(), color='red', linestyle='--',
                       label=f"Median: {coverage_df['span_months'].median():.1f} months")
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        # 5. Completeness distribution
        if 'temporal_coverage' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            coverage_df = self.results['temporal_coverage']['coverage_df']
            
            ax5.hist(coverage_df['completeness_pct'], bins=20, color='teal', 
                    edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Data Completeness (%)', fontsize=11)
            ax5.set_ylabel('Number of Customers', fontsize=11)
            ax5.set_title('Distribution of Data Completeness', fontsize=12, fontweight='bold')
            ax5.axvline(80, color='red', linestyle='--', linewidth=2, label='80% threshold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # 6. Missing periods summary
        if 'missing_periods' in self.results and 'gaps_df' in self.results['missing_periods']:
            ax6 = plt.subplot(2, 3, 6)
            gaps_df = self.results['missing_periods']['gaps_df']
            
            ax6.hist(gaps_df['missing_statements'], bins=20, color='crimson', 
                    edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Number of Missing Statements', fontsize=11)
            ax6.set_ylabel('Number of Customers', fontsize=11)
            ax6.set_title('Distribution of Missing Statements\n(For Customers with Gaps)', 
                         fontsize=12, fontweight='bold')
            ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_quality_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: data_quality_analysis.png")
        plt.show()
    
    # ========================================================================
    # FINAL SUMMARY REPORT
    # ========================================================================
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of all checks.
        
        Returns:
            str: Summary report text
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA QUALITY SUMMARY REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DATA QUALITY ANALYSIS - SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {len(self.data):,} rows, {len(self.data.columns)} columns")
        
        # Check 1: Customer Grouping
        if 'customer_grouping' in self.results:
            r = self.results['customer_grouping']
            report_lines.append("\n" + "-"*80)
            report_lines.append("1. CUSTOMER ID GROUPING AND SORTING")
            report_lines.append("-"*80)
            report_lines.append(f"Total Customers: {r['unique_customers']:,}")
            report_lines.append(f"Average Rows per Customer: {r['avg_rows_per_customer']:.2f}")
            report_lines.append(f"Grouped: {'✅ YES' if r['is_grouped'] else '❌ NO'}")
            report_lines.append(f"Sorted: {'✅ YES' if r['is_sorted'] else 'ℹ️  NO (OK)'}")
            
        # Check 2: Date Columns
        if 'date_columns' in self.results:
            r = self.results['date_columns']
            report_lines.append("\n" + "-"*80)
            report_lines.append("2. DATE COLUMNS")
            report_lines.append("-"*80)
            report_lines.append(f"Date Columns Found: {r['count']}")
            report_lines.append(f"Columns: {', '.join(r['columns_found'])}")
            if r['count'] == 1 and 'S_2' in r['columns_found']:
                report_lines.append("Status: ✅ PASS (Only S_2 found as expected)")
            elif r['count'] > 1:
                report_lines.append("Status: ⚠️  WARNING (Multiple date columns)")
        
        # Check 3: Monthly Intervals
        if 'monthly_intervals' in self.results:
            r = self.results['monthly_intervals']
            report_lines.append("\n" + "-"*80)
            report_lines.append("3. MONTHLY INTERVAL VALIDATION")
            report_lines.append("-"*80)
            report_lines.append(f"Mean Interval: {r['mean_days']:.1f} days ({r['mean_days']/30.44:.2f} months)")
            report_lines.append(f"Monthly Intervals: {r['monthly_percentage']:.1f}%")
            
            if r['monthly_percentage'] >= 90:
                status = "✅ EXCELLENT"
            elif r['monthly_percentage'] >= 75:
                status = "✅ GOOD"
            elif r['monthly_percentage'] >= 50:
                status = "⚠️  WARNING"
            else:
                status = "❌ POOR"
            report_lines.append(f"Status: {status}")
            
            if r['duplicate_dates_customers'] > 0:
                report_lines.append(f"⚠️  Duplicate Dates: {r['duplicate_dates_customers']} customers affected")
        
        # Check 4: Temporal Coverage
        if 'temporal_coverage' in self.results:
            r = self.results['temporal_coverage']
            report_lines.append("\n" + "-"*80)
            report_lines.append("4. TEMPORAL COVERAGE")
            report_lines.append("-"*80)
            report_lines.append(f"Mean Coverage Span: {r['mean_span_months']:.1f} months")
            report_lines.append(f"Mean Completeness: {r['mean_completeness']:.1f}%")
            report_lines.append(f"Customers with <80% completeness: {r['incomplete_customers']:,}")
            
            if r['incomplete_customers'] == 0:
                report_lines.append("Status: ✅ EXCELLENT")
            elif r['incomplete_customers'] < r['coverage_df'].shape[0] * 0.1:
                report_lines.append("Status: ✅ GOOD")
            else:
                report_lines.append("Status: ⚠️  WARNING")
        
        # Check 5: Missing Periods (MOST CRITICAL)
        if 'missing_periods' in self.results:
            r = self.results['missing_periods']
            report_lines.append("\n" + "-"*80)
            report_lines.append("5. MISSING TIME PERIODS (⭐ CRITICAL FOR TIME SERIES)")
            report_lines.append("-"*80)
            report_lines.append(f"Total Customers: {r['total_customers']:,}")
            report_lines.append(f"Complete Data: {r['customers_complete']:,} "
                              f"({r['customers_complete']/r['total_customers']*100:.1f}%)")
            report_lines.append(f"Missing Periods: {r['customers_with_gaps']:,} "
                              f"({r['gap_percentage']:.1f}%)")
            
            if r['customers_with_gaps'] == 0:
                report_lines.append("\nStatus: ✅ EXCELLENT - Perfect for time series decomposition!")
                recommendation = "PROCEED with time series decomposition"
            else:
                avg_missing = r.get('avg_missing_pct', 0)
                if avg_missing < 10:
                    status = "⚠️  LOW Severity"
                    recommendation = "PROCEED with time series decomposition (use imputation)"
                elif avg_missing < 25:
                    status = "⚠️  MODERATE Severity"
                    recommendation = "PROCEED WITH CAUTION (consider filtering)"
                else:
                    status = "❌ HIGH Severity"
                    recommendation = "NOT RECOMMENDED for time series decomposition"
                
                report_lines.append(f"\nStatus: {status}")
                report_lines.append(f"Average Missing: {avg_missing:.1f}%")
                report_lines.append(f"\nRecommendation: {recommendation}")
        
        # Overall Conclusion
        report_lines.append("\n" + "="*80)
        report_lines.append("OVERALL CONCLUSION")
        report_lines.append("="*80)
        
        # Determine overall suitability for time series analysis
        if 'missing_periods' in self.results:
            gap_pct = self.results['missing_periods']['gap_percentage']
            
            if gap_pct == 0:
                report_lines.append("\n✅ DATA IS EXCELLENT FOR TIME SERIES ANALYSIS")
                report_lines.append("   - All customers have complete monthly statements")
                report_lines.append("   - Perfect for seasonal decomposition and autocorrelation")
                report_lines.append("   - Proceed with full time series feature extraction")
            elif gap_pct < 20:
                report_lines.append("\n✅ DATA IS GOOD FOR TIME SERIES ANALYSIS")
                report_lines.append("   - Most customers have complete data")
                report_lines.append("   - Minor gaps can be handled with imputation")
                report_lines.append("   - Proceed with time series decomposition")
                report_lines.append("   - Consider forward-fill or interpolation for missing values")
            else:
                report_lines.append("\n⚠️  DATA HAS LIMITATIONS FOR TIME SERIES ANALYSIS")
                report_lines.append(f"   - {gap_pct:.1f}% of customers have missing periods")
                report_lines.append("   - Time series decomposition may produce unreliable results")
                report_lines.append("   - RECOMMENDED: Use simple aggregation features instead")
                report_lines.append("   - OR: Filter to customers with >80% completeness")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("FILES GENERATED")
        report_lines.append("="*80)
        report_lines.append("- temporal_coverage_report.csv (detailed coverage per customer)")
        if 'missing_periods' in self.results and self.results['missing_periods']['customers_with_gaps'] > 0:
            report_lines.append("- customers_with_missing_periods.csv (customers with gaps)")
            report_lines.append("- missing_periods_detailed.csv (all missing months)")
        report_lines.append("- data_quality_analysis.png (visualization)")
        report_lines.append("- data_quality_summary.txt (this report)")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open('data_quality_summary.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\nSummary report saved to: data_quality_summary.txt")
        
        return report_text
    
    # ========================================================================
    # RUN ALL CHECKS
    # ========================================================================
    
    def run_all_checks(self, date_column: str = 'S_2'):
        """
        Run all data quality checks in sequence.
        
        Args:
            date_column (str): Name of the date column to analyze
        """
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("="*80)
        print("\nThis will perform 5 critical checks for time series data quality.\n")
        
        # Run checks
        self.check_customer_grouping()
        self.check_date_columns()
        self.check_monthly_intervals(date_column)
        self.analyze_temporal_coverage(date_column)
        self.detect_missing_periods(date_column)
        
        # Create visualizations
        self.create_visualizations(date_column)
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS COMPLETE!")
        print("="*80)
        print("\nPlease review:")
        print("1. data_quality_summary.txt - Overall summary")
        print("2. data_quality_analysis.png - Visual analysis")
        print("3. temporal_coverage_report.csv - Coverage details")
        print("4. Console output above for detailed findings")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main function to run data quality analysis.
    """
    import sys
    
    # Get filename from command line or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'train_data.csv'
    
    print("="*80)
    print("CREDIT CARD DATA QUALITY ANALYZER")
    print("="*80)
    print(f"\nAnalyzing file: {filename}\n")
    
    # Initialize analyzer
    analyzer = CreditDataQualityAnalyzer(filename)
    
    # Run all checks
    analyzer.run_all_checks(date_column='S_2')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - REVIEW OUTPUT FILES")
    print("="*80)


if __name__ == "__main__":
    main()