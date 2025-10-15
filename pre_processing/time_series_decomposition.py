"""
================================================================================
TIME SERIES DECOMPOSITION PREPROCESSING
================================================================================

Purpose: Extract temporal patterns from credit card data using time series analysis
Best for: Bayesian Networks, interpretable models requiring temporal features

Input:  train_data.csv with columns:
        - customer_ID: Unique customer identifier
        - S_2: Statement date (format: YYYY-MM-DD)
        - 188 feature columns (mix of continuous and discrete)
        - target: Binary default indicator (0=no default, 1=default)

Output: Rich temporal features including:
        - Trend features (slope, direction, strength)
        - Seasonality features (periodic patterns, amplitude)
        - Volatility features (irregular fluctuations, spikes)
        - Autocorrelation features (temporal dependencies)
        - Change point features (behavioral shifts)

Author: Credit Risk Team
Date: 2025
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
import warnings
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')


class TimeSeriesPreprocessor:
    """
    Preprocess temporal credit card data using time series decomposition.
    
    This class extracts rich temporal features by decomposing time series into:
    - Trend: Long-term directional changes
    - Seasonality: Repeating patterns
    - Volatility: Irregular fluctuations
    - Autocorrelation: Temporal dependencies
    - Change points: Sudden behavioral shifts
    
    These features are particularly useful for Bayesian networks and models
    that benefit from explicit temporal structure.
    """
    
    def __init__(self, seasonal_period: int = 3):
        """
        Initialize the time series preprocessor.
        
        Args:
            seasonal_period (int): Period for seasonality detection (default=3 for quarterly)
                                  3 = quarterly patterns (common in finance)
                                  12 = annual patterns (if monthly data available)
        """
        self.seasonal_period = seasonal_period
        
        # Known categorical features (these won't get time series decomposition)
        self.categorical_features = [
            'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 
            'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
        ]
        
        self.numerical_features = None
        
    def identify_feature_types(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features.
        
        Args:
            data (pd.DataFrame): Raw temporal data
            
        Returns:
            Tuple[List[str], List[str]]: (numerical_features, categorical_features)
        """
        all_columns = data.columns.tolist()
        exclude_columns = ['customer_ID', 'S_2', 'target']
        feature_columns = [col for col in all_columns if col not in exclude_columns]
        
        numerical = [col for col in feature_columns 
                    if col not in self.categorical_features]
        categorical = [col for col in feature_columns 
                      if col in self.categorical_features]
        
        print(f"Identified {len(numerical)} numerical features for time series analysis")
        print(f"Identified {len(categorical)} categorical features (simple aggregation)")
        
        return numerical, categorical
    
    # ========================================================================
    # TREND EXTRACTION
    # ========================================================================
    
    def extract_trend_features(self, series: pd.Series) -> Dict[str, float]:
        """
        Extract trend features using linear regression.
        
        Trend captures the long-term directional change in a time series.
        For credit default prediction:
        - Positive slope in debt = increasing risk
        - Negative slope in payments = increasing risk
        - High R² = consistent trend (predictable)
        - Low R² = erratic behavior (unpredictable, risky)
        
        Args:
            series (pd.Series): Time series of a single feature
            
        Returns:
            Dict[str, float]: Dictionary with keys:
                - trend_slope: Rate of change per time period
                - trend_intercept: Starting value of trend line
                - trend_r2: Strength of trend (0-1, higher = stronger trend)
                
        Example:
            series = [1000, 1100, 1200, 1300, 1400]  # Increasing debt
            result = {
                'trend_slope': 100.0,      # Increasing by $100/month
                'trend_intercept': 1000.0, # Started at $1000
                'trend_r2': 1.0            # Perfect linear trend
            }
        """
        # Need at least 3 points to fit a line
        if len(series) < 3 or series.isna().all():
            return {
                'trend_slope': np.nan,
                'trend_intercept': np.nan,
                'trend_r2': np.nan
            }
        
        # Remove missing values
        valid_mask = ~series.isna()
        if valid_mask.sum() < 3:
            return {
                'trend_slope': np.nan,
                'trend_intercept': np.nan,
                'trend_r2': np.nan
            }
        
        y = series[valid_mask].values
        x = np.arange(len(y))
        
        try:
            # Fit linear regression: y = slope * x + intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'trend_slope': slope,           # Rate of change
                'trend_intercept': intercept,   # Starting point
                'trend_r2': r_value ** 2        # Goodness of fit (0-1)
            }
        except:
            return {
                'trend_slope': np.nan,
                'trend_intercept': np.nan,
                'trend_r2': np.nan
            }
    
    # ========================================================================
    # SEASONALITY EXTRACTION
    # ========================================================================
    
    def extract_seasonality_features(self, series: pd.Series) -> Dict[str, float]:
        """
        Extract seasonality features from time series.
        
        Seasonality represents repeating patterns at fixed intervals.
        For credit cards:
        - Monthly/quarterly spending cycles
        - Payday patterns
        - Holiday spending spikes
        
        High seasonality = predictable patterns (lower risk)
        Low seasonality = random behavior (higher risk)
        
        Args:
            series (pd.Series): Time series of a single feature
            
        Returns:
            Dict[str, float]: Dictionary with keys:
                - seasonal_strength: How much variance is explained by seasonality (0-1)
                - seasonal_amplitude: Size of seasonal swings
                
        Example:
            series = [1000, 1200, 1100, 1000, 1200, 1100, ...]  # Repeating pattern
            result = {
                'seasonal_strength': 0.85,  # 85% of variance is seasonal
                'seasonal_amplitude': 200   # Swings of $200
            }
        """
        # Need at least 2 complete cycles
        min_length = 2 * self.seasonal_period
        
        if len(series) < min_length or series.isna().all():
            return {
                'seasonal_strength': 0.0,
                'seasonal_amplitude': 0.0
            }
        
        # Remove missing values
        valid_mask = ~series.isna()
        if valid_mask.sum() < min_length:
            return {
                'seasonal_strength': 0.0,
                'seasonal_amplitude': 0.0
            }
        
        values = series[valid_mask].values
        
        try:
            # Create seasonal indices (0, 1, 2, ..., period-1, 0, 1, 2, ...)
            seasonal_indices = np.arange(len(values)) % self.seasonal_period
            
            # Calculate average value for each position in the cycle
            seasonal_means = np.array([
                values[seasonal_indices == i].mean() 
                for i in range(self.seasonal_period)
            ])
            
            # Reconstruct seasonal component for each observation
            seasonal_component = np.array([
                seasonal_means[i % self.seasonal_period] 
                for i in range(len(values))
            ])
            
            # Calculate seasonal strength
            # = variance of seasonal component / total variance
            total_var = np.var(values)
            seasonal_var = np.var(seasonal_component)
            
            seasonal_strength = seasonal_var / total_var if total_var > 0 else 0.0
            
            # Calculate seasonal amplitude
            # = difference between highest and lowest seasonal means
            seasonal_amplitude = np.max(seasonal_means) - np.min(seasonal_means)
            
            return {
                'seasonal_strength': seasonal_strength,
                'seasonal_amplitude': seasonal_amplitude
            }
        except:
            return {
                'seasonal_strength': 0.0,
                'seasonal_amplitude': 0.0
            }
    
    # ========================================================================
    # VOLATILITY EXTRACTION
    # ========================================================================
    
    def extract_volatility_features(self, series: pd.Series) -> Dict[str, float]:
        """
        Extract volatility and irregular fluctuation features.
        
        Volatility measures unpredictability and instability.
        High volatility = erratic behavior = higher default risk
        
        Args:
            series (pd.Series): Time series of a single feature
            
        Returns:
            Dict[str, float]: Dictionary with keys:
                - volatility_std: Standard deviation
                - volatility_cv: Coefficient of variation (std/mean)
                - volatility_range: Max - min
                - volatility_iqr: Interquartile range (robust to outliers)
                - num_spikes: Count of extreme values (>2 std from mean)
                
        Example:
            series = [500, 2000, 100, 1800, 300, ...]  # Erratic payments
            result = {
                'volatility_std': 750,      # High standard deviation
                'volatility_cv': 1.2,       # Very high relative to mean
                'volatility_range': 1900,   # Large swings
                'volatility_iqr': 900,      # Wide middle 50%
                'num_spikes': 3             # 3 extreme values
            }
        """
        if len(series) < 2 or series.isna().all():
            return {
                'volatility_std': np.nan,
                'volatility_cv': np.nan,
                'volatility_range': np.nan,
                'volatility_iqr': np.nan,
                'num_spikes': 0
            }
        
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return {
                'volatility_std': np.nan,
                'volatility_cv': np.nan,
                'volatility_range': np.nan,
                'volatility_iqr': np.nan,
                'num_spikes': 0
            }
        
        values = valid_series.values
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Coefficient of variation (normalized volatility)
        cv = std_val / abs(mean_val) if mean_val != 0 else np.nan
        
        # Range
        value_range = np.max(values) - np.min(values)
        
        # Interquartile range (robust measure)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        
        # Count spikes (values > 2 standard deviations from mean)
        if std_val > 0:
            z_scores = np.abs((values - mean_val) / std_val)
            num_spikes = np.sum(z_scores > 2)
        else:
            num_spikes = 0
        
        return {
            'volatility_std': std_val,
            'volatility_cv': cv,
            'volatility_range': value_range,
            'volatility_iqr': iqr,
            'num_spikes': int(num_spikes)
        }
    
    # ========================================================================
    # AUTOCORRELATION EXTRACTION
    # ========================================================================
    
    def extract_autocorrelation_features(self, series: pd.Series, 
                                        lags: List[int] = [1, 2, 3]) -> Dict[str, float]:
        """
        Extract autocorrelation features.
        
        Autocorrelation measures how much a time series depends on its past values.
        
        High positive autocorrelation (e.g., 0.8):
            - Past predicts future
            - Stable, consistent behavior
            - Lower risk
            
        Low autocorrelation (e.g., 0.1):
            - Past doesn't predict future
            - Random, unpredictable behavior
            - Higher risk
            
        Negative autocorrelation (e.g., -0.7):
            - High value → low value → high value (oscillating)
            - "Paycheck to paycheck" pattern
            - Very high risk
        
        Args:
            series (pd.Series): Time series of a single feature
            lags (List[int]): Which time lags to compute autocorrelation for
            
        Returns:
            Dict[str, float]: Dictionary with keys:
                - acf_lag1: Autocorrelation at lag 1 (most important)
                - acf_lag2: Autocorrelation at lag 2
                - acf_lag3: Autocorrelation at lag 3
                
        Example:
            series = [500, 520, 510, 505, 515, ...]  # Stable payments
            result = {
                'acf_lag1': 0.85,  # High correlation with last period
                'acf_lag2': 0.72,  # Good correlation 2 periods back
                'acf_lag3': 0.65   # Decent correlation 3 periods back
            }
        """
        max_lag = max(lags)
        
        if len(series) < max_lag + 1 or series.isna().all():
            return {f'acf_lag{lag}': np.nan for lag in lags}
        
        valid_series = series.dropna()
        if len(valid_series) < max_lag + 1:
            return {f'acf_lag{lag}': np.nan for lag in lags}
        
        try:
            # Compute autocorrelation function
            # acf_values[0] = 1.0 (correlation with itself)
            # acf_values[1] = correlation with lag 1
            # etc.
            acf_values = acf(valid_series.values, nlags=max_lag, fft=False)
            
            return {f'acf_lag{lag}': acf_values[lag] for lag in lags}
        except:
            return {f'acf_lag{lag}': np.nan for lag in lags}
    
    # ========================================================================
    # CHANGE POINT DETECTION
    # ========================================================================
    
    def extract_changepoint_features(self, series: pd.Series) -> Dict[str, float]:
        """
        Detect change points (sudden shifts in behavior).
        
        Change points indicate life events that may trigger default:
        - Job loss → sudden drop in payments
        - Medical emergency → spike in debt
        - Divorce → financial instability
        
        Args:
            series (pd.Series): Time series of a single feature
            
        Returns:
            Dict[str, float]: Dictionary with keys:
                - changepoint_detected: 1 if significant change detected, 0 otherwise
                - changepoint_magnitude: Size of the change
                
        Example:
            series = [1000, 1050, 1020, 3000, 3100, 2900, ...]  # Sudden jump
            result = {
                'changepoint_detected': 1,     # Significant change detected
                'changepoint_magnitude': 2000  # Changed by $2000
            }
        """
        if len(series) < 4 or series.isna().all():
            return {
                'changepoint_detected': 0,
                'changepoint_magnitude': 0.0
            }
        
        valid_series = series.dropna()
        if len(valid_series) < 4:
            return {
                'changepoint_detected': 0,
                'changepoint_magnitude': 0.0
            }
        
        values = valid_series.values
        
        # Split series into two halves
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        # Compare means
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        magnitude = abs(mean_second - mean_first)
        
        # Test if change is statistically significant using t-test
        try:
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            # p < 0.05 indicates significant difference
            changepoint_detected = 1 if p_value < 0.05 else 0
        except:
            changepoint_detected = 0
        
        return {
            'changepoint_detected': changepoint_detected,
            'changepoint_magnitude': magnitude
        }
    
    # ========================================================================
    # AGGREGATE ALL TIME SERIES FEATURES
    # ========================================================================
    
    def extract_all_temporal_features(self, series: pd.Series, 
                                     feature_name: str) -> Dict[str, float]:
        """
        Extract all time series features for a single variable.
        
        This is the main method that combines all time series decomposition
        techniques to create a comprehensive feature set.
        
        Args:
            series (pd.Series): Time series of values
            feature_name (str): Name of the original feature
            
        Returns:
            Dict[str, float]: Dictionary of all temporal features with prefixed names
            
        Example:
            series = customer_data['B_1']  # Balance feature
            feature_name = 'B_1'
            
            result = {
                'B_1_trend_slope': 100.0,
                'B_1_trend_r2': 0.95,
                'B_1_seasonal_strength': 0.65,
                'B_1_volatility_cv': 0.3,
                'B_1_acf_lag1': 0.85,
                ...  # 15+ features total
            }
        """
        features = {}
        
        # 1. Trend features
        trend_feats = self.extract_trend_features(series)
        for key, val in trend_feats.items():
            features[f'{feature_name}_{key}'] = val
        
        # 2. Seasonality features
        seasonal_feats = self.extract_seasonality_features(series)
        for key, val in seasonal_feats.items():
            features[f'{feature_name}_{key}'] = val
        
        # 3. Volatility features
        volatility_feats = self.extract_volatility_features(series)
        for key, val in volatility_feats.items():
            features[f'{feature_name}_{key}'] = val
        
        # 4. Autocorrelation features
        acf_feats = self.extract_autocorrelation_features(series, lags=[1, 2, 3])
        for key, val in acf_feats.items():
            features[f'{feature_name}_{key}'] = val
        
        # 5. Change point features
        cp_feats = self.extract_changepoint_features(series)
        for key, val in cp_feats.items():
            features[f'{feature_name}_{key}'] = val
        
        return features
    
    # ========================================================================
    # MAIN AGGREGATION METHOD
    # ========================================================================
    
    def aggregate_customer_timeseries(self, customer_group: pd.DataFrame,
                                     numerical_features: List[str],
                                     categorical_features: List[str]) -> Dict:
        """
        Aggregate a single customer's data using time series decomposition.
        
        Args:
            customer_group (pd.DataFrame): All statements for one customer
            numerical_features (List[str]): List of numerical feature names
            categorical_features (List[str]): List of categorical feature names
            
        Returns:
            Dict: Dictionary of aggregated features
        """
        agg_dict = {}
        
        # Customer identifier
        agg_dict['customer_ID'] = customer_group['customer_ID'].iloc[0]
        
        # Target variable
        if 'target' in customer_group.columns:
            agg_dict['target'] = customer_group['target'].iloc[0]
        
        # Meta feature
        agg_dict['num_statements'] = len(customer_group)
        
        # Process numerical features with time series decomposition
        for col in numerical_features:
            if col not in customer_group.columns:
                continue
            
            series = customer_group[col]
            
            # Basic aggregations (for comparison)
            agg_dict[f'{col}_last'] = series.iloc[-1] if len(series) > 0 else np.nan
            agg_dict[f'{col}_mean'] = series.mean()
            agg_dict[f'{col}_std'] = series.std()
            agg_dict[f'{col}_min'] = series.min()
            agg_dict[f'{col}_max'] = series.max()
            
            # Time series decomposition features
            ts_features = self.extract_all_temporal_features(series, col)
            agg_dict.update(ts_features)
        
        # Process categorical features (simple aggregation only)
        for col in categorical_features:
            if col not in customer_group.columns:
                continue
            
            series = customer_group[col]
            agg_dict[f'{col}_last'] = series.iloc[-1] if len(series) > 0 else np.nan
            mode_values = series.mode()
            agg_dict[f'{col}_mode'] = mode_values.iloc[0] if len(mode_values) > 0 else np.nan
        
        return agg_dict
    
    # ========================================================================
    # FIT-TRANSFORM METHOD
    # ========================================================================
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data (pd.DataFrame): Raw temporal data with multiple rows per customer
            
        Returns:
            pd.DataFrame: Aggregated data with comprehensive temporal features
        """
        print("="*80)
        print("TIME SERIES DECOMPOSITION PREPROCESSING")
        print("="*80)
        
        # Validate input
        required_cols = ['customer_ID']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert and sort by date
        if 'S_2' in data.columns:
            print("\nConverting S_2 to datetime...")
            data['S_2'] = pd.to_datetime(data['S_2'], errors='coerce')
            print("Sorting data by customer_ID and date...")
            data = data.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)
        else:
            print("\nWarning: No S_2 (date) column found.")
            data = data.sort_values('customer_ID').reset_index(drop=True)
        
        # Identify feature types
        print("\nIdentifying feature types...")
        numerical, categorical = self.identify_feature_types(data)
        self.numerical_features = numerical
        self.categorical_features = categorical
        
        # Data statistics
        print(f"\nData Statistics:")
        print(f"  Total rows: {len(data):,}")
        print(f"  Unique customers: {data['customer_ID'].nunique():,}")
        print(f"  Avg statements per customer: {len(data) / data['customer_ID'].nunique():.1f}")
        if 'target' in data.columns:
            default_rate = data.groupby('customer_ID')['target'].first().mean()
            print(f"  Default rate: {default_rate:.2%}")
        
        # Aggregate with time series features
        print(f"\nExtracting time series features for {data['customer_ID'].nunique():,} customers...")
        print("This will take longer than simple aggregation (5-15 minutes for large datasets)...")
        print(f"Seasonal period: {self.seasonal_period}")
        
        aggregated_list = []
        
        for i, (customer_id, group) in enumerate(data.groupby('customer_ID')):
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1:,} customers...")
            
            agg_dict = self.aggregate_customer_timeseries(
                group,
                self.numerical_features,
                self.categorical_features
            )
            aggregated_list.append(agg_dict)
        
        agg_df = pd.DataFrame(aggregated_list)
        
        # Summary
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Input shape: {data.shape}")
        print(f"Output shape: {agg_df.shape}")
        print(f"Features created: {agg_df.shape[1] - 2}")
        
        # Count feature types
        feature_types = {
            'basic': len([c for c in agg_df.columns if any(x in c for x in ['_last', '_mean', '_std', '_min', '_max'])]),
            'trend': len([c for c in agg_df.columns if 'trend' in c]),
            'seasonal': len([c for c in agg_df.columns if 'seasonal' in c]),
            'volatility': len([c for c in agg_df.columns if 'volatility' in c or 'spike' in c]),
            'autocorrelation': len([c for c in agg_df.columns if 'acf' in c]),
            'changepoint': len([c for c in agg_df.columns if 'changepoint' in c])
        }
        
        print("\nFeature Breakdown:")
        for feat_type, count in feature_types.items():
            print(f"  {feat_type.capitalize()}: {count} features")
        
        missing_pct = (agg_df.isnull().sum().sum() / (agg_df.shape[0] * agg_df.shape[1])) * 100
        print(f"\nMissing values: {missing_pct:.2f}% of total cells")
        
        return agg_df
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature types.
        """
        if self.numerical_features is None or self.categorical_features is None:
            raise ValueError("Must call fit_transform before transform")
        
        print("Transforming new data with time series features...")
        
        if 'S_2' in data.columns:
            data['S_2'] = pd.to_datetime(data['S_2'], errors='coerce')
            data = data.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)
        else:
            data = data.sort_values('customer_ID').reset_index(drop=True)
        
        aggregated_list = []
        for customer_id, group in data.groupby('customer_ID'):
            agg_dict = self.aggregate_customer_timeseries(
                group,
                self.numerical_features,
                self.categorical_features
            )
            aggregated_list.append(agg_dict)
        
        agg_df = pd.DataFrame(aggregated_list)
        print(f"Transformed shape: {agg_df.shape}")
        
        return agg_df


# ================================================================================
# USAGE EXAMPLE
# ================================================================================

def main():
    """
    Main function demonstrating how to use the TimeSeriesPreprocessor.
    """
    
    # Load data
    print("Loading data...")
    data = pd.read_csv('train_data.csv')
    
    print(f"Loaded {len(data):,} rows")
    print(f"Columns: {len(data.columns)}")
    
    # Initialize preprocessor
    preprocessor = TimeSeriesPreprocessor(seasonal_period=3)
    
    # Fit and transform
    aggregated_data = preprocessor.fit_transform(data)
    
    # Save results
    output_file = 'train_data_aggregated_timeseries.csv'
    aggregated_data.to_csv(output_file, index=False)
    print(f"\nAggregated data saved to: {output_file}")
    
    # Display sample results
    print(f"\nSample of aggregated data:")
    print(aggregated_data.head())
    
    return aggregated_data


if __name__ == "__main__":
    aggregated_data = main()