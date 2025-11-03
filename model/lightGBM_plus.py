import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, make_scorer

import logging
import pickle
import lightgbm as lgb
import os
import gc
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)

# Feature file paths (update these to your actual paths)
FEATURE_FILES = {
    'basic_stats': './features/basic_agg_feats.feather',
    'advanced_stats': './features/advanced_agg_feats.feather',
    'diff_feats': './features/diff_feats.feather',
    'time_series_feats': './features/time_series_feats.feather',
    'lag_feats': './features/lag_feats.feather',
    'rank_feats': './features/rank_feats.feather',
    'count_feats': './features/count_feats.feather',
    'category_agg_feats': './features/category_agg_feats.feather',
    'trend_feats': './features/trend_feats.feather',
}

LABEL_FILE = './input/train_labels.csv'
TEST_FEATURE_FILES = {
    'basic_stats': './features/test_basic_agg_feats.feather',
    'advanced_stats': './features/test_advanced_agg_feats.feather',
    'diff_feats': './features/test_diff_feats.feather',
    'time_series_feats': './features/test_time_series_feats.feather',
    'lag_feats': './features/test_lag_feats.feather',
    'rank_feats': './features/test_rank_feats.feather',
    'count_feats': './features/test_count_feats.feather',
    'category_agg_feats': './features/test_category_agg_feats.feather',
    'trend_feats': './features/test_trend_feats.feather',
}

OUTPUT_DIR = './ensemble_models/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# AMEX METRIC (from Kaggle)
# ============================================================================

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Original AMEX metric implementation"""
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)

def amex_metric_sklearn(y_true, y_pred_proba):
    """Sklearn-compatible wrapper"""
    y_true_df = pd.DataFrame({'target': np.array(y_true).ravel()}).reset_index(drop=True)
    y_pred_df = pd.DataFrame({'prediction': np.array(y_pred_proba).ravel()}).reset_index(drop=True)
    return amex_metric(y_true_df, y_pred_df)

# ============================================================================
# FEATURE LOADING & MERGING
# ============================================================================

def load_feature_set(feature_keys: list, is_train: bool = True) -> pd.DataFrame:
    """
    Load and merge multiple feature files.
    
    Args:
        feature_keys: List of keys from FEATURE_FILES to load (e.g., ['basic_stats', 'diff_feats'])
        is_train: If True, load train features; otherwise load test features
    
    Returns:
        Merged DataFrame with customer_ID as index
    """
    feature_dict = FEATURE_FILES if is_train else TEST_FEATURE_FILES
    
    print(f"\n{'='*60}")
    print(f"Loading {'TRAIN' if is_train else 'TEST'} features: {feature_keys}")
    print(f"{'='*60}")
    
    dfs = []
    for key in feature_keys:
        if key not in feature_dict:
            print(f"WARNING: Feature key '{key}' not found!")
            continue
            
        file_path = feature_dict[key]
        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
            continue
        
        print(f"Loading {key} from {file_path}...")
        df = pd.read_feather(file_path)
        print(f"  Shape: {df.shape}, Columns: {df.columns.tolist()[:5]}...")
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No feature files loaded!")
    
    # Merge on customer_ID
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='customer_ID', how='inner', suffixes=('', '_dup'))
        # Drop duplicate columns (can happen if same feature in multiple files)
        dup_cols = [c for c in merged.columns if c.endswith('_dup')]
        if dup_cols:
            print(f"  Dropping {len(dup_cols)} duplicate columns")
            merged.drop(columns=dup_cols, inplace=True)
    
    print(f"\nMerged shape: {merged.shape}")
    return merged

def prepare_data(feature_keys: list, holdout_size: float = 0.2):
    """
    Load features and labels, split into train/validation
    
    Returns:
        X_train, y_train, X_val, y_val, feature_names
    """
    # Load features
    X = load_feature_set(feature_keys, is_train=True)
    
    # Load labels
    print(f"\nLoading labels from {LABEL_FILE}...")
    labels = pd.read_csv(LABEL_FILE)
    
    # Merge
    data = X.merge(labels, on='customer_ID', how='inner')
    print(f"After label merge: {data.shape}")
    
    # Separate features and target
    y = data['target'].astype('int8')
    X = data.drop(columns=['customer_ID', 'target'])
    
    # Handle categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"\nEncoding {len(cat_cols)} categorical columns...")
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Remove constant features
    print("\nRemoving constant features...")
    nunique = X.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols:
        print(f"  Dropping {len(constant_cols)} constant features")
        X.drop(columns=constant_cols, inplace=True)
    
    # Remove high-missing features (>95% missing)
    print("\nRemoving high-missing features...")
    missing_pct = X.isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.95].index.tolist()
    if high_missing_cols:
        print(f"  Dropping {len(high_missing_cols)} high-missing features")
        X.drop(columns=high_missing_cols, inplace=True)
    
    # Replace inf with nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"\nFinal feature count: {X.shape[1]}")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=holdout_size, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    
    return X_train, y_train, X_val, y_val, X.columns.tolist()

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_single_model(
    X_train, y_train, X_val, y_val,
    model_name: str,
    lgb_params: dict = None,
    use_dart: bool = False
):
    """
    Train a single LightGBM model
    
    Returns:
        model, val_score (AMEX metric), val_auc
    """
    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"{'='*60}")
    
    # Default params (similar to your original script)
    if lgb_params is None:
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'dart' if use_dart else 'gbdt',
            'max_depth': -1,
            'num_leaves': 64,
            'learning_rate': 0.035 if use_dart else 0.05,
            'bagging_freq': 5,
            'bagging_fraction': 0.75,
            'feature_fraction': 0.75,
            'min_data_in_leaf': 256,
            'max_bin': 63,
            'min_data_in_bin': 256,
            'lambda_l1': 0.1,
            'lambda_l2': 30,
            'num_threads': -1,
            'verbosity': -1,
            'seed': RANDOM_SEED,
        }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    evals_result = {}
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=4500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    # Predict on validation
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_val_pred)
    val_amex = amex_metric_sklearn(y_val, y_val_pred)
    
    print(f"\nValidation AUC: {val_auc:.6f}")
    print(f"Validation AMEX: {val_amex:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    
    return model, val_amex, val_auc

# ============================================================================
# ENSEMBLE TRAINING
# ============================================================================

def train_ensemble_models(feature_combinations: dict, holdout_size: float = 0.2):
    """
    Train multiple models on different feature combinations
    
    Args:
        feature_combinations: Dict of {model_name: [list of feature keys]}
        
    Example:
        feature_combinations = {
            'model_basic': ['basic_stats'],
            'model_advanced': ['basic_stats', 'advanced_stats'],
            'model_all_stats': ['basic_stats', 'advanced_stats', 'diff_feats', 'time_series_feats'],
            'model_full': list(FEATURE_FILES.keys()),
        }
    """
    results = []
    
    for model_name, feature_keys in feature_combinations.items():
        print(f"\n\n{'#'*80}")
        print(f"# {model_name.upper()}")
        print(f"# Features: {feature_keys}")
        print(f"{'#'*80}\n")
        
        try:
            # Load data
            X_train, y_train, X_val, y_val, feature_names = prepare_data(
                feature_keys, holdout_size
            )
            
            # Train model
            model, val_amex, val_auc = train_single_model(
                X_train, y_train, X_val, y_val,
                model_name,
                use_dart=True  # Use DART boosting like original script
            )
            
            # Save model
            model_path = os.path.join(OUTPUT_DIR, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")
            
            # Save feature names
            feature_path = os.path.join(OUTPUT_DIR, f'{model_name}_features.json')
            with open(feature_path, 'w') as f:
                json.dump({
                    'feature_keys': feature_keys,
                    'feature_names': feature_names,
                }, f, indent=2)
            
            # Record results
            results.append({
                'model_name': model_name,
                'feature_keys': feature_keys,
                'n_features': len(feature_names),
                'val_amex': val_amex,
                'val_auc': val_auc,
                'model_path': model_path,
            })
            
            # Clean up
            del X_train, y_train, X_val, y_val, model
            gc.collect()
            
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_amex', ascending=False)
    
    results_path = os.path.join(OUTPUT_DIR, 'ensemble_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n\n{'='*80}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to {results_path}")
    print(f"\nTop models by AMEX metric:")
    print(results_df[['model_name', 'n_features', 'val_amex', 'val_auc']].to_string(index=False))
    
    return results_df

# ============================================================================
# PREDICTION & ENSEMBLE
# ============================================================================

def predict_ensemble(models_to_use: list, weights: list = None, top_n: int = None):
    """
    Generate ensemble predictions on test set
    
    Args:
        models_to_use: List of model names (e.g., ['model_basic', 'model_advanced'])
        weights: List of weights for each model (if None, use equal weights)
        top_n: If specified, automatically use top N models by validation AMEX score
    """
    # Load results
    results_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'ensemble_results.csv'))
    
    if top_n is not None:
        results_df = results_df.sort_values('val_amex', ascending=False).head(top_n)
        models_to_use = results_df['model_name'].tolist()
        print(f"Using top {top_n} models: {models_to_use}")
    
    if weights is None:
        weights = [1.0] * len(models_to_use)
    weights = np.array(weights) / np.sum(weights)  # Normalize
    
    print(f"\nEnsemble configuration:")
    for model_name, weight in zip(models_to_use, weights):
        val_amex = results_df[results_df['model_name'] == model_name]['val_amex'].values[0]
        print(f"  {model_name}: weight={weight:.4f}, val_amex={val_amex:.6f}")
    
    # Collect predictions
    all_preds = []
    customer_ids = None
    
    for model_name in models_to_use:
        print(f"\nGenerating predictions for {model_name}...")
        
        # Load feature config
        feature_path = os.path.join(OUTPUT_DIR, f'{model_name}_features.json')
        with open(feature_path, 'r') as f:
            config = json.load(f)
        feature_keys = config['feature_keys']
        
        # Load test features
        X_test = load_feature_set(feature_keys, is_train=False)
        
        if customer_ids is None:
            customer_ids = X_test['customer_ID'].values
        
        X_test = X_test.drop(columns=['customer_ID'])
        
        # Handle categoricals (simple approach: label encode)
        cat_cols = X_test.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))
        
        # Replace inf
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Load model
        model_path = os.path.join(OUTPUT_DIR, f'{model_name}.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Predict
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        all_preds.append(preds)
        
        del X_test, model
        gc.collect()
    
    # Weighted average
    ensemble_preds = np.average(all_preds, axis=0, weights=weights)
    
    # Save submission
    submission = pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction': ensemble_preds
    })
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = os.path.join(OUTPUT_DIR, f'submission_ensemble_{timestamp}.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"\nEnsemble submission saved to {submission_path}")
    print(f"Prediction stats:")
    print(f"  Mean: {ensemble_preds.mean():.6f}")
    print(f"  Std: {ensemble_preds.std():.6f}")
    print(f"  Min: {ensemble_preds.min():.6f}")
    print(f"  Max: {ensemble_preds.max():.6f}")
    
    return submission

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # STEP 1: DEFINE FEATURE COMBINATIONS
    # ========================================================================
    
    feature_combinations = {
        # Individual feature sets (for comparison)
        'model_basic_only': ['basic_stats'],
        'model_advanced_only': ['advanced_stats'],
        'model_diff_only': ['diff_feats'],
        'model_timeseries_only': ['time_series_feats'],
        
        # Combined feature sets (strategic combinations)
        'model_stats_combo': ['basic_stats', 'advanced_stats'],
        'model_temporal_combo': ['diff_feats', 'time_series_feats', 'lag_feats', 'trend_feats'],
        'model_ranking_combo': ['rank_feats', 'count_feats'],
        
        # Progressive builds
        'model_small': ['basic_stats', 'diff_feats'],
        'model_medium': ['basic_stats', 'advanced_stats', 'diff_feats', 'time_series_feats'],
        'model_large': ['basic_stats', 'advanced_stats', 'diff_feats', 'time_series_feats', 
                        'lag_feats', 'rank_feats'],
        
        # Full model (all features)
        'model_full': list(FEATURE_FILES.keys()),
    }
    
    # ========================================================================
    # STEP 2: TRAIN ENSEMBLE
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: TRAINING ENSEMBLE MODELS")
    print("="*80)
    
    results_df = train_ensemble_models(
        feature_combinations,
        holdout_size=0.2
    )
    
    # ========================================================================
    # STEP 3: GENERATE ENSEMBLE PREDICTIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: GENERATING ENSEMBLE PREDICTIONS")
    print("="*80)
    
    # Option 1: Use top 3 models (automatic selection)
    submission_top3 = predict_ensemble(models_to_use=None, top_n=3)
    
    # Option 2: Manually specify models with custom weights
    # submission_custom = predict_ensemble(
    #     models_to_use=['model_full', 'model_large', 'model_temporal_combo'],
    #     weights=[0.5, 0.3, 0.2]  # Higher weight to full model
    # )
    
    print("\n" + "="*80)
    print("ENSEMBLE PIPELINE COMPLETE!")
    print("="*80)