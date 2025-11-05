import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    make_scorer
)

import logging
import pickle
import lightgbm as lgb
import os
import gc

# Use RandomizedSearchCV with this distribution
from scipy.stats import uniform, randint, loguniform

"""
Light Gradient Boosting Machine (LightGBM) model training and evaluation.

LightBGM is a gradient boosting framework that uses tree-based learning algorithms.
LightGBM builds an ensemble of decision trees in a sequential manner, where each new tree 
aims to correct the errors made by the previous trees.

Gradient boosting is an ensemble learning technique that combines the predictions of multiple weak learners,
which optimizes a loss function by adding models (eg trees) that minimize the residual errors of prior models
known as gradient descent.

In contract, Random Forest builds multiple decision trees independently and simultaenously (parallel) 
and aggregates their predictions to improve accuracy and reduce overfitting.
"""


# Set up two dedicated loggers: one for training and one for evaluation/test
# Both will stream to console and write to separate files so logs are easier to
# inspect independently.
_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
_STREAM_HANDLER = logging.StreamHandler()
_STREAM_HANDLER.setFormatter(logging.Formatter(_LOG_FORMAT))



# Set random seed for reproducibility
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)


def _load_and_merge_features(
        feature_files: dict,
        label_file: str,
        train_logger: logging.Logger,
        is_train: bool = True
    ) -> pd.DataFrame:
    """
    Load and merge multiple feather files by customer_ID.
    
    Args:
        feature_files: Dictionary mapping feature set names to file paths
        label_file: Path to labels CSV (only used for training)
        train_logger: Logger instance
        is_train: Whether this is training data (requires labels)
    
    Returns:
        Merged DataFrame with all features
    """
    train_logger.info(f"Loading and merging {len(feature_files)} feature files...")
    
    # Load first feature file as base
    first_key = list(feature_files.keys())[0]
    train_logger.info(f"Loading base feature set: {first_key}")
    data = pd.read_feather(feature_files[first_key])
    train_logger.info(f"Base shape: {data.shape}")
    
    # Merge remaining feature files
    for feature_name, file_path in list(feature_files.items())[1:]:
        train_logger.info(f"Loading and merging: {feature_name}")
        feature_df = pd.read_feather(file_path)
        train_logger.info(f"Feature shape: {feature_df.shape}")
        
        # Merge on customer_ID
        data = data.merge(
            feature_df,
            on='customer_ID',
            how='inner',
            suffixes=('', f'_{feature_name}')
        )
        train_logger.info(f"Shape after merge: {data.shape}")
        
        del feature_df
        gc.collect()
    
    # For training data, merge labels
    if is_train:
        train_logger.info(f"Loading labels from {label_file}...")
        labels = pd.read_csv(label_file)
        labels['target'] = labels['target'].astype('int8')
        train_logger.info(f"Labels loaded: {labels.shape}")
        
        # Merge labels
        data = data.merge(
            labels[['customer_ID', 'target']],
            on='customer_ID',
            how='inner'
        )
        train_logger.info(f"Shape after merging labels: {data.shape}")
        
        del labels
        gc.collect()
    
    train_logger.info(f"Final merged data shape: {data.shape}")
    return data


def _preprocess_data(
        data: pd.DataFrame,
        max_missing_rate: float,
        train_logger: logging.Logger,
        removed_features_path: str = None,
        is_train: bool = True
    ) -> tuple[pd.DataFrame, list]:
    """
    Preprocess merged feature data.
    
    Args:
        data: Merged feature DataFrame
        max_missing_rate: Maximum allowed missing rate per feature
        train_logger: Logger instance
        removed_features_path: Path to save/load removed features
        is_train: Whether this is training data
    
    Returns:
        Preprocessed data and list of removed features
    """
    train_logger.info("Starting data preprocessing...")
    
    # Get feature columns (exclude customer_ID and target)
    exclude_cols = {'customer_ID', 'target'}
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    train_logger.info(f"Total features before preprocessing: {len(feature_cols)}")
    
    removed_features = []
    
    if is_train:
        # Training mode: compute and save removed features
        
        # 1) Replace ±Inf with NaN
        train_logger.info("Replacing ±Inf with NaN...")
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        
        # 2) Check for high missing rate
        train_logger.info(f"Checking for features with >{max_missing_rate*100:.1f}% missing values...")
        missing_rate = data[feature_cols].isnull().sum() / len(data)
        high_missing_cols = missing_rate[missing_rate > max_missing_rate].index.tolist()
        
        if high_missing_cols:
            train_logger.info(f"Removing {len(high_missing_cols)} features with high missing rate")
            removed_features.extend(high_missing_cols)
        
        # 3) Check for constant features
        train_logger.info("Checking for constant features...")
        constant_features = []
        for col in feature_cols:
            if col not in removed_features:
                series = data[col]
                if series.notna().any():
                    vmin = series.min(skipna=True)
                    vmax = series.max(skipna=True)
                    if vmin == vmax:
                        constant_features.append(col)
        
        if constant_features:
            train_logger.info(f"Removing {len(constant_features)} constant features")
            removed_features.extend(constant_features)
        
        # 4) Check for all-NaN columns
        train_logger.info("Checking for all-NaN columns...")
        all_nan_cols = [col for col in feature_cols if data[col].isna().all()]
        
        if all_nan_cols:
            train_logger.info(f"Removing {len(all_nan_cols)} all-NaN columns")
            removed_features.extend(all_nan_cols)
        
        # Remove duplicates while preserving order
        removed_features = list(dict.fromkeys(removed_features))
        
        # Save removed features
        if removed_features_path:
            with open(removed_features_path, 'wb') as f:
                pickle.dump(removed_features, f)
            train_logger.info(f"Saved {len(removed_features)} removed feature names to {removed_features_path}")
        
    else:
        # Test mode: load and apply removed features
        if removed_features_path and os.path.exists(removed_features_path):
            with open(removed_features_path, 'rb') as f:
                removed_features = pickle.load(f)
            train_logger.info(f"Loaded {len(removed_features)} removed features from {removed_features_path}")
        else:
            train_logger.warning(f"Removed features file not found at {removed_features_path}")
            removed_features = []
    
    # Drop removed features
    cols_to_drop = [col for col in removed_features if col in data.columns]
    if cols_to_drop:
        train_logger.info(f"Dropping {len(cols_to_drop)} features")
        data = data.drop(columns=cols_to_drop)
    
    # Final feature count
    final_feature_cols = [col for col in data.columns if col not in exclude_cols]
    train_logger.info(f"Features after preprocessing: {len(final_feature_cols)}")
    
    # NaN statistics
    if is_train:
        nan_counts = data[final_feature_cols].isnull().sum()
        total_nans = int(nan_counts.sum())
        cols_with_nans = int((nan_counts > 0).sum())
        
        if total_nans > 0:
            train_logger.info(f"NaN statistics: {total_nans} NaN values across {cols_with_nans} columns")
            top_nan = nan_counts.sort_values(ascending=False).head(10)
            train_logger.info(f"Top columns with NaN:\n{top_nan}")
    
    return data, removed_features


def _amex_metric(
        y_true: pd.DataFrame, 
        y_pred: pd.DataFrame
    ) -> float:
    """
    Original AMEX metric implementation.
    https://www.kaggle.com/code/inversion/amex-competition-metric-python
    """

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


def _amex_metric_sklearn(y_true, y_pred_proba):
    """
    Wrapper for sklearn compatibility (GridSearchCV scorer).
    
    Args:
        y_true: array-like of true labels (0 or 1)
        y_pred_proba: array-like of predicted probabilities (0 to 1)
    
    Returns:
        AMEX metric score
    """
    # Convert to DataFrame format that amex_metric expects
    # CRITICAL: Reset index to avoid alignment issues
    y_true_df = pd.DataFrame({
        'target': np.array(y_true).ravel()
    }).reset_index(drop=True)
    
    y_pred_df = pd.DataFrame({
        'prediction': np.array(y_pred_proba).ravel()
    }).reset_index(drop=True)
    
    return _amex_metric(y_true_df, y_pred_df)


def lightgbm_train(
        feature_files           : dict                                          ,
        label_file              : str                                           ,
        output_dir              : str                                           ,
        
        num_iter                : int   = 10                                    ,
        param_grid              : dict  = None                                  ,
        cv_folds                : int   = 5                                     ,
        holdout_size            : float = 0.2                                   ,
        hyperparam_sample_frac  : float = 0.15                                  ,  # NEW PARAMETER
        max_missing_rate        : float = 0.95                                  ,
        top_n_param             : int   = 5                                     ,
        top_n_features          : int   = 20                                    ,
    ):
    """
    Train LightGBM model with hyperparameter tuning.
    
    Args:
        feature_files: Dictionary mapping feature set names to file paths
        label_file: Path to labels CSV
        output_dir: Directory to save all output files
        num_iter: Number of iterations for RandomizedSearchCV
        param_grid: Hyperparameter grid for tuning
        cv_folds: Number of CV folds
        holdout_size: Fraction of data to hold out for validation
        hyperparam_sample_frac: Fraction of training data to use for hyperparameter tuning (0 < x <= 1.0)
        max_missing_rate: Maximum allowed missing rate per feature
        top_n_param: Top N parameter combinations to log
        top_n_features: Top N features to plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    model_output_path       = os.path.join(output_dir, 'lightgbm_model.pkl')
    removed_features_path   = os.path.join(output_dir, 'removed_features.pkl')
    cv_results_path         = os.path.join(output_dir, 'lightgbm_grid_search_cv_results.csv')
    confusion_matrix_path   = os.path.join(output_dir, 'confusion_matrix_lgb.png')
    feature_importance_path = os.path.join(output_dir, 'feature_importance_lgb.png')
    train_log_path          = os.path.join(output_dir, 'lightgbm_train.log')
    
    # Setup logger
    _FILE_HANDLER_TRAIN = logging.FileHandler(train_log_path, mode='w')
    _FILE_HANDLER_TRAIN.setFormatter(logging.Formatter(_LOG_FORMAT))

    train_logger = logging.getLogger(f'lightgbm.train.{output_dir}')
    train_logger.setLevel(logging.INFO)
    train_logger.handlers.clear()
    train_logger.addHandler(_STREAM_HANDLER)
    train_logger.addHandler(_FILE_HANDLER_TRAIN)
    
    train_logger.info("="*80)
    train_logger.info(f"Training model with output directory: {output_dir}")
    train_logger.info("="*80)
    
    # Load and merge features
    data = _load_and_merge_features(
        feature_files,
        label_file,
        train_logger,
        is_train=True
    )
    
    # Preprocess data
    data, removed_features = _preprocess_data(
        data,
        max_missing_rate,
        train_logger,
        removed_features_path,
        is_train=True
    )
    
    # Split features and target
    train_logger.info("Splitting features and target...")
    y = data['target'].astype('int8', copy=False)
    X = data.drop(columns=['customer_ID', 'target'])
    del data
    gc.collect()
    
    train_logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Verify all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        train_logger.error(f"ERROR: Non-numeric columns found: {non_numeric.tolist()}")
        raise ValueError("Non-numeric columns found")
    
    # Train/validation split
    train_logger.info("Splitting into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    
    del X, y
    gc.collect()
    
    train_logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    train_logger.info(f"Training labels:\n{y_train.value_counts(normalize=True)}")
    train_logger.info(f"Validation labels:\n{y_val.value_counts(normalize=True)}")
    
    # ========== NEW: Sample data for hyperparameter tuning ==========
    if hyperparam_sample_frac < 1.0:
        train_logger.info("="*80)
        train_logger.info(f"Sampling {hyperparam_sample_frac*100:.1f}% of training data for hyperparameter tuning...")
        
        sample_size = int(len(X_train) * hyperparam_sample_frac)
        
        # Stratified sampling to maintain class distribution
        X_hp_tune, X_hp_holdout, y_hp_tune, y_hp_holdout = train_test_split(
            X_train,
            y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=RANDOM_SEED
        )
        
        train_logger.info(f"Hyperparameter tuning sample: {len(X_hp_tune):,} rows ({hyperparam_sample_frac*100:.1f}%)")
        train_logger.info(f"Held out from tuning: {len(X_hp_holdout):,} rows ({(1-hyperparam_sample_frac)*100:.1f}%)")
        train_logger.info(f"Full training data: {len(X_train):,} rows (100%)")
        train_logger.info(f"HP tuning sample labels:\n{y_hp_tune.value_counts(normalize=True)}")
        train_logger.info("="*80)
        
        # Use sample for hyperparameter tuning
        X_for_tuning = X_hp_tune
        y_for_tuning = y_hp_tune
        
        # Clean up
        del X_hp_holdout, y_hp_holdout
        gc.collect()
    else:
        train_logger.info("Using full training data for hyperparameter tuning (hyperparam_sample_frac=1.0)")
        X_for_tuning = X_train
        y_for_tuning = y_train
    # ================================================================
    
    # Create LightGBM classifier
    train_logger.info("Creating LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_SEED,
        force_col_wise=True,
    )
    
    # Hyperparameter tuning
    train_logger.info("Starting hyperparameter tuning...")
    train_logger.info("Scoring metric: ROC-AUC (proxy for AMEX metric)")
    
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED
    )
    
    grid_search = RandomizedSearchCV(
        lgb_model,
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        return_train_score=True,
        n_iter=num_iter,
        n_jobs=1,
        pre_dispatch=1,
        random_state=RANDOM_SEED,
    )
    
    # Fit on sampled data (or full data if hyperparam_sample_frac=1.0)
    grid_search.fit(X_for_tuning, y_for_tuning)
    
    train_logger.info("Hyperparameter tuning completed.")
    
    # Clean up tuning data
    if hyperparam_sample_frac < 1.0:
        del X_for_tuning, y_for_tuning, X_hp_tune, y_hp_tune
        gc.collect()
    
    # Log CV results
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_
    n_candidates = len(cv_results['params'])
    total_fits = n_candidates * cv_folds
    
    mean_fit_time = cv_results['mean_fit_time'][best_idx]
    mean_score_time = cv_results['mean_score_time'][best_idx]
    
    total_fit_time_seconds = (cv_results['mean_fit_time'] * cv_folds).sum()
    total_fit_time_minutes = total_fit_time_seconds / 60.0
    
    refit_time = getattr(grid_search, "refit_time_", None)
    
    train_logger.info(f"Evaluated {n_candidates} hyperparameter combinations across {cv_folds} folds "
                      f"(total fits: {total_fits})")
    train_logger.info(f"Best params: {grid_search.best_params_} (candidate index: {best_idx})")
    train_logger.info(f"     Mean ROC-AUC score (across {cv_folds} folds): {grid_search.best_score_:.6f} "
                      f"(std: {cv_results['std_test_score'][best_idx]:.6f})")
    
    split_scores = [cv_results[f'split{i}_test_score'][best_idx] for i in range(cv_folds)]
    train_logger.info(f"     Best candidate scores per fold: {[f'{s:.6f}' for s in split_scores]}")
    train_logger.info(f"     Min: {min(split_scores):.6f}, Max: {max(split_scores):.6f}, "
                      f"Range: {max(split_scores) - min(split_scores):.6f}")
    
    train_logger.info(f"     Average fit time per fold: {mean_fit_time:.3f}s")
    train_logger.info(f"     Average score time per fold: {mean_score_time:.3f}s")
    if refit_time is not None:
        train_logger.info(f"     Refit time on full training set: {refit_time:.3f}s")
    
    train_logger.info(f"Estimated total time spent fitting (all candidates, all folds): "
                      f"{total_fit_time_seconds:.2f}s ({total_fit_time_minutes:.2f} min)")
    
    # Log top N candidates
    try:
        top_indices = np.argsort(cv_results['mean_test_score'])[::-1][:top_n_param]
        train_logger.info(f"Top {len(top_indices)} hyperparameter candidates (by mean CV score):")
        
        for rank, idx in enumerate(top_indices, start=1):
            mean_score = cv_results['mean_test_score'][idx]
            std_score = cv_results['std_test_score'][idx]
            fit_time = cv_results['mean_fit_time'][idx]
            params = cv_results['params'][idx]
            
            train_logger.info(f"     Rank {rank}: mean_score={mean_score:.6f} std={std_score:.6f} "
                              f"avg_fit_time_per_fold={fit_time:.3f}s params={params}")
    except Exception as e:
        train_logger.warning(f"Could not extract top candidates from cv_results_: {e}")
    
    # Save CV results
    try:
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(cv_results_path, index=False)
        train_logger.info(f"Grid search CV results saved to: {cv_results_path}")
    except Exception as e:
        train_logger.warning(f"Failed to save cv_results_ to CSV: {e}")
    
    # ========== NEW: Train final model on FULL training data ==========
    train_logger.info("="*80)
    train_logger.info("Training final model on FULL training data with best hyperparameters...")
    train_logger.info(f"Best hyperparameters: {grid_search.best_params_}")
    
    final_model = lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_SEED,
        force_col_wise=True,
        **grid_search.best_params_
    )
    
    final_model.fit(X_train, y_train)
    
    n_trees = final_model.n_estimators
    train_logger.info(f"Final model trained with {n_trees} trees on {len(X_train):,} samples")
    train_logger.info("="*80)
    # ==================================================================
    
    # Evaluate on validation set (using final model trained on full data)
    train_logger.info("Evaluating final model on holdout validation set...")
    
    y_pred_proba = final_model.predict_proba(X_val)[:, 1]
    y_pred = final_model.predict(X_val)
    
    val_auc = roc_auc_score(y_val, y_pred_proba)
    train_logger.info(f"Validation ROC-AUC: {val_auc:.4f}")
    
    amex_score = _amex_metric_sklearn(y_val, y_pred_proba)
    train_logger.info(f"Validation AMEX metric: {amex_score:.6f}")
    train_logger.info(f"CV to Validation ROC-AUC difference: {val_auc - grid_search.best_score_:.6f}")
    train_logger.info("Note: CV score was computed on sample data; validation score uses model trained on full data.")
    
    train_logger.info("\nClassification Report:")
    train_logger.info("\n" + classification_report(y_val, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Holdout Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    train_logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
    plt.close()
    
    # Feature Importance (from final model)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    train_logger.info(f"Top {top_n_features} Feature Importances:")
    for idx, row in feature_importance.head(top_n_features).iterrows():
        train_logger.info(f"     {row['feature']}: {row['importance']:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(top_n_features), x='importance', y='feature')
    plt.title(f'Top {top_n_features} Feature Importances (LightGBM)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    train_logger.info(f"Feature importance saved to {feature_importance_path}")
    plt.close()
    
    # Save final model (trained on full data)
    with open(model_output_path, 'wb') as f:
        pickle.dump(final_model, f)
    train_logger.info(f"Final model saved to {model_output_path}")
    
    train_logger.info("="*80)
    train_logger.info("Training completed successfully!")
    train_logger.info("="*80)


def lightgbm_test(
        feature_files: dict,
        output_dir: str,
    ):
    """
    Evaluate LightGBM model on test data.
    
    Args:
        feature_files: Dictionary mapping feature set names to test file paths
        output_dir: Directory containing trained model and other files
    """
    # Define paths
    model_path = os.path.join(output_dir, 'lightgbm_model.pkl')
    removed_features_path = os.path.join(output_dir, 'removed_features.pkl')
    test_log_path = os.path.join(output_dir, 'lightgbm_test.log')
    predictions_path = os.path.join(output_dir, 'lightgbm_test_predictions.csv')
    
    # Setup logger
    _FILE_HANDLER_EVAL = logging.FileHandler(test_log_path, mode='w')
    _FILE_HANDLER_EVAL.setFormatter(logging.Formatter(_LOG_FORMAT))
    
    eval_logger = logging.getLogger(f'lightgbm.eval.{output_dir}')
    eval_logger.setLevel(logging.INFO)
    eval_logger.handlers.clear()
    eval_logger.addHandler(_STREAM_HANDLER)
    eval_logger.addHandler(_FILE_HANDLER_EVAL)
    
    eval_logger.info("="*80)
    eval_logger.info(f"Testing model from directory: {output_dir}")
    eval_logger.info("="*80)
    
    # Load and merge features
    data = _load_and_merge_features(
        feature_files,
        None,  # No labels for test data
        eval_logger,
        is_train=False
    )
    
    # Preprocess data
    data, _ = _preprocess_data(
        data,
        0.95,  # This parameter doesn't matter for test since we load removed features
        eval_logger,
        removed_features_path,
        is_train=False
    )
    
    # Extract customer IDs and features
    eval_logger.info("Preparing test data...")
    customer_ids = data['customer_ID'].copy()
    X_test = data.drop(columns=['customer_ID'])
    del data
    gc.collect()
    
    eval_logger.info(f"Test features shape: {X_test.shape}")
    
    # Verify all features are numeric
    non_numeric = X_test.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        eval_logger.error(f"ERROR: Non-numeric columns found: {non_numeric.tolist()}")
        raise ValueError("Non-numeric columns found")
    
    # Load trained model
    eval_logger.info(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict probabilities
    eval_logger.info("Generating predictions on test data...")
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction': y_test_pred_proba
    })
    predictions_df.to_csv(predictions_path, index=False)
    eval_logger.info(f"Test predictions saved to {predictions_path}")
    
    eval_logger.info("="*80)
    eval_logger.info("Testing completed successfully!")
    eval_logger.info("="*80)


if __name__ == "__main__":
    # ==================================== Configuration ====================================
    
    # Feature files for training
    FEATURE_FILES = {
        'cat_feature'           : r"S:/ML_Project/new_data/input_train/cat_feature.feather",
        'diff_feature'          : r"S:/ML_Project/new_data/input_train/diff_feature.feather",
        'last3_cat_feature'     : r"S:/ML_Project/new_data/input_train/last3_cat_feature.feather",
        'last3_diff_feature'    : r"S:/ML_Project/new_data/input_train/last3_diff_feature.feather",
        'last3_num_feature'     : r"S:/ML_Project/new_data/input_train/last3_num_feature.feather",
        'last6_num_feature'     : r"S:/ML_Project/new_data/input_train/last6_num_feature.feather",
        'num_feature'           : r"S:/ML_Project/new_data/input_train/num_feature.feather",
        'rank_num_feature'      : r"S:/ML_Project/new_data/input_train/rank_num_feature.feather",
        'ym_rank_num_feature'   : r"S:/ML_Project/new_data/input_train/ym_rank_num_feature.feather",
    }
    
    LABEL_FILE = r"S:/ML_Project/amex-default-prediction/train_labels.csv"
    
    # Feature files for testing
    TEST_FEATURE_FILES = {
        'cat_feature'           : r"S:/ML_Project/new_data/input_test/cat_feature.feather",
        'diff_feature'          : r"S:/ML_Project/new_data/input_test/diff_feature.feather",
        'last3_cat_feature'     : r"S:/ML_Project/new_data/input_test/last3_cat_feature.feather",
        'last3_diff_feature'    : r"S:/ML_Project/new_data/input_test/last3_diff_feature.feather",
        'last3_num_feature'     : r"S:/ML_Project/new_data/input_test/last3_num_feature.feather",
        'last6_num_feature'     : r"S:/ML_Project/new_data/input_test/last6_num_feature.feather",
        'num_feature'           : r"S:/ML_Project/new_data/input_test/num_feature.feather",
        'rank_num_feature'      : r"S:/ML_Project/new_data/input_test/rank_num_feature.feather",
        'ym_rank_num_feature'   : r"S:/ML_Project/new_data/input_test/ym_rank_num_feature.feather",
    }
    
    OUTPUT_DIR = r"S:/ML_Project/new_data/pro_run1/"
    
    # Feature combinations to train/test
    feature_combinations = {
        # ============== Individual Feature Sets (Baseline Comparisons) ==============
        # 'model_cat_only':           ['cat_feature'],
        # 'model_num_only':           ['num_feature'],
        # 'model_diff_only':          ['diff_feature'],
        # 'model_rank_only':          ['rank_num_feature'],
        # 'model_ym_rank_only':       ['ym_rank_num_feature'],
        'model_last3_num_only':     ['last3_num_feature'],
        'model_last6_num_only':     ['last6_num_feature'],
        
        # ============== Two-Feature Combinations (Synergy Testing) ==============
        'model_cat_num':            ['cat_feature', 'num_feature'],
        'model_cat_diff':           ['cat_feature', 'diff_feature'],
        'model_num_diff':           ['num_feature', 'diff_feature'],
        'model_num_rank':           ['num_feature', 'rank_num_feature'],
        'model_cat_rank':           ['cat_feature', 'rank_num_feature'],
        'model_rank_ym_rank':       ['rank_num_feature', 'ym_rank_num_feature'],
        
        # ============== Temporal Feature Combinations ==============
        'model_temporal_last3':     ['last3_num_feature', 'last3_cat_feature', 'last3_diff_feature'],
        'model_temporal_last36':    ['last3_num_feature', 'last6_num_feature'],
        'model_temporal_all':       ['last3_num_feature', 'last6_num_feature', 'last3_cat_feature', 'last3_diff_feature'],
        'model_temporal_with_rank': ['last3_num_feature', 'last6_num_feature', 'ym_rank_num_feature'],
        
        # ============== Ranking Feature Combinations ==============
        'model_all_ranks':          ['rank_num_feature', 'ym_rank_num_feature'],
        'model_ranks_with_base':    ['num_feature', 'rank_num_feature', 'ym_rank_num_feature'],
        
        # ============== Core Features (No Temporal) ==============
        'model_core_basic':         ['cat_feature', 'num_feature', 'diff_feature'],
        'model_core_with_rank':     ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature'],
        'model_core_with_all_rank': ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature'],
        
        # ============== Progressive Builds (Incremental Feature Addition) ==============
        'model_small':              ['cat_feature', 'num_feature'],
        'model_medium':             ['cat_feature', 'num_feature', 'diff_feature'],
        'model_medium_plus':        ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature'],
        'model_large':              ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature'],
        'model_large_plus':         ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature', 'last3_num_feature'],
        'model_xlarge':             ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature', 
                                    'last3_num_feature', 'last6_num_feature'],
        
        # ============== Specialized Combinations ==============
        'model_stat_focused':       ['num_feature', 'diff_feature', 'last3_num_feature', 'last6_num_feature'],
        'model_cat_focused':        ['cat_feature', 'last3_cat_feature'],
        'model_trend_focused':      ['diff_feature', 'last3_diff_feature', 'ym_rank_num_feature'],
        'model_recent_only':        ['last3_num_feature', 'last3_cat_feature', 'last3_diff_feature'],
        
        # ============== Exclusion Testing (What happens without X?) ==============
        'model_no_cat':             ['num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature', 
                                    'last3_num_feature', 'last6_num_feature'],
        'model_no_temporal':        ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature'],
        'model_no_rank':            ['cat_feature', 'num_feature', 'diff_feature', 'last3_num_feature', 
                                    'last6_num_feature', 'last3_cat_feature', 'last3_diff_feature'],
        'model_no_diff':            ['cat_feature', 'num_feature', 'rank_num_feature', 'ym_rank_num_feature', 
                                    'last3_num_feature', 'last6_num_feature', 'last3_cat_feature'],
        
        # ============== Kitchen Sink Models ==============
        'model_almost_full':        ['cat_feature', 'num_feature', 'diff_feature', 'rank_num_feature', 'ym_rank_num_feature',
                                    'last3_num_feature', 'last6_num_feature', 'last3_cat_feature'],
        'model_full':               list(FEATURE_FILES.keys()),  # All features
    }    
    # ==================================== Hyperparameter Grid ====================================
    
    # Number of iterations for RandomizedSearchCV
    num_iter = 50                  # TODO: Increase for thorough tuning (e.g., 10-50)
    hyperparam_sample_frac = 0.2    # TODO: Fraction of training data for hyperparameter tuning (0 < x <= 1.0)
    
    # Simplified param grid for testing
    # param_grid = {
    #     'n_estimators': [100],
    #     'max_depth': [-1],
    #     'learning_rate': [0.1],
    #     'num_leaves': [31],
    #     'min_child_samples': [20],
    #     'subsample': [1.0],
    #     'colsample_bytree': [1.0],
    #     'reg_alpha': [0],
    #     'reg_lambda': [0],
    # }
    
    # Full param grid for thorough tuning (uncomment for full runs)
    # param_grid = {
    #     'n_estimators'      : [100, 500, 1000]              ,
    #     'max_depth'         : [7, 10, 15, -1]               ,
    #     'learning_rate'     : [0.01, 0.03, 0.05, 0.1]       ,
    #     'num_leaves'        : [31, 63, 127, 255]            ,
    #     'min_child_samples' : [10, 20, 30, 50]              ,
    #     'subsample'         : [0.7, 0.8, 0.9]               ,
    #     'colsample_bytree'  : [0.7, 0.8, 0.9]               ,
    #     'reg_alpha'         : [0, 0.1, 0.5, 1.0]            ,
    #     'reg_lambda'        : [0, 0.1, 0.5, 1.0]            ,
    #     'min_split_gain'    : [0.0, 0.1, 0.5]               ,   # Minimum loss reduction required to make a split
    #     'scale_pos_weight'  : [1, 2, 3]                     ,   # Handle class imbalance
    # }

    # Moderate param grid for balanced tuning
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [7, 10, -1],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'min_child_samples': [20, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1],
    }
    # Total combinations: ~512

    # Expanded param grid with distributions for RandomizedSearchCV
    # param_grid = {
    #     # Tree structure
    #     'n_estimators'      : randint(800, 2500),
    #     'max_depth'         : [7, 10, 12, 15, 20, -1],
    #     'num_leaves'        : randint(31, 200),
        
    #     # Learning
    #     'learning_rate'     : loguniform(0.01, 0.2),  # Log-uniform for learning rate
    #     'min_child_samples' : randint(10, 100),
        
    #     # Sampling
    #     'subsample'         : uniform(0.7, 0.3),  # 0.7 to 1.0
    #     'subsample_freq'    : [0, 1, 5, 10],
    #     'colsample_bytree'  : uniform(0.7, 0.3),  # 0.7 to 1.0
        
    #     # Regularization
    #     'reg_alpha'         : loguniform(0.001, 10),
    #     'reg_lambda'        : loguniform(0.001, 10),
    #     'min_split_gain'    : loguniform(0.001, 1.0),
        
    #     # Additional
    #     'path_smooth'       : uniform(0, 1.0),
    # }

    # ==================================== Optional Parameters ====================================
    
    cv_folds            = 5
    holdout_size        = 0.2
    max_missing_rate    = 0.9
    top_n_param         = 5         # Log top 5 hyperparameter combinations
    top_n_features      = 20        # Plot top 20 features by importance
    
    # =============================================================================================
    
    print(
        "==============================================\n"
        "=== LIGHTGBM MODEL TRAINING & TESTING ===\n"
        "==============================================\n"
    )
    
    _ = input(
        "Before proceeding, remember to update paths and configurations.\n"
        "After each run, output files will be saved in separate folders per combination.\n"
        "\nPress Enter to continue...\n"
    )
    
    choice = input("Enter 't' to train the model or 'e' to evaluate on test data: ").strip().lower()
    
    if choice == 't':
        # Training mode: train model for each feature combination
        for combo_name, feature_list in feature_combinations.items():
            print(f"\n{'='*80}")
            print(f"Training model: {combo_name}")
            print(f"Features: {feature_list}")
            print(f"{'='*80}\n")
            
            # Create subset of feature files for this combination
            train_features = {k: FEATURE_FILES[k] for k in feature_list}
            
            # Create output directory for this combination
            combo_output_dir = os.path.join(OUTPUT_DIR, combo_name)
            
            try:
                lightgbm_train(
                    feature_files=train_features,
                    label_file=LABEL_FILE,
                    output_dir=combo_output_dir,
                    num_iter=num_iter,
                    param_grid=param_grid,
                    cv_folds=cv_folds,
                    hyperparam_sample_frac=hyperparam_sample_frac,  
                    holdout_size=holdout_size,
                    max_missing_rate=max_missing_rate,
                    top_n_param=top_n_param,
                    top_n_features=top_n_features,
                )
                print(f"\n✓ Successfully trained {combo_name}\n")
            except Exception as e:
                print(f"\n✗ Error training {combo_name}: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print("All training completed!")
        print(f"{'='*80}\n")
    
    elif choice == 'e':
        # Testing mode: test model for each feature combination
        for combo_name, feature_list in feature_combinations.items():
            print(f"\n{'='*80}")
            print(f"Testing model: {combo_name}")
            print(f"Features: {feature_list}")
            print(f"{'='*80}\n")
            
            # Create subset of feature files for this combination
            test_features = {k: TEST_FEATURE_FILES[k] for k in feature_list}
            
            # Output directory for this combination
            combo_output_dir = os.path.join(OUTPUT_DIR, combo_name)
            
            # Check if model exists
            model_path = os.path.join(combo_output_dir, 'lightgbm_model.pkl')
            if not os.path.exists(model_path):
                print(f"\n✗ Model not found for {combo_name} at {model_path}")
                print(f"   Please train the model first.\n")
                continue
            
            try:
                lightgbm_test(
                    feature_files=test_features,
                    output_dir=combo_output_dir,
                )
                print(f"\n✓ Successfully tested {combo_name}\n")
            except Exception as e:
                print(f"\n✗ Error testing {combo_name}: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print("All testing completed!")
        print(f"{'='*80}\n")
    
    else:
        print("Invalid choice. Please enter 't' to train or 'e' to evaluate.")