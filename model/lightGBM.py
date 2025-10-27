import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
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


def _load_training_data(
        data_file               : str           , 
        label_file              : str           ,
        holdout_size            : float         ,
        max_missing_rate        : float         ,  
        encoders_path           : str           ,
        train_logger            : logging.Logger,
        removed_features_path   : str           ,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and preprocess training data with memory optimization.
    - Uses pandas ArrayManager to avoid huge 2D block allocations during concat.
    - Two-pass: computes global stats (non-null counts, min/max, constants) in pass 1,
      then reads only the needed columns in pass 2 with optimal dtypes.
    - Handles ±Inf per chunk, avoids full-frame boolean materializations.
    """

    # 1) Make pandas concat safer: avoid 2D block consolidation
    #    This stores each column as its own 1D array and skips giant 2D merges.
    try:
        pd.set_option("mode.data_manager", "array")
        train_logger.info("Pandas ArrayManager enabled (mode.data_manager='array').")
    except Exception as e:
        train_logger.warning(f"Could not enable ArrayManager: {e}")

    # Also reduce unnecessary copies
    try:
        pd.options.mode.copy_on_write = True
        train_logger.info("Pandas copy-on-write enabled.")
    except Exception:
        pass

    # Tune chunk size much smaller: with ~3500 cols, 50k rows/chunk is ~700MB/chunk just for float32.
    chunk_size = 10_000
    sample_fraction = 1.0

    train_logger.info(f"Loading training data from {data_file}...")
    train_logger.info(f"Using chunk size: {chunk_size}, sample fraction: {sample_fraction}")

    # =========================
    # == First pass (stats)  ==
    # =========================
    train_logger.info("First pass: Counting rows and computing column-wise stats...")

    total_rows              = 0
    first_chunk             = None
    all_cols    : list[str] = []
    numeric_cols: list[str] = []
    int_cols    : list[str] = []
    float_cols  : list[str] = []
    object_cols : list[str] = []

    # Global stats
    non_null_counts : pd.Series = None  # per column
    gmin            : pd.Series = None  # per numeric col (floats+ints)
    gmax            : pd.Series = None
    int_gmin        : pd.Series = None  # per int col
    int_gmax        : pd.Series = None

    # Track which object columns are constant (only 5 object cols in your data)
    obj_const_value : dict[str, object] = {}    # first non-null seen
    obj_nonconst    : set[str] = set()          # columns proven non-constant

    reader1 = pd.read_csv(data_file, chunksize=chunk_size, low_memory=True)
    for i, chunk in enumerate(reader1):
        if first_chunk is None:
            # Capture schema from the first chunk
            first_chunk = chunk.head(0)  # header only
            all_cols    = list(first_chunk.columns)

            # Determine dtypes using the first chunk
            dtypes_first    = chunk.dtypes
            numeric_cols    = [c for c in all_cols if np.issubdtype(dtypes_first[c], np.number)     and c != 'customer_ID']
            int_cols        = [c for c in all_cols if np.issubdtype(dtypes_first[c], np.integer)    and c != 'customer_ID']
            float_cols      = [c for c in all_cols if np.issubdtype(dtypes_first[c], np.floating)   and c != 'customer_ID']
            object_cols     = [c for c in all_cols if dtypes_first[c] == 'O'                        and c != 'customer_ID']

            # Initialize aggregators
            non_null_counts = pd.Series(0, index=all_cols, dtype="int64")
            if numeric_cols:
                gmin = pd.Series(np.inf, index=numeric_cols, dtype="float64")
                gmax = pd.Series(-np.inf, index=numeric_cols, dtype="float64")
            if int_cols:
                int_gmin = pd.Series(np.iinfo(np.int64).max, index=int_cols, dtype="int64")
                int_gmax = pd.Series(np.iinfo(np.int64).min, index=int_cols, dtype="int64")

            for c in object_cols:
                obj_const_value[c] = None  # unknown yet

            train_logger.info(f"Columns detected: {len(all_cols)}")
            train_logger.info(f"Data types (first chunk): {dtypes_first.value_counts().to_dict()}")

        total_rows += len(chunk)

        # Non-null counts (uses a column-wise reduction; memory friendly)
        non_null_counts += chunk.count()

        # Numeric min/max
        if numeric_cols:
            # note: .min()/.max() here operate per column and return a Series
            num_chunk = chunk[numeric_cols]
            gmin = gmin.combine(num_chunk.min(numeric_only=True), np.minimum)
            gmax = gmax.combine(num_chunk.max(numeric_only=True), np.maximum)

        if int_cols:
            int_chunk = chunk[int_cols]
            int_gmin = int_gmin.combine(int_chunk.min(numeric_only=True), np.minimum)
            int_gmax = int_gmax.combine(int_chunk.max(numeric_only=True), np.maximum)

        # Object constant detection (few columns, do cheaply)
        for col in object_cols:
            if col in obj_nonconst:
                continue
            nn = chunk[col].dropna()
            if nn.empty:
                continue
            uniq = nn.unique()
            if len(uniq) > 1:
                obj_nonconst.add(col)
                obj_const_value[col] = "MULTI"
            else:
                v = uniq[0]
                if obj_const_value[col] is None:
                    obj_const_value[col] = v
                elif obj_const_value[col] != v:
                    obj_nonconst.add(col)
                    obj_const_value[col] = "MULTI"

        if i % 10 == 0:
            train_logger.info(f"Counted {total_rows} rows so far...")

        del chunk
        gc.collect()

    train_logger.info(f"Total rows in file: {total_rows}")

    # Calculate how many rows to actually load
    rows_to_load = int(total_rows * sample_fraction)
    train_logger.info(f"Will load {rows_to_load} rows ({sample_fraction*100:.1f}% of data)")

    # Load labels (small)
    train_logger.info(f"Loading labels from {label_file}...")
    labels = pd.read_csv(label_file)
    labels['target'] = labels['target'].astype('int8')
    valid_customers = set(labels['customer_ID'].values)
    train_logger.info(f"Labels loaded: {labels.shape}, valid customers: {len(valid_customers)}")

    # Decide which columns to drop BEFORE the second pass
    # 1) High-missing columns
    missing_rate = 1.0 - (non_null_counts / total_rows)
    high_missing_cols = missing_rate.index[(missing_rate > max_missing_rate) & (missing_rate.index != 'customer_ID')].tolist()

    # 2) Constant numeric columns
    constant_numeric_cols = []
    if numeric_cols:
        # Constant if non-null exists and min == max
        for c in numeric_cols:
            if non_null_counts.get(c, 0) > 0 and np.isfinite(gmin[c]) and gmin[c] == gmax[c]:
                constant_numeric_cols.append(c)

    # 3) Constant object columns
    constant_object_cols = [c for c, v in obj_const_value.items() if v is not None and v != "MULTI"]

    # 4) All-NaN columns (non_null_counts == 0)
    all_nan_cols = [c for c in all_cols if c != 'customer_ID' and non_null_counts.get(c, 0) == 0]

    # Consolidate drop list
    drop_cols_set = set(high_missing_cols) | set(constant_numeric_cols) | set(constant_object_cols) | set(all_nan_cols)
    keep_cols = [c for c in all_cols if c not in drop_cols_set]
    # We must keep customer_ID for merging
    if 'customer_ID' not in keep_cols:
        keep_cols = ['customer_ID'] + keep_cols
    else:
        # ensure it's the first column to keep for clarity/perf
        keep_cols = ['customer_ID'] + [c for c in keep_cols if c != 'customer_ID']

    train_logger.info(f"Columns to drop before load: {len(drop_cols_set)} "
                      f"(high-missing: {len(high_missing_cols)}, const-num: {len(constant_numeric_cols)}, "
                      f"const-obj: {len(constant_object_cols)}, all-NaN: {len(all_nan_cols)})")
    train_logger.info(f"Columns to keep: {len(keep_cols)} (incl. customer_ID)")

    # 2) Build optimized dtype dict using GLOBAL min/max from pass 1, not only first chunk
    def _best_int_dtype(mn: int, mx: int) -> str:
        if mn >= 0:
            if mx <= np.iinfo(np.uint8).max:
                return 'uint8'
            if mx <= np.iinfo(np.uint16).max:
                return 'uint16'
            if mx <= np.iinfo(np.uint32).max:
                return 'uint32'
            return 'uint64'
        else:
            if mn >= np.iinfo(np.int8).min and mx <= np.iinfo(np.int8).max:
                return 'int8'
            if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
                return 'int16'
            if mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                return 'int32'
            return 'int64'

    dtype_dict: dict[str, str] = {}
    # Use first chunk dtypes as guidance for type family
    # Fallbacks: floats -> float32; ints -> best compact int; objects -> category
    dtypes_first = None
    if first_chunk is not None:
        # we didn't keep the full first chunk, but we can re-read a tiny sample for dtype guide
        dtypes_first = pd.read_csv(data_file, nrows=10).dtypes

    for col in keep_cols:
        if col == 'customer_ID':
            dtype_dict[col] = 'string'  # minimize memory but preserve mergeability
            continue
        if dtypes_first is not None:
            dt = dtypes_first.get(col, None)
        else:
            dt = None
        if dt is not None:
            if np.issubdtype(dt, np.floating):
                dtype_dict[col] = 'float32'
            elif np.issubdtype(dt, np.integer):
                mn = int(int_gmin[col]) if int_gmin is not None and col in int_gmin.index else None
                mx = int(int_gmax[col]) if int_gmax is not None and col in int_gmax.index else None
                if mn is not None and mx is not None:
                    dtype_dict[col] = _best_int_dtype(mn, mx)
                else:
                    dtype_dict[col] = 'int32'
            elif dt == 'O':
                dtype_dict[col] = 'category'
            else:
                # default to float32 for safety with LightGBM
                dtype_dict[col] = 'float32'
        else:
            dtype_dict[col] = 'float32'

    train_logger.info(f"Optimized dtype mapping for {len(dtype_dict)} columns (pre-load).")

    # =========================
    # == Second pass (load)  ==
    # =========================
    train_logger.info("Second pass: Loading filtered data with optimized dtypes...")

    chunks: list[pd.DataFrame] = []
    rows_loaded = 0

    chunk_iter = pd.read_csv(
        data_file,
        chunksize=chunk_size,
        dtype=dtype_dict,
        usecols=keep_cols,   # critical: don't even read dropped columns
        low_memory=True
    )

    # For memory safety, do not force mid-run concatenations (that triggered OOM).
    # Just keep a moderate number of chunks and only concat once at the end.
    max_in_memory_chunks = 30  # 30 x (10k rows x ~3000 cols float32) ~ acceptable on 32GB; adjust to your RAM.

    # Precompute numeric columns among keep_cols
    keep_numeric_cols = [c for c in keep_cols if c != 'customer_ID' and (c in numeric_cols or (dtypes_first is not None and np.issubdtype(dtypes_first.get(c, np.float32), np.number)))]

    for i, chunk in enumerate(chunk_iter):
        if rows_loaded >= rows_to_load:
            break

        # Filter to only customers with labels
        chunk = chunk[chunk['customer_ID'].isin(valid_customers)]
        if chunk.empty:
            train_logger.info(f"Chunk {i+1}: No matching customers, skipping")
            del chunk
            gc.collect()
            continue

        # Merge target immediately
        chunk = chunk.merge(
            labels[['customer_ID', 'target']],
            on='customer_ID',
            how='inner',
            copy=False
        )

        # Handle ±Inf in this chunk only (avoid giant 2D boolean arrays)
        if keep_numeric_cols:
            # column-wise check to keep peak memory small
            for col in keep_numeric_cols:
                col_vals = chunk[col].to_numpy(copy=False)
                if np.isinf(col_vals).any():
                    chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan)

        # Drop customer_ID asap
        chunk.drop(columns=['customer_ID'], inplace=True)

        chunks.append(chunk)
        rows_loaded += len(chunk)

        train_logger.info(f"Loaded chunk {i+1}: {len(chunk)} rows (Total: {rows_loaded}/{rows_to_load})")

        # Limit number of in-memory chunks to avoid long lists and Python overhead.
        # NOTE: We don't concatenate here to avoid the giant 2D block allocation; ArrayManager further reduces risk.
        if len(chunks) >= max_in_memory_chunks:
            train_logger.info(f"Reached {len(chunks)} in-memory chunks; performing a safe concat (ArrayManager).")
            tmp = pd.concat(chunks, ignore_index=True, copy=False)
            chunks = [tmp]
            del tmp
            gc.collect()
            train_logger.info(f"Current aggregated shape: {chunks[0].shape}")

    # Clean up labels to free memory
    del labels, valid_customers
    gc.collect()

    # Final concatenation (on ArrayManager; no giant 2D blocks)
    train_logger.info("Final concatenation of remaining chunks...")
    if len(chunks) == 1:
        data = chunks[0]
    else:
        data = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()

    train_logger.info(f"Data loaded and merged successfully: {data.shape}")


    # =========================
    # ==    Split X and y    ==
    # =========================
    y = data['target'].astype('int8', copy=False)
    X = data.drop('target', axis=1)
    del data
    gc.collect()

    train_logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")


    # =========================
    # == Encode categoricals ==
    # =========================
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        train_logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    train_logger.info(f"Label encoders saved to {encoders_path}")


    # =========================
    # ==   Critical checks   ==
    # =========================
    # 1) Replace any remaining ±Inf per column (avoid full-matrix boolean)
    numeric_cols_after = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols_after:
        col_vals = X[col].to_numpy(copy=False)
        if np.isinf(col_vals).any():
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    # 2) Enforce numeric-only
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        train_logger.error(f"ERROR: Non-numeric columns remain: {non_numeric.tolist()}")
        raise ValueError("Non-numeric columns found after encoding")

    # 3) Constant features (this should be rare now; we removed most earlier)
    train_logger.info("Checking for constant features (post-load)...")
    constant_features = []
    for col in X.columns:
        # cheap check: min==max and not all NaN
        series = X[col]
        if series.notna().any():
            vmin = series.min(skipna=True)
            vmax = series.max(skipna=True)
            if vmin == vmax:
                constant_features.append(col)

    if constant_features:
        train_logger.warning(f"Removing {len(constant_features)} constant features (post-load)")
        X.drop(columns=constant_features, inplace=True)
        gc.collect()

    # 4) All-NaN columns (post-load safety)
    train_logger.info("Checking for all-NaN columns (post-load)...")
    # do per-column to avoid a giant boolean frame
    post_all_nan_cols = []
    for col in X.columns:
        if X[col].isna().all():
            post_all_nan_cols.append(col)
    if post_all_nan_cols:
        train_logger.warning(f"Removing {len(post_all_nan_cols)} all-NaN columns (post-load)")
        X.drop(columns=post_all_nan_cols, inplace=True)
        gc.collect()

    # 5) Missingness stats (memory-friendly)
    nan_counts = pd.Series({col: int(X[col].isna().sum()) for col in X.columns})
    total_nans = int(nan_counts.sum())
    cols_with_nans = int((nan_counts > 0).sum())
    if total_nans > 0:
        train_logger.info(f"NaN statistics: {total_nans} NaN values across {cols_with_nans} columns")
        top_nan = nan_counts.sort_values(ascending=False).head(10)
        train_logger.info(f"Top columns with NaN:\n{top_nan}")

    # Remove features with too many NaNs (using same threshold)
    nan_rates = nan_counts / len(X)
    high_missing_cols_post = nan_rates.index[nan_rates > max_missing_rate].tolist()
    if high_missing_cols_post:
        train_logger.warning(f"Removing {len(high_missing_cols_post)} features with >{max_missing_rate*100:.1f}% missing (post-load)")
        X.drop(columns=high_missing_cols_post, inplace=True)
        gc.collect()

    if X.shape[1] == 0:
        train_logger.error("ERROR: No features remaining after cleaning!")
        raise ValueError("All features were removed")

    train_logger.info(f"After cleaning: {X.shape[1]} features, {X.shape[0]} samples")

    # Record removed features
    removed = []
    removed.extend(high_missing_cols)
    removed.extend(constant_numeric_cols)
    removed.extend(constant_object_cols)
    removed.extend(all_nan_cols)
    removed.extend(constant_features)
    removed.extend(post_all_nan_cols)
    removed.extend(high_missing_cols_post)

    # Keep unique while preserving order
    seen = set()
    removed_unique = [x for x in removed if not (x in seen or seen.add(x))]
    if removed_unique:
        with open(removed_features_path, 'wb') as f:
            pickle.dump(removed_unique, f)
        train_logger.info(f"Saved {len(removed_unique)} removed feature names to {removed_features_path}")


    # =========================
    # ==  Train/Val split    ==
    # =========================
    train_logger.info("Splitting into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X                               ,
        y                               ,
        test_size       = holdout_size  ,
        random_state    = RANDOM_SEED   ,
        stratify        = y             ,
    )

    # Clean up
    del X, y
    gc.collect()

    train_logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    train_logger.info(f"Training labels:\n{y_train.value_counts(normalize=True)}")
    train_logger.info(f"Validation labels:\n{y_val.value_counts(normalize=True)}")

    return X_train, y_train, X_val, y_val


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
        train_data_file         : str                                           ,
        train_label_file        : str                                           ,
        
        num_iter                : int   = 10                                    ,
        param_grid              : dict  = None                                  ,
        cv_folds                : int   = 5                                     ,
        holdout_size            : float = 0.2                                   ,
        max_missing_rate        : float = 0.95                                  ,
        top_n_param             : int   = 5                                     ,
        top_n_features          : int   = 20                                    ,
        
        encoders_path           : str   = 'label_encoders.pkl'                  ,
        model_output_path       : str   = 'lightgbm_model.pkl'                  ,
        cv_results_path         : str   = 'lightgbm_grid_search_cv_results.csv' ,
        confusion_matrix_path   : str   = 'confusion_matrix_lgb.png'            ,
        feature_importance_path : str   = 'feature_importance_lgb.png'          ,
        train_log_path          : str   = 'lightgbm_train.log'                  ,
        removed_features_path   : str   = 'removed_features.pkl'                ,
    ):
    """
    Train LightGBM model with hyperparameter tuning.
    """

    _FILE_HANDLER_TRAIN = logging.FileHandler(train_log_path, mode='w')
    _FILE_HANDLER_TRAIN.setFormatter(logging.Formatter(_LOG_FORMAT))

    # Train logger
    train_logger = logging.getLogger('lightgbm.train')
    train_logger.setLevel(logging.INFO)
    if not train_logger.hasHandlers():
        train_logger.addHandler(_STREAM_HANDLER)
        train_logger.addHandler(_FILE_HANDLER_TRAIN)

    X_train, y_train, X_val, y_val = _load_training_data(
        train_data_file     , 
        train_label_file    ,

        holdout_size        ,
        max_missing_rate    ,

        encoders_path       ,
        train_logger        ,
        removed_features_path,
    )

    train_logger.info ("Creating LightGBM model...")
    
    # Create LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        objective       = 'binary'      ,
        random_state    = RANDOM_SEED   ,
        force_col_wise  = True          ,   # Remove the overhead warning
        # verbose=-1                     ,   # Reduce warning spam
    )
    
    # GridSearchCV
    train_logger.info("Starting hyperparameter tuning...")
    train_logger.info("Scoring metric: ROC-AUC (proxy for AMEX metric)")

    # amex_scorer = make_scorer(
    #     _amex_metric_sklearn, 
    #     needs_proba         = True,  # Use predicted probabilities
    #     greater_is_better   = True
    # )

    # Define stratified CV
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    grid_search = RandomizedSearchCV(
        lgb_model                           ,
        param_grid                          ,
        cv                  = cv_strategy   ,
        scoring             = 'roc_auc'     ,   # Using ROC-AUC as proxy for AMEX metric
        return_train_score  = True          ,
        n_iter              = num_iter      ,
        n_jobs              = 1             ,   # Process one fold at a time
        pre_dispatch        = 1             ,   # Only pre-dispatch 1 job (don't queue multiple)
    )
    grid_search.fit(X_train, y_train)

    train_logger.info("Hyperparameter tuning completed.")
    
    cv_results      = grid_search.cv_results_   # Dctionary of CV results
    best_idx        = grid_search.best_index_   # Index of best param combo (ie candidate)
    n_candidates    = len(cv_results['params']) # Total param combo evaluated
    total_fits      = n_candidates * cv_folds   # Total fits across all folds

    # Timing for best candidate
    mean_fit_time   = cv_results['mean_fit_time'][best_idx]     # Average fit time per fold
    mean_score_time = cv_results['mean_score_time'][best_idx]   # Average score time per fold

    # Estimate total time spent fitting across all candidates and folds
    total_fit_time_seconds = (cv_results['mean_fit_time'] * cv_folds).sum()
    total_fit_time_minutes = total_fit_time_seconds / 60.0

    # Refit time (time to retrain best model on full training set)
    refit_time = getattr(grid_search, "refit_time_", None)

    train_logger.info(f"Evaluated {n_candidates} hyperparameter combinations across {cv_folds} folds "
                 f"(total fits: {total_fits})")
    train_logger.info(f"Best params: {grid_search.best_params_} (candidate index: {best_idx})")
    train_logger.info(f"     Mean AMEX metric score (across {cv_folds} folds): {grid_search.best_score_:.6f} "
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

    # Log top-N candidates with their mean/std CV scores for quick inspection
    try:
        top_indices = np.argsort(cv_results['mean_test_score'])[::-1][:top_n_param]
        train_logger.info(f"Top {len(top_indices)} hyperparameter candidates (by mean CV score):")

        for rank, idx in enumerate(top_indices, start=1):
            mean_score  = cv_results['mean_test_score'][idx]
            std_score   = cv_results['std_test_score'][idx]
            fit_time    = cv_results['mean_fit_time'][idx]
            params      = cv_results['params'][idx]

            train_logger.info(f"     Rank {rank}: mean_score={mean_score:.6f} std={std_score:.6f} "
                         f"avg_fit_time_per_fold={fit_time:.3f}s params={params}")
    except Exception as e:
        train_logger.warning(f"Could not extract top candidates from cv_results_: {e}")

    # Persist CV results for offline analysis
    try:
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(cv_results_path, index=False)
        train_logger.info(f"Grid search CV results saved to: {cv_results_path}")
    except Exception as e:
        train_logger.warning(f"Failed to save cv_results_ to CSV: {e}")
    
    # Get best model
    best_model = grid_search.best_estimator_

    # For LightGBM specifically
    n_trees = best_model.n_estimators
    train_logger.info(f"Best model uses {n_trees} trees")
    

    # ============== EVALUATION ON HOLDOUT ==============
    train_logger.info("Evaluating best model on holdout validation set...")
    
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    # Metrics
    val_auc = roc_auc_score(y_val, y_pred_proba)
    train_logger.info(f"Validation ROC-AUC: {val_auc:.4f}")
    
    amex_score = _amex_metric_sklearn (y_val, y_pred_proba)
    train_logger.info(f"Validation AMEX metric: {amex_score:.6f}")
    train_logger.info(f"CV to Validation ROC-AUC difference: {val_auc - grid_search.best_score_:.6f}")
    train_logger.info("Note: Negative difference may indicate slight overfitting during hyperparameter tuning.")
    
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
    train_logger.info("Confusion matrix saved")
    plt.close()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature'   : X_train.columns,
        'importance': best_model.feature_importances_
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
    train_logger.info("Feature importance saved")
    plt.close()
    
    # Save model
    with open(model_output_path, 'wb') as f:
        pickle.dump(best_model, f)

    
def lightgbm_test(
        test_data_file  : str,
        encoders_path   : str,
        model_path      : str,
        test_log_path   : str = 'lightgbm_eval.log',
        predictions_path: str = 'lightgbm_test_predictions.csv',
        removed_features_path: str = 'removed_features.pkl',
    ):
    """
    Evaluate LightGBM model on test data.
    """

    # File handlers
    _FILE_HANDLER_EVAL = logging.FileHandler(test_log_path, mode='w')
    _FILE_HANDLER_EVAL.setFormatter(logging.Formatter(_LOG_FORMAT))


    # Evaluation / test logger
    eval_logger = logging.getLogger('lightgbm.eval')
    eval_logger.setLevel(logging.INFO)
    if not eval_logger.hasHandlers():
        eval_logger.addHandler(_STREAM_HANDLER)
        eval_logger.addHandler(_FILE_HANDLER_EVAL)
        
    # Load test data
    eval_logger.info(f"Loading test data from {test_data_file}...")
    # X_test : pd.DataFrame  = pd.read_csv (test_data_file)

    X_test = None
    try:
        # Load in chunks with optimized dtypes
        chunks = []
        chunk_size=100000
        chunk_iter = pd.read_csv(
            test_data_file,
            chunksize=chunk_size,
            # dtype=np.float32,  # Use float32 instead of float64 (saves 50% memory)
            low_memory=True
        )
        
        eval_logger.info("Loading data in chunks...")
        for i, chunk in enumerate(chunk_iter):
            eval_logger.info(f"Processing chunk {i+1} ({len(chunk)} rows)...")
            chunks.append(chunk)
            
            # # Stop if we're running out of memory
            # if len(chunks) * chunk_size >= 1000:  # Safety limit
            #     logging.warning("Reached safety limit for in-memory chunks")
            #     break

            chunk = None  # Free memory
        
        X_test = pd.concat(chunks, ignore_index=True)
        eval_logger.info(f"Test data loaded: {X_test.shape}")
        
    except MemoryError as e:
        eval_logger.error(f"Memory error while loading test data: {e}")
        eval_logger.info("Try reducing chunk_size")
        raise

    # Load label encoders
    eval_logger.info(f"Loading label encoders from {encoders_path}...")
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)

    # Load removed features list (if exists) and drop these columns from test set
    try:
        if os.path.exists(removed_features_path):
            with open(removed_features_path, 'rb') as f:
                removed_features = pickle.load(f)
            # Drop only columns that actually exist in X_test
            cols_to_drop = [c for c in removed_features if c in X_test.columns]
            if len(cols_to_drop) > 0:
                eval_logger.info(f"Dropping {len(cols_to_drop)} features from test data that were removed during training: {cols_to_drop[:20]}")
                X_test = X_test.drop(columns=cols_to_drop)
            else:
                eval_logger.info("No removed-features from training were present in test data to drop.")
        else:
            eval_logger.info(f"Removed-features file not found at {removed_features_path}; skipping drop step.")
    except Exception as e:
        eval_logger.warning(f"Could not load/drop removed features from {removed_features_path}: {e}")

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in X_test.columns:
            eval_logger.info(f"Label encoding column: {col}")
            X_test[col] = le.transform(X_test[col].astype(str))
        else:
            eval_logger.warning(f"Column {col} not found in test data for encoding.")

    # Load trained model
    eval_logger.info(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Predict probabilities
    eval_logger.info("Generating predictions on test data...")
    customer_ids = X_test['customer_ID'].copy()
    X_test = X_test.drop(columns=['customer_ID'])

    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Save predictions
    predictions_df = pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction' : y_test_pred_proba
    })
    predictions_df.to_csv(predictions_path, index=False)
    eval_logger.info(f"Test predictions saved to {predictions_path}")
    

if __name__ == "__main__":
    # ==================================== TODO - Compulsory Section ====================================
    # TODO: Update file paths before running
    train_data_file     = r"C:\Users\leyan\Downloads\train_data_with_time_series_statistics.csv"
    train_label_file    = r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000-ML-Project\data\train_labels.csv"
    test_data_file      = r"C:\Users\leyan\Downloads\test_data_with_simple_summary_statistics.csv"
    
    # During Cross Validation, the model will try different hyperparameter combinations in random. 
    # The number of combinations tried is controlled by `num_iter`.
    # If you want to do a quick test run, I recommend setting num_iter to 1 and using the simplified param_grid below.
    # For thorough hyperparameter tuning, increase num_iter (eg 10 to 50) and use the full param_grid.
    
    num_iter = 1   # TODO: Number of hyperparameter combinations to try in RandomizedSearchCV

    # Simplified param grid for testing (TODO - use this for quick test runs)
    param_grid = {
        'n_estimators'      : [100]         ,   # Just one value
        'max_depth'         : [-1]          ,   # No limit
        'learning_rate'     : [0.1]         ,   # Single value
        'num_leaves'        : [31]          ,   # Default
        'min_child_samples' : [20]          ,   # Default
        'subsample'         : [1.0]         ,   # No subsampling
        'colsample_bytree'  : [1.0]         ,   # Use all features
        'reg_alpha'         : [0]           ,   # No regularization
        'reg_lambda'        : [0]           ,   # No regularization
    }

    # Full param grid for thorough tuning (TODO - use this for full runs)
    # param_grid = {
    #     'n_estimators'      : [100, 200, 500]          ,
    #     'max_depth'         : [-1, 10, 20]             ,
    #     'learning_rate'     : [0.01, 0.05, 0.1]        ,
    #     'num_leaves'        : [31, 63, 127]            ,
    #     'min_child_samples' : [10, 20, 50]             ,
    #     'subsample'         : [0.8, 0.9, 1.0]          ,
    #     'colsample_bytree'  : [0.8, 0.9, 1.0]          ,
    #     'reg_alpha'         : [0, 0.01, 0.1]           ,
    #     'reg_lambda'        : [0, 0.01, 0.1]           ,
    # }

    # =============================================================================================


    # ==================================== Optional Parameters ====================================
    
    # They have default values, modify if needed
    cv_folds                = 2                                         # Number of CV folds for hyperparameter tuning
    holdout_size            = 0.2                                       # Fraction of data to hold out for validation
    max_missing_rate        = 0.95                                      # Max allowed missing rate per feature (features with higher missing rate will be removed)
    top_n_param             = 5                                         # Top-N candidates to log from hyperparameter tuning (for inspection, no effect on model)
    top_n_features          = 20                                        # Top-N features to plot for feature importance (for inspection, no effect on model)

    # You can modify these paths if needed
    # If not, I believe these files will end up either in your current directory or C:\Users\<username>>
    encoders_path           = 'label_encoders.pkl'                      # Path to save/load label encoders
    model_output_path       = 'lightgbm_model.pkl'                      # Path to save trained model
    cv_results_path         = 'lightgbm_grid_search_cv_results.csv'     # Path to save CV results CSV
    confusion_matrix_path   = 'confusion_matrix_lgb.png'                # Path to save confusion matrix plot
    feature_importance_path = 'feature_importance_lgb.png'              # Path to save feature importance plot
    train_log_path          = 'lightgbm_train.log'                      # Path to save training log
    test_log_path           = 'lightgbm_test.log'                       # Path to save etesting log
    prediction_results_path = 'lightgbm_test_predictions.csv'           # Path to save test predictions CSV
    removed_features_path   = 'removed_features.pkl'                    # Path to save/load removed features list
    
    # Note:
    # encoders_path: This script automatically encodes any categorical features with non-numeric values.
    #                The encoders are saved to this path during training and loaded from this path during testing.
    #                So that the same encoding is applied to both training and test data, ensure this path is consistent.
    #
    # model_output_path: Path to save the trained LightGBM model after training.
    #                    During testing, the model is loaded from this path to make predictions on test data.

    # =============================================================================================


    print (
        "==============================================\n"\
        "=== RANDOM FOREST MODEL TRAINING & TESTING ===\n"\
        "==============================================\n"
    )
    
    _ = input (
        "Before proceeding, remenber to update TODO under `if __name__ == __main__`\n" \
        "And after you have run finished train / test, save the output files in another location\n" \
        "to avoid overwriting them in subsequent runs.\n" \
        "\n"
        "\nPress Enter to continue...\n"
    )

    choice = input("Enter 't' to train the model or 'e' to evaluate on test data: ").strip().lower()
    if choice == 't':
        lightgbm_train(
            train_data_file         ,
            train_label_file        ,

            num_iter                = num_iter                ,
            param_grid              = param_grid              ,
            cv_folds                = cv_folds                ,
            holdout_size            = holdout_size            ,
            max_missing_rate        = max_missing_rate        ,
            top_n_param             = top_n_param             ,
            top_n_features          = top_n_features          ,

            encoders_path           = encoders_path           ,
            model_output_path       = model_output_path       ,
            cv_results_path         = cv_results_path         ,
            confusion_matrix_path   = confusion_matrix_path   ,
            feature_importance_path = feature_importance_path ,
            train_log_path          = train_log_path          ,
            removed_features_path   = removed_features_path   ,
        )
    elif choice == 'e':
        lightgbm_test(
            test_data_file  ,
            encoders_path   ,
            model_output_path,
            test_log_path   ,
            prediction_results_path,
            removed_features_path,
        )
    else:
        print("Invalid choice. Please enter 't' to train or 'e' to evaluate.")      


