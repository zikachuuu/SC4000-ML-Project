"""
================================================================================
RANDOM FOREST CLASSIFIER FOR CREDIT DEFAULT PREDICTION
================================================================================

Purpose: Train and evaluate Random Forest model for predicting credit card default
Input:  Preprocessed aggregated data (train_data_aggregated_simple.csv)
Output: Trained model, performance metrics, feature importance analysis

Features:
- Proper train/validation/test split
- Hyperparameter tuning with cross-validation
- Class imbalance handling
- Comprehensive evaluation metrics
- Feature importance analysis
- Model persistence (save/load)

Author: Credit Risk Team
Date: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import joblib
import warnings
from typing import Dict, Tuple, List
import time

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class CreditDefaultRandomForest:
    """
    Complete Random Forest pipeline for credit default prediction.
    
    This class handles:
    - Data loading and preprocessing
    - Train/validation/test splitting
    - Model training with optimized hyperparameters
    - Comprehensive evaluation
    - Feature importance analysis
    - Model persistence
    
    Attributes:
        model: Trained RandomForestClassifier
        feature_names: List of feature column names
        performance_metrics: Dictionary of evaluation metrics
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 random_state: int = RANDOM_SEED):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of each tree
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=100,      # Minimum samples to split a node
            min_samples_leaf=50,        # Minimum samples in leaf node
            max_features='sqrt',        # √m features per split
            class_weight='balanced',    # Handle class imbalance
            random_state=random_state,
            n_jobs=-1,                  # Use all CPU cores
            verbose=1,                  # Show progress
            oob_score=True             # Out-of-bag score for validation
        )
        
        self.feature_names = None
        self.performance_metrics = {}
        self.importance_df = None
        
    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    def load_and_prepare_data(self, filepath: str, 
                             test_size: float = 0.2,
                             val_size: float = 0.1) -> Tuple:
        """
        Load preprocessed data and split into train/validation/test sets.
        
        Args:
            filepath (str): Path to preprocessed CSV file
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of remaining data for validation set
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            
        Data Split Strategy:
            - Test set: 20% (held out for final evaluation)
            - Validation set: 10% of remaining (for hyperparameter tuning)
            - Train set: 70% (for model training)
            
        Example with 100,000 customers:
            - Train: 70,000 customers
            - Validation: 10,000 customers
            - Test: 20,000 customers
        """
        print("="*80)
        print("DATA LOADING AND PREPARATION")
        print("="*80)
        
        # Load data
        print(f"\nLoading data from: {filepath}")
        data = pd.read_csv(filepath)
        
        print(f"Data shape: {data.shape}")
        print(f"Columns: {len(data.columns)}")
        
        # Validate required columns
        if 'customer_ID' not in data.columns:
            raise ValueError("Missing 'customer_ID' column")
        if 'target' not in data.columns:
            raise ValueError("Missing 'target' column")
        
        # Separate features and target
        X = data.drop(['customer_ID', 'target'], axis=1)
        y = data['target']
        
        self.feature_names = X.columns.tolist()
        
        print(f"\nFeatures: {len(self.feature_names)}")
        print(f"Samples: {len(X)}")
        
        # Check class distribution
        class_dist = y.value_counts(normalize=True)
        print(f"\nClass distribution:")
        print(f"  Non-default (0): {class_dist[0]:.2%}")
        print(f"  Default (1): {class_dist[1]:.2%}")
        print(f"  Imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
        
        # Check missing values
        missing_pct = (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100
        print(f"\nMissing values: {missing_pct:.2f}%")
        
        if missing_pct > 0:
            print("Filling missing values with median...")
            # Store medians for later use on test data
            self.feature_medians = X.median()
            X = X.fillna(self.feature_medians)
        
        # Split into train and test
        print(f"\nSplitting data:")
        print(f"  Test set: {test_size:.0%}")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=RANDOM_SEED,
            stratify=y  # Maintain class distribution
        )
        
        # Split remaining into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        print(f"  Validation set: {val_size:.0%}")
        print(f"  Train set: {1 - test_size - val_size:.0%}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=RANDOM_SEED,
            stratify=y_temp
        )
        
        # Summary
        print(f"\nFinal split:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
        print(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
        
        # Verify class distribution in each set
        print(f"\nDefault rate by split:")
        print(f"  Train: {y_train.mean():.2%}")
        print(f"  Validation: {y_val.mean():.2%}")
        print(f"  Test: {y_test.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'CreditDefaultRandomForest':
        """
        Train the Random Forest model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            
        Returns:
            self: Fitted model instance
        """
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        print(f"\nModel configuration:")
        print(f"  Number of trees: {self.model.n_estimators}")
        print(f"  Max depth: {self.model.max_depth}")
        print(f"  Min samples split: {self.model.min_samples_split}")
        print(f"  Min samples leaf: {self.model.min_samples_leaf}")
        print(f"  Max features: {self.model.max_features}")
        print(f"  Class weight: {self.model.class_weight}")
        
        print(f"\nTraining Random Forest on {len(X_train):,} samples...")
        print("This may take a few minutes...\n")
        
        start_time = time.time()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Out-of-bag score (internal validation)
        print(f"\nOut-of-bag (OOB) score: {self.model.oob_score_:.4f}")
        print("(OOB score estimates generalization performance)")
        
        return self
    
    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                dataset_name: str = "Test") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            dataset_name (str): Name of dataset for reporting
            
        Returns:
            Dict: Dictionary of performance metrics
        """
        print("\n" + "="*80)
        print(f"MODEL EVALUATION - {dataset_name} Set")
        print("="*80)
        
        # Get predictions
        print(f"\nGenerating predictions for {len(X_test):,} samples...")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # 1. ROC-AUC (primary metric for imbalanced classification)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics['roc_auc'] = auc
        
        # 2. Average Precision (good for imbalanced data)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        metrics['avg_precision'] = avg_precision
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # 4. Derived metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                             (metrics['precision'] + metrics['recall']) if \
                             (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Print results
        print(f"\nPERFORMANCE METRICS:")
        print(f"{'='*60}")
        print(f"ROC-AUC Score:        {metrics['roc_auc']:.4f} ⭐ (Primary metric)")
        print(f"Average Precision:    {metrics['avg_precision']:.4f}")
        print(f"Accuracy:             {metrics['accuracy']:.4f}")
        print(f"Precision:            {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity:          {metrics['specificity']:.4f}")
        print(f"F1-Score:             {metrics['f1_score']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"{'='*60}")
        print(f"                 Predicted")
        print(f"                 No Default  |  Default")
        print(f"Actual  No Default    {tn:6d}    |   {fp:6d}")
        print(f"        Default       {fn:6d}    |   {tp:6d}")
        
        # Interpretation
        print(f"\nINTERPRETATION:")
        print(f"{'='*60}")
        print(f"True Negatives:  {tn:6,} (Correctly predicted no default)")
        print(f"False Positives: {fp:6,} (Incorrectly predicted default) - Type I Error")
        print(f"False Negatives: {fn:6,} (Missed actual defaults) - Type II Error ⚠️")
        print(f"True Positives:  {tp:6,} (Correctly predicted default)")
        
        # Business impact
        total_defaults = tp + fn
        detected_defaults = tp
        missed_defaults = fn
        false_alarms = fp
        
        print(f"\nBUSINESS IMPACT:")
        print(f"{'='*60}")
        print(f"Total actual defaults:     {total_defaults:6,}")
        print(f"Defaults detected:         {detected_defaults:6,} ({detected_defaults/total_defaults:.1%})")
        print(f"Defaults missed:           {missed_defaults:6,} ({missed_defaults/total_defaults:.1%}) ⚠️")
        print(f"False alarms:              {false_alarms:6,}")
        
        # Store metrics
        self.performance_metrics[dataset_name] = metrics
        
        return metrics
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      save_path: str = None):
        """
        Plot ROC curve.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            save_path (str): Path to save plot (optional)
        """
        print("\nGenerating ROC curve...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: str = None):
        """
        Plot Precision-Recall curve.
        
        Particularly useful for imbalanced datasets.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            save_path (str): Path to save plot (optional)
        """
        print("\nGenerating Precision-Recall curve...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.axhline(y=y_test.mean(), color='navy', linestyle='--', lw=2,
                   label=f'Baseline (Default rate = {y_test.mean():.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                            save_path: str = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            save_path (str): Path to save plot (optional)
        """
        print("\nGenerating confusion matrix heatmap...")
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    
    def analyze_feature_importance(self, top_n: int = 30, save_path: str = None):
        """
        Analyze and visualize feature importance.
        
        Args:
            top_n (int): Number of top features to display
            save_path (str): Path to save plot (optional)
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.importance_df = importance_df
        
        # Display top features
        print(f"\nTop {top_n} Most Important Features:")
        print("="*80)
        print(importance_df.head(top_n).to_string(index=False))
        
        # Analyze feature types
        print(f"\nFeature Type Analysis:")
        print("="*60)
        
        feature_type_importance = {}
        
        for feat_type in ['_last', '_mean', '_std', '_min', '_max', 
                         'trend', 'seasonal', 'volatility', 'acf', 'changepoint']:
            type_features = importance_df[importance_df['feature'].str.contains(feat_type, case=False)]
            if len(type_features) > 0:
                total_importance = type_features['importance'].sum()
                avg_importance = type_features['importance'].mean()
                count = len(type_features)
                
                feature_type_importance[feat_type] = {
                    'count': count,
                    'total_importance': total_importance,
                    'avg_importance': avg_importance
                }
                
                print(f"{feat_type:15s}: {count:4d} features, "
                      f"total importance = {total_importance:.4f}, "
                      f"avg = {avg_importance:.6f}")
        
        # Visualization
        print(f"\nGenerating feature importance plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Top N features
        top_features = importance_df.head(top_n)
        ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance', fontsize=12)
        ax1.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Feature type importance
        if feature_type_importance:
            types = list(feature_type_importance.keys())
            importances = [feature_type_importance[t]['total_importance'] for t in types]
            
            ax2.bar(range(len(types)), importances, color='coral')
            ax2.set_xticks(range(len(types)))
            ax2.set_xticklabels(types, rotation=45, ha='right')
            ax2.set_ylabel('Total Importance', fontsize=12)
            ax2.set_title('Feature Type Importance', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
        
        return importance_df
    
    # ========================================================================
    # HYPERPARAMETER TUNING
    # ========================================================================
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                            cv: int = 3) -> Dict:
        """
        Tune hyperparameters using Grid Search with Cross-Validation.
        
        WARNING: This can be very slow! Consider using a subset of data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict: Best parameters found
        """
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING")
        print("="*80)
        print("\n⚠️  WARNING: This may take 30+ minutes for large datasets!")
        print("Consider using a subset of data for faster tuning.\n")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [50, 100, 200],
            'min_samples_leaf': [25, 50, 100],
            'max_features': ['sqrt', 'log2']
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
        print(f"Cross-validation folds: {cv}")
        print(f"Total fits: {np.prod([len(v) for v in param_grid.values()]) * cv}")
        
        # Initialize grid search
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            verbose=2,
            n_jobs=1  # GridSearchCV parallelizes over folds, not estimators
        )
        
        print("\nStarting grid search...")
        start_time = time.time()
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGrid search completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Results
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    # ========================================================================
    # MODEL PERSISTENCE
    # ========================================================================
    
    def save_model(self, filepath: str = 'random_forest_model.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        print(f"\nSaving model to: {filepath}")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_medians': getattr(self, 'feature_medians', None),
            'performance_metrics': self.performance_metrics,
            'importance_df': self.importance_df
        }
        
        joblib.dump(model_data, filepath)
        print("Model saved successfully!")
    
    @classmethod
    def load_model(cls, filepath: str = 'random_forest_model.pkl') -> 'CreditDefaultRandomForest':
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            CreditDefaultRandomForest: Loaded model instance
        """
        print(f"\nLoading model from: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.feature_medians = model_data.get('feature_medians')
        instance.performance_metrics = model_data['performance_metrics']
        instance.importance_df = model_data.get('importance_df')
        
        print("Model loaded successfully!")
        
        return instance
    
    # ========================================================================
    # PREDICTION ON NEW DATA
    # ========================================================================
    
    def predict_new_data(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X_new (pd.DataFrame): New customer data (same features as training)
            
        Returns:
            np.ndarray: Predicted probabilities of default
        """
        print(f"\nMaking predictions for {len(X_new):,} samples...")
        
        # Ensure features match
        if set(X_new.columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(X_new.columns)
            extra = set(X_new.columns) - set(self.feature_names)
            
            if missing:
                print(f"Warning: Missing features: {missing}")
            if extra:
                print(f"Warning: Extra features will be ignored: {extra}")
        
        # Reorder columns to match training
        X_new = X_new[self.feature_names]
        
        # Fill missing values
        if hasattr(self, 'feature_medians'):
            X_new = X_new.fillna(self.feature_medians)
        else:
            X_new = X_new.fillna(X_new.median())
        
        # Predict
        y_pred_proba = self.model.predict_proba(X_new)[:, 1]
        
        print("Predictions complete!")
        
        return y_pred_proba


# ================================================================================
# MAIN EXECUTION PIPELINE
# ================================================================================

def main():
    """
    Main execution pipeline for Random Forest training and evaluation.
    """
    
    print("="*80)
    print("CREDIT DEFAULT PREDICTION - RANDOM FOREST")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Initialize model
    # ========================================================================
    
    rf_model = CreditDefaultRandomForest(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED
    )
    
    # ========================================================================
    # STEP 2: Load and prepare data
    # ========================================================================
    
    data_file = 'train_data_aggregated_simple.csv'
    
    X_train, X_val, X_test, y_train, y_val, y_test = rf_model.load_and_prepare_data(
        filepath=data_file,
        test_size=0.2,
        val_size=0.1
    )
    
    # ========================================================================
    # STEP 3: Train model
    # ========================================================================
    
    rf_model.train(X_train, y_train)
    
    # ========================================================================
    # STEP 4: Evaluate on validation set
    # ========================================================================
    
    val_metrics = rf_model.evaluate(X_val, y_val, dataset_name="Validation")
    
    # ========================================================================
    # STEP 5: Evaluate on test set (final performance)
    # ========================================================================
    
    test_metrics = rf_model.evaluate(X_test, y_test, dataset_name="Test")
    
    # ========================================================================
    # STEP 6: Visualizations
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # ROC Curve
    rf_model.plot_roc_curve(X_test, y_test, save_path='roc_curve.png')
    
    # Precision-Recall Curve
    rf_model.plot_precision_recall_curve(X_test, y_test, save_path='pr_curve.png')
    
    # Confusion Matrix
    rf_model.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
    
    # ========================================================================
    # STEP 7: Feature Importance Analysis
    # ========================================================================
    
    importance_df = rf_model.analyze_feature_importance(
        top_n=30, 
        save_path='feature_importance.png'
    )
    
    # Save feature importance to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    print("\nFeature importance saved to: feature_importance.csv")
    
    # ========================================================================
    # STEP 8: Save model
    # ========================================================================
    
    rf_model.save_model('random_forest_model.pkl')
    
    # ========================================================================
    # STEP 9: Summary Report
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    print(f"\nModel: Random Forest Classifier")
    print(f"Features: {len(rf_model.feature_names)}")
    print(f"Training samples: {len(X_train):,}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  Test ROC-AUC:       {test_metrics['roc_auc']:.4f}")
    print(f"  Test Precision:     {test_metrics['precision']:.4f}")
    print(f"  Test Recall:        {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score:      {test_metrics['f1_score']:.4f}")
    
    print(f"\nTop 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:40s} ({row['importance']:.6f})")
    
    print(f"\nOutputs generated:")
    print(f"  - random_forest_model.pkl (trained model)")
    print(f"  - feature_importance.csv (feature importance scores)")
    print(f"  - roc_curve.png (ROC curve plot)")
    print(f"  - pr_curve.png (Precision-Recall curve)")
    print(f"  - confusion_matrix.png (confusion matrix heatmap)")
    print(f"  - feature_importance.png (feature importance plots)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return rf_model, test_metrics


# ================================================================================
# OPTIONAL: HYPERPARAMETER TUNING (USE ON SUBSET)
# ================================================================================

def run_hyperparameter_tuning(sample_size: int = 10000):
    """
    Run hyperparameter tuning on a subset of data.
    
    Args:
        sample_size (int): Number of samples to use for tuning
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING ON SUBSET")
    print("="*80)
    
    # Load data
    data = pd.read_csv('train_data_aggregated_simple.csv')
    
    # Sample subset
    data_subset = data.sample(n=min(sample_size, len(data)), random_state=RANDOM_SEED)
    
    X = data_subset.drop(['customer_ID', 'target'], axis=1)
    y = data_subset['target']
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Initialize and tune
    rf_model = CreditDefaultRandomForest()
    rf_model.feature_names = X.columns.tolist()
    
    best_params = rf_model.tune_hyperparameters(X_train, y_train, cv=3)
    
    # Evaluate with best parameters
    metrics = rf_model.evaluate(X_test, y_test, dataset_name="Test (Subset)")
    
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nTest ROC-AUC with best parameters: {metrics['roc_auc']:.4f}")
    
    return best_params


# ================================================================================
# EXAMPLE: MAKING PREDICTIONS ON NEW DATA
# ================================================================================

def predict_on_new_data(model_path: str = 'random_forest_model.pkl',
                       new_data_path: str = 'new_customers.csv'):
    """
    Example of using trained model to predict on new data.
    
    Args:
        model_path (str): Path to saved model
        new_data_path (str): Path to new customer data
    """
    print("\n" + "="*80)
    print("PREDICTING ON NEW DATA")
    print("="*80)
    
    # Load trained model
    rf_model = CreditDefaultRandomForest.load_model(model_path)
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    print(f"\nNew data shape: {new_data.shape}")
    
    # Extract features
    X_new = new_data.drop(['customer_ID'], axis=1)
    customer_ids = new_data['customer_ID']
    
    # Make predictions
    predictions = rf_model.predict_new_data(X_new)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'customer_ID': customer_ids,
        'default_probability': predictions,
        'prediction': (predictions > 0.5).astype(int)
    })
    
    # Sort by probability (highest risk first)
    results = results.sort_values('default_probability', ascending=False)
    
    print("\nTop 10 highest risk customers:")
    print(results.head(10))
    
    # Save predictions
    results.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to: predictions.csv")
    
    return results


# ================================================================================
# RUN THE PIPELINE
# ================================================================================

if __name__ == "__main__":
    
    # Main training pipeline
    rf_model, test_metrics = main()
    
    # Optional: Run hyperparameter tuning on subset
    # Uncomment the line below to run tuning
    # best_params = run_hyperparameter_tuning(sample_size=10000)
    
    # Optional: Make predictions on new data
    # Uncomment the lines below if you have new data
    # predictions = predict_on_new_data(
    #     model_path='random_forest_model.pkl',
    #     new_data_path='new_customers.csv'
    # )