import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import re
import warnings
import os
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    def __init__(self, benchmark_scores):
        """
        Initialize the ensemble optimizer
        
        Parameters:
        -----------
        benchmark_scores : dict
            Dictionary with keys: 'top_70', 'top_50', 'top_30', 'top_10'
            representing the public score thresholds
        """
        self.benchmark_scores = benchmark_scores
        # Use 'name' as the unique identifier for each prediction file.
        # Keep the original full file path as provided by the user.
        # Data structures:
        # - self.names: ordered list of names (unique, user-provided)
        # - self.filepaths: dict name -> full filepath (string)
        # - self.scores: dict name -> public score (float)
        # - self.predictions_data: dict name -> pd.Series(index=customer_ID)
        self.names = []
        self.filepaths = {}
        self.scores = {}
        self.predictions_data = {}
        
    def add_prediction_file(self, filepath, public_score, name):
        """
        Add a prediction file with its public score
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file with columns: customer_ID, prediction
        public_score : float
            The public score achieved by this prediction file
        name : str
            Optional name for the prediction file
        """
        if name in self.names:
            raise ValueError(f"Prediction file with name '{name}' already added.")

        df = pd.read_csv(filepath)

        # Validate the dataframe
        if 'customer_ID' not in df.columns or 'prediction' not in df.columns:
            raise ValueError(f"File {filepath} must have 'customer_ID' and 'prediction' columns")

        if not all((df['prediction'] >= 0) & (df['prediction'] <= 1)):
            raise ValueError(f"Predictions in {filepath} must be between 0 and 1")

        # Store the data using the provided 'name' as the unique key and keep full path
        self.names.append(name)
        self.filepaths[name] = str(filepath)
        self.scores[name] = float(public_score)
        # store Series indexed by customer_ID for easy alignment
        self.predictions_data[name] = df.set_index('customer_ID')['prediction']

        print(f"Added: {name} (file: {filepath}, score: {public_score:.6f})")
        
    def calculate_performance_tier(self, score):
        """Calculate which performance tier a score falls into"""
        if score >= self.benchmark_scores['top_10']:
            return 'top_10'
        elif score >= self.benchmark_scores['top_30']:
            return 'top_30'
        elif score >= self.benchmark_scores['top_50']:
            return 'top_50'
        elif score >= self.benchmark_scores['top_70']:
            return 'top_70'
        else:
            return 'below_70'
    
    def calculate_weights(self, selected_files, method='rank'):
        """
        Calculate weights for selected files based on their scores
        
        Parameters:
        -----------
        selected_files : list
            List of file names to ensemble
        method : str or dict
            Weighting method: 'rank', 'score', 'exponential', 'uniform'
            Or a dict mapping file names to weights
        """
        # If method is a dictionary, use custom weights (keys should be names)
        if isinstance(method, dict):
            weights = np.array([method.get(f, 1.0) for f in selected_files])
            weights = weights / weights.sum()
            return weights

        # selected_files should be a list of 'name' identifiers
        selected_scores = [self.scores[f] for f in selected_files]
        
        if method == 'uniform':
            weights = np.ones(len(selected_files)) / len(selected_files)
        
        elif method == 'rank':
            # Better scores get higher weights
            ranks = np.argsort(np.argsort(selected_scores))  # 0 for worst, n-1 for best
            weights = (ranks + 1) / (ranks + 1).sum()
        
        elif method == 'score':
            # Direct proportional to score
            weights = np.array(selected_scores)
            weights = weights / weights.sum()
        
        elif method == 'exponential':
            # Exponential weighting favoring top performers
            weights = np.exp(np.array(selected_scores) * 10)
            weights = weights / weights.sum()
        
        elif method == 'softmax':
            # Softmax weighting (more aggressive than exponential)
            weights = np.exp(np.array(selected_scores) * 100)
            weights = weights / weights.sum()
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return weights
    
    def create_ensemble(self, selected_files, weights, output_path):
        """
        Create an ensemble prediction file
        
        Parameters:
        -----------
        selected_files : list
            List of file names to ensemble
        weights : array-like
            Weights for each file
        output_path : str
            Path to save the ensemble file
        """
        # Get all predictions aligned by customer_ID
        ensemble_pred = None
        
        for file_name, weight in zip(selected_files, weights):
            pred = self.predictions_data[file_name]
            if ensemble_pred is None:
                ensemble_pred = pred * weight
            else:
                ensemble_pred = ensemble_pred + pred * weight
        
        # Create output dataframe
        result_df = pd.DataFrame({
            'customer_ID': ensemble_pred.index,
            'prediction': ensemble_pred.values
        })
        
        result_df.to_csv(output_path, index=False)
        print(f"Ensemble saved to: {output_path}")
        
        return result_df
    
    def create_custom_ensemble(self, file_weights, output_name, output_dir='.'):
        """
        Create a custom ensemble with specific files and weights
        
        Parameters:
        -----------
        file_weights : dict
            Dictionary mapping file names (or indices) to weights
            Example: {'nn_simple': 0.5, 'gbm_simple': 0.5}
            Or: {0: 0.5, 1: 0.5}  # using indices
        output_name : str
            Name for the output file
        output_dir : str
            Directory to save the output
        """
        # Convert indices to file names if needed
        if all(isinstance(k, int) for k in file_weights.keys()):
            file_weights = {self.names[k]: v for k, v in file_weights.items()}
        
        # Get the actual file names (not display names)
        selected_files = []
        weights = []
        
        for name, weight in file_weights.items():
            if name in self.names:
                # store the name (unique id) instead of any stem
                selected_files.append(name)
                weights.append(weight)
            else:
                raise ValueError(f"Unknown file name: {name}")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Build filename
        name_weight_parts = []
        print(f"\nCustom Ensemble: {output_name}")
        print(f"Files used: {len(selected_files)}")
        
        for f, w in zip(selected_files, weights):
            # f is the user-provided name
            score = self.scores[f]
            display_name = f
            safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(display_name))
            name_weight_parts.append(f"{safe_name}-{w:.4f}")
            print(f"  - {display_name}: weight={w:.4f}, score={score:.6f}")
        
        filename = f"ensemble_{output_name}_{'_'.join(name_weight_parts)}.csv"
        output_path = os.path.join(output_dir, filename)
        
        self.create_ensemble(selected_files, weights, output_path)
        return output_path
    
    def generate_all_combinations(self, min_files=2, max_files=None, 
                                  weight_methods=None, output_dir='.'):
        """
        Generate ALL possible combinations with different weighting methods
        
        Parameters:
        -----------
        min_files : int
            Minimum number of files to combine
        max_files : int
            Maximum number of files to combine
        weight_methods : list
            List of weighting methods to try
        output_dir : str
            Directory to save outputs
        """
        if max_files is None:
            max_files = len(self.names)
        
        if weight_methods is None:
            weight_methods = ['uniform', 'rank', 'score', 'exponential']
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"GENERATING ALL COMBINATIONS")
        print(f"{'='*80}\n")
        
        count = 0
        
        for num_files in range(min_files, max_files + 1):
            for combo in combinations(range(len(self.names)), num_files):
                # selected identifiers are the user-provided names
                combo_files = [self.names[i] for i in combo]
                combo_names = combo_files
                
                for method in weight_methods:
                    count += 1
                    weights = self.calculate_weights(combo_files, method)
                    
                    # Build filename
                    name_weight_parts = []
                    print(f"\n{count}. Combo: {num_files} files, Method: {method}")
                    
                    for f, w in zip(combo_files, weights):
                        score = self.scores[f]
                        display_name = f
                        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(display_name))
                        name_weight_parts.append(f"{safe_name}-{w:.4f}")
                        print(f"  - {display_name}: weight={w:.4f}, score={score:.6f}")
                    
                    filename = f"ensemble_{method}_{'_'.join(name_weight_parts)}.csv"
                    output_path = os.path.join(output_dir, filename)
                    
                    self.create_ensemble(combo_files, weights, output_path)
        
        print(f"\n{'='*80}")
        print(f"Generated {count} ensemble files")
        print(f"{'='*80}\n")
    
    def generate_best_ensembles(self, num_outputs=5, min_files=2, max_files=None, 
                               output_dir='.', skip_duplicates=True,
                               extra_methods=None):
        """
        Generate the best ensemble combinations
        
        Parameters:
        -----------
        num_outputs : int
            Number of different ensemble files to generate
        min_files : int
            Minimum number of files to combine
        max_files : int
            Maximum number of files to combine (None = all files)
        output_dir : str
            Directory to save outputs
        skip_duplicates : bool
            Whether to skip duplicate file+method combinations
        extra_methods : list
            Additional weighting methods to try
        """
        if max_files is None:
            max_files = len(self.names)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"GENERATING {num_outputs} BEST ENSEMBLES")
        print(f"{'='*80}\n")
        
        strategies = []
        
        # Get top performers (names sorted by score desc)
        sorted_names = sorted(self.names, key=lambda n: self.scores[n], reverse=True)
        top_3 = sorted_names[:min(3, len(sorted_names))]
        top_2 = sorted_names[:min(2, len(sorted_names))]
        
        # Strategy 1: Top 2 with different methods
        for method in ['rank', 'exponential', 'uniform', 'score']:
            strategies.append((f'top_2_{method}', top_2, method))
        
        # Strategy 2: Top 3 with different methods
        if len(top_3) >= min_files:
            for method in ['rank', 'exponential', 'uniform', 'softmax']:
                strategies.append((f'top_3_{method}', top_3, method))
        
        # Strategy 3: All files with different methods
        if len(self.names) >= min_files:
            for method in ['rank', 'exponential', 'uniform']:
                strategies.append((f'all_files_{method}', list(self.names), method))
        
        # Strategy 4: Best combinations of each size
        for combo_size in range(min_files, min(max_files + 1, len(self.names) + 1)):
            best_combo = None
            best_avg_score = -1
            
            for combo in combinations(range(len(self.names)), combo_size):
                avg_score = np.mean([self.scores[self.names[i]] for i in combo])
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_combo = combo
            
            combo_files = [self.names[i] for i in best_combo]
            for method in ['rank', 'exponential']:
                strategies.append((f'best_{combo_size}_combo_{method}', combo_files, method))
        
        # Add extra methods if provided
        if extra_methods:
            for method in extra_methods:
                strategies.append((f'all_files_{method}', list(self.names), method))
        
        # Generate outputs
        output_count = 0
        generated_strategies = set()
        
        for strategy_name, selected_files, weight_method in strategies:
            if output_count >= num_outputs:
                break
            
            # Avoid duplicate strategies if enabled
            if skip_duplicates:
                strategy_key = tuple(sorted(selected_files)) + (weight_method,)
                if strategy_key in generated_strategies:
                    continue
                generated_strategies.add(strategy_key)
            
            weights = self.calculate_weights(selected_files, weight_method)

            # Build filename
            name_weight_parts = []
            print(f"\nEnsemble {output_count + 1}: {strategy_name}")
            print(f"Files used: {len(selected_files)}")
            
            for f, w in zip(selected_files, weights):
                # f is the user-provided name; file path is preserved in self.filepaths
                score = self.scores[f]
                display_name = f
                safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(display_name))
                name_weight_parts.append(f"{safe_name}-{w:.4f}")
                file_path = self.filepaths.get(f, '<unknown path>')
                print(f"  - {display_name} (file: {file_path}): weight={w:.4f}, score={score:.6f}")

            prefix = f"ensemble_{strategy_name}"
            filename = prefix + "_" + "_".join(name_weight_parts) + ".csv"
            output_path = os.path.join(output_dir, filename)

            self.create_ensemble(selected_files, weights, output_path)
            output_count += 1
        
        print(f"\n{'='*80}")
        print(f"Generated {output_count} ensemble files")
        print(f"{'='*80}\n")


# Example usage:
if __name__ == "__main__":
    # Define competition benchmarks
    benchmarks = {
        'top_70': 0.78901,
        'top_50': 0.79617,
        'top_30': 0.79967,
        'top_10': 0.79990
    }
    
    # Initialize optimizer
    optimizer = EnsembleOptimizer(benchmarks)
    
    # Add prediction files
    optimizer.add_prediction_file(
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2\lightgbm_test_predictions.csv", 
        0.78831, 
        "gbm_simple"
    )
    optimizer.add_prediction_file(
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 2 Ensembled\final_submission_nn.csv", 
        0.78874, 
        "nn_simple"
    )
    optimizer.add_prediction_file(
        r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 3\lightgbm_test_predictions.csv", 
        0.78313, 
        "gbm_time_no_cv"
    )
    
    output_folder = r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\Run 3 Ensembled smart"
    
    # Option 1: Generate smart ensembles (NO duplicates by default, more diverse strategies)
    optimizer.generate_best_ensembles(
        num_outputs=10, 
        min_files=2, 
        max_files=None,
        output_dir=output_folder,
        skip_duplicates=False  # Set to False to get all 10
    )
    
    # Option 2: Generate ALL possible combinations (can create many files!)
    # optimizer.generate_all_combinations(
    #     min_files=2,
    #     max_files=3,
    #     weight_methods=['uniform', 'rank', 'exponential', 'softmax'],
    #     output_dir=output_folder
    # )
    
    # Option 3: Create custom ensemble with specific weights
    # optimizer.create_custom_ensemble(
    #     file_weights={'nn_simple': 0.3, 'gbm_simple': 0.7},
    #     output_name='custom_70_30',
    #     output_dir=output_folder
    # )x