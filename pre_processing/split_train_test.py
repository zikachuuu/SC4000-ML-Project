import pandas as pd
import os

# Load original train and test data to get customer_ids
train_original = pd.read_feather(r"S:/ML_Project/new_data/train_data.feather")
test_original = pd.read_feather(r"S:/ML_Project/new_data/test_data.feather")

# Get the customer_ids for train and test
train_ids = set(train_original['customer_ID'])
test_ids = set(test_original['customer_ID'])

# Process each feature-engineered file in the input folder
input_folder        = r"S:/ML_Project/new_data/input/"
output_train_folder = r"S:/ML_Project/new_data/input_train/"
output_test_folder  = r"S:/ML_Project/new_data/input_test/"

# Create output folders if they don't exist
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# Iterate through all feather files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.feather'):
        # Load the feature-engineered file
        filepath = os.path.join(input_folder, filename)
        df = pd.read_feather(filepath)
        
        # Split based on customer_id
        train_df = df[df['customer_ID'].isin(train_ids)]
        test_df = df[df['customer_ID'].isin(test_ids)]
        
        # Save to respective folders
        train_df.to_feather(os.path.join(output_train_folder, filename))
        test_df.to_feather(os.path.join(output_test_folder, filename))
        
        print(f"Processed {filename}: {len(train_df)} train rows, {len(test_df)} test rows")