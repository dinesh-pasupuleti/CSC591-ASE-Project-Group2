import pandas as pd
import os

# List of dataset filenames
dataset_filenames = ['SS-A.csv', 'SS-C.csv']

# Initialize an empty list to store dataframes
dfs = []

# Read each dataset and append to the list
for filename in dataset_filenames:
    filepath = os.path.join(
        '/Users/dineshpasupuleti/CSC591-ASE-Project-Group2/data/', filename
    )  # Adjust path as needed
    df = pd.read_csv(filepath)
    dfs.append(df)

# Concatenate all dataframes in the list along rows (axis=0)
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_filepath = 'SS2.csv'
combined_df.to_csv(combined_filepath, index=False)

print(f"Combined dataset saved to {combined_filepath}")
