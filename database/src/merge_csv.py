import os
import pandas as pd

def merge_csv_files(folder_path, output_file):
    # List to hold DataFrames
    data_frames = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            data_frames.append(df)

    # Merge all DataFrames
    merged_df = pd.concat(data_frames, ignore_index=True)

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")

# Specify the folder containing CSV files and the output file
folder_path = "./preprocessed/batch"
output_file = "./preprocessed/merged_output.csv"

merge_csv_files(folder_path, output_file)
