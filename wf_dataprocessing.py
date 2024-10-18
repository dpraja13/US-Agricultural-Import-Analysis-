from pathlib import Path
import os
import pandas as pd

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to clean the dataset
def clean_data(input_file, output_file):
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Handle missing values (drop rows with missing values for simplicity)
    df = df.dropna() 
    
    # 3. Correct data types 
    # Convert "Fiscal year" to integer
    if 'Fiscal year' in df.columns:
        df['Fiscal year'] = pd.to_numeric(df['Fiscal year'], errors='coerce')
    
    # Convert "Fiscal quarter" to integer (assuming it's in the dataset)
    if 'Fiscal quarter' in df.columns:
        df['Fiscal quarter'] = pd.to_numeric(df['Fiscal quarter'], errors='coerce')
    
    # Convert "Dollar value" to float
    if 'Dollar value' in df.columns:
        df['Dollar value'] = pd.to_numeric(df['Dollar value'], errors='coerce')
    
    # 4. Standardize categorical columns (trim spaces and ensure consistency)
    if 'Country' in df.columns:
        df['Country'] = df['Country'].str.strip().str.title()
    
    if 'Commodity name' in df.columns:
        df['Commodity name'] = df['Commodity name'].str.strip().str.title()

    # 5. Remove rows where fiscal quarter is 0
    if 'Fiscal quarter' in df.columns:
        df = df[df['Fiscal quarter'] != 0]

    # 6. Save the cleaned data to the output file
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

def main():
	# Define file paths
    input_exports_file = 'data_original/Top%205%20Agricultural%20Exports%20by%20State.csv'
    input_imports_file = 'data_original/Top%205%20Agricultural%20Imports%20by%20State.csv'
    
    output_exports_file = 'data_processed/Cleaned_Exports.csv'
    output_imports_file = 'data_processed/Cleaned_Imports.csv'
    
    # Clean both datasets
    clean_data(input_exports_file, output_exports_file)
    clean_data(input_imports_file, output_imports_file)

if __name__ == '__main__':
    create_directory(Path('data_original'))
    create_directory(Path('data_processed'))
    main()