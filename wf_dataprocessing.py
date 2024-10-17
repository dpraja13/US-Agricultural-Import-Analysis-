import pandas as pd
import os

def main():
	input_folder = "data_original"
	output_folder = "data_processed"
	output_file = "processed_data.csv"

	# Create the output folder if it doesn't exist
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Initialize an empty list to store dataframes
	dataframes = []

	# Read all CSV files in the input folder
	for filename in os.listdir(input_folder):
		if filename.endswith(".csv"):
			file_path = os.path.join(input_folder, filename)
			df = pd.read_csv(file_path)
			dataframes.append(df)

	# Combine all dataframes
	combined_df = pd.concat(dataframes, ignore_index=True)

	# Remove duplicate rows if any
	combined_df = combined_df.drop_duplicates()

	# Save the combined dataframe to a new CSV file in the output folder
	output_path = os.path.join(output_folder, output_file)
	combined_df.to_csv(output_path, index=False)

	print(f"Combined CSV file saved to: {output_path}")

if __name__ == "__main__":
    main()