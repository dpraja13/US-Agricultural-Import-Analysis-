import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
	# Define input and output folders
	input_folder = "data_processed"
	output_folder = "visuals"

	# Create the output folder if it doesn't exist
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Load the processed data
	df = pd.read_csv(os.path.join(input_folder, "processed_data.csv"))

	# Select features for analysis
	quantitative_features = ['Dollar value', 'Fiscal year', 'Fiscal quarter']
	qualitative_features = ['State', 'Commodity name']

	# 1. Compute summary statistics
	def compute_summary_statistics(df, quant_features, qual_features):
		summary = []
		
		for feature in quant_features:
			summary.append(f"{feature}:")
			summary.append(f"  Min: {df[feature].min()}")
			summary.append(f"  Max: {df[feature].max()}")
			summary.append(f"  Median: {df[feature].median()}")
			summary.append("")
		
		for feature in qual_features:
			value_counts = df[feature].value_counts()
			summary.append(f"{feature}:")
			summary.append(f"  Number of categories: {len(value_counts)}")
			summary.append(f"  Most frequent category: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
			summary.append(f"  Least frequent category: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
			summary.append("")
		
		return "\n".join(summary)

	summary_stats = compute_summary_statistics(df, quantitative_features, qualitative_features)
	with open(os.path.join(input_folder, "summary.txt"), "w") as f:
		f.write(summary_stats)

	# 2. Compute pairwise correlations
	correlation_matrix = df[quantitative_features].corr()
	correlation_output = correlation_matrix.to_string()
	with open(os.path.join(input_folder, "correlations.txt"), "w") as f:
		f.write(correlation_output)

	# 3. Plots of distributions
	def create_scatter_plots(df, quant_features):
		for i in range(len(quant_features)):
			for j in range(i+1, len(quant_features)):
				feature1 = quant_features[i]
				feature2 = quant_features[j]
				plt.figure(figsize=(10, 6))
				plt.scatter(df[feature1], df[feature2], alpha=0.5)
				plt.xlabel(feature1)
				plt.ylabel(feature2)
				plt.title(f"Scatter plot: {feature1} vs {feature2}")
				plt.savefig(os.path.join(output_folder, f"scatter_{feature1}_{feature2}.png"))
				plt.close()

	def create_histograms(df, qual_features):
		for feature in qual_features:
			plt.figure(figsize=(12, 6))
			df[feature].value_counts().plot(kind='bar')
			plt.xlabel(feature)
			plt.ylabel("Count")
			plt.title(f"Histogram: {feature}")
			plt.xticks(rotation=45, ha='right')
			plt.tight_layout()
			plt.savefig(os.path.join(output_folder, f"histogram_{feature}.png"))
			plt.close()

	create_scatter_plots(df, quantitative_features)
	create_histograms(df, qualitative_features)

	print("Visualization completed. Output saved in 'data_processed' and 'visuals' folders.")

if __name__ == "__main__":
    main()