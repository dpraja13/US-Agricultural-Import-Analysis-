import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define directories
data_dir = 'data_processed'
output_dir = 'visuals'
summary_file = os.path.join(data_dir, 'summary.txt')
correlations_file = os.path.join(data_dir, 'correlations.txt')

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Define a function to calculate summary statistics
def compute_summary_statistics(df, quantitative_features, qualitative_features, prefix):
    mode = 'w' if prefix == 'Exports' else 'a'  # 'w' for first write, 'a' for subsequent appends
    with open(summary_file, mode) as f:
        f.write(f"Summary Statistics for {prefix}:\n")
        # Summary for quantitative features
        f.write("Quantitative Feature Summary (min, max, median):\n")
        for feature in quantitative_features:
            if feature in df.columns:
                min_val = df[feature].min()
                max_val = df[feature].max()
                median_val = df[feature].median()
                f.write(f"{feature}: Min={min_val}, Max={max_val}, Median={median_val}\n")

        # Summary for qualitative features
        f.write("\nQualitative Feature Summary:\n")
        for feature in qualitative_features:
            if feature in df.columns:
                unique_categories = df[feature].nunique()
                most_frequent = df[feature].mode().values
                least_frequent = df[feature].value_counts().idxmin()
                f.write(f"{feature}: Unique Categories={unique_categories}, Most Frequent={most_frequent}, Least Frequent={least_frequent}\n")
        f.write("\n")  # Add a blank line between summaries

# Define a function to compute pairwise correlations
def compute_correlations(df, quantitative_features, prefix):
    available_features = [feature for feature in quantitative_features if feature in df.columns]
    if available_features:
        correlation_matrix = df[available_features].corr()
        
        # Mask the upper triangular part of the matrix
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        correlation_matrix = correlation_matrix.mask(mask)
        
        # Save the correlation matrix to a text file
        mode = 'w' if prefix == 'Exports' else 'a'  # 'w' for first write, 'a' for subsequent appends
        with open(correlations_file, mode) as f:
            f.write(f"Correlation Matrix (Pairwise) for {prefix}:\n")
            f.write(correlation_matrix.to_string())
            f.write("\n\n")  # Add blank lines between matrices

# Define a function to create scatter plots and histograms
def create_visualizations(df_exports, df_imports, quantitative_features, qualitative_features):
    # Create scatter plots for all pairs of quantitative features
    for i in range(len(quantitative_features)):
        for j in range(i + 1, len(quantitative_features)):
            if (quantitative_features[i] in df_exports.columns and quantitative_features[j] in df_exports.columns and
                quantitative_features[i] in df_imports.columns and quantitative_features[j] in df_imports.columns):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Exports scatter plot
                ax1.scatter(df_exports[quantitative_features[i]], df_exports[quantitative_features[j]], alpha=0.5)
                ax1.set_title(f'Exports: {quantitative_features[i]} vs {quantitative_features[j]}')
                ax1.set_xlabel(quantitative_features[i])
                ax1.set_ylabel(quantitative_features[j])
                
                # Imports scatter plot
                ax2.scatter(df_imports[quantitative_features[i]], df_imports[quantitative_features[j]], alpha=0.5)
                ax2.set_title(f'Imports: {quantitative_features[i]} vs {quantitative_features[j]}')
                ax2.set_xlabel(quantitative_features[i])
                ax2.set_ylabel(quantitative_features[j])
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"scatter_{quantitative_features[i]}_{quantitative_features[j]}.png"))
                plt.close()
    
    # Create histograms for each qualitative feature
    for feature in qualitative_features:
        if feature in df_exports.columns and feature in df_imports.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(45, 10))
            
            # Exports histogram
            df_exports[feature].value_counts().plot(kind='bar', ax=ax1)
            ax1.set_title(f'Exports: Distribution of {feature}')
            ax1.set_xlabel(feature)
            ax1.set_ylabel('Frequency')
            ax1.tick_params(axis='x', rotation=45)
            plt.setp(ax1.get_xticklabels(), ha='right')
            
            # Imports histogram
            df_imports[feature].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title(f'Imports: Distribution of {feature}')
            ax2.set_xlabel(feature)
            ax2.set_ylabel('Frequency')
            ax2.tick_params(axis='x', rotation=45)
            plt.setp(ax2.get_xticklabels(), ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"histogram_{feature}.png"))
            plt.close()

def main():
    # Load the datasets
    exports_file = 'data_processed/Cleaned_Exports.csv'
    imports_file = 'data_processed/Cleaned_Imports.csv'

    df_exports = pd.read_csv(exports_file)
    df_imports = pd.read_csv(imports_file)

    # Specify features to analyze 
    quantitative_features = ['Dollar value', 'Fiscal quarter', 'Fiscal year']
    qualitative_features = ['Commodity name', 'Country']

    # Compute summary statistics for exports and imports
    compute_summary_statistics(df_exports, quantitative_features, qualitative_features, prefix='Exports')
    compute_summary_statistics(df_imports, quantitative_features, qualitative_features, prefix='Imports')

    # Compute pairwise correlations for exports and imports
    compute_correlations(df_exports, quantitative_features, prefix='Exports')
    compute_correlations(df_imports, quantitative_features, prefix='Imports')

    # Create combined visualizations for exports and imports
    create_visualizations(df_exports, df_imports, quantitative_features, qualitative_features)

    print("Summary statistics, correlations, and visualizations generated successfully.")

if __name__ == '__main__':
    main()