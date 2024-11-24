from sklearn.utils import shuffle
import os
from sklearn.metrics import mean_absolute_error, median_absolute_error, accuracy_score
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def split_data(import_csv, export_csv):
    imports = pd.read_csv(import_csv)
    exports = pd.read_csv(export_csv)

    shuffle_imports = shuffle(imports)
    shuffle_exports = shuffle(exports)

    samples_im = len(shuffle_imports)
    training_im = int(samples_im * 0.8)
    import_train = shuffle_imports[:training_im]
    import_test = shuffle_imports[training_im:]

    samples_ex = len(shuffle_exports)
    training_ex = int(samples_ex * 0.8)
    export_train = shuffle_exports[:training_ex]
    export_test = shuffle_exports[training_ex:]

    import_train.to_csv(os.path.join('data_processed', 'import_train.csv'), index=False)
    import_test.to_csv(os.path.join('data_processed', 'import_test.csv'), index=False)
    export_train.to_csv(os.path.join('data_processed', 'export_train.csv'), index=False)
    export_test.to_csv(os.path.join('data_processed', 'export_test.csv'), index=False)

def train_and_predict():
    with open("wf_ml_training.py") as trainer:
        exec(trainer.read())

    with open("wf_ml_prediction.py") as predictor:
        exec(predictor.read())

def model_evaluation():
    # Load the saved model results from CSV files
    import_results = pd.read_csv("evaluation/import_dependency_predictions.csv")
    seasonal_results = pd.read_csv("evaluation/seasonal_imports_predictions.csv")

    # Initialize result string to store evaluation metrics
    result_str = ""

    ### 1. Metrics for Import Dependency Model (Classification)
    # Get true labels and predicted labels
    y_true_import = import_results['Encoded Label']
    y_pred_import = import_results['Predicted Label']

    # Mapping categorical labels to numeric labels (if needed)
    label_mapping = {"Low": 0, "Medium": 1, "High": 2}
    y_true_import = y_true_import.map(label_mapping)
    y_pred_import = y_pred_import.map(label_mapping)

    # Check for any missing values after mapping
    if y_true_import.isnull().any() or y_pred_import.isnull().any():
        print("Warning: There are missing values after mapping labels.")
        y_true_import = y_true_import.fillna(0)  # Fill with default value if needed
        y_pred_import = y_pred_import.fillna(0)

    # Calculate accuracy, precision, recall, f1-score
    accuracy = accuracy_score(y_true_import, y_pred_import)
    precision = precision_score(y_true_import, y_pred_import, average='weighted', zero_division=0)
    recall = recall_score(y_true_import, y_pred_import, average='weighted')
    f1 = f1_score(y_true_import, y_pred_import, average='weighted')

    result_str += "### Import Dependency Model (Classification) Metrics ###\n"
    result_str += f"Accuracy: {accuracy:.4f}\n"
    result_str += f"Precision (Weighted): {precision:.4f}\n"
    result_str += f"Recall (Weighted): {recall:.4f}\n"
    result_str += f"F1 Score (Weighted): {f1:.4f}\n\n"

    ### 2. Metrics for Seasonal Model (Regression)
    # Get true values and predicted values for dollar values
    y_true_seasonal = seasonal_results['Dollar value']
    y_pred_seasonal = seasonal_results['Predicted Dollar Value']

    # Calculate RMSE, MAE, and R-squared
    rmse = np.sqrt(mean_squared_error(y_true_seasonal, y_pred_seasonal))
    mae = mean_absolute_error(y_true_seasonal, y_pred_seasonal)
    r2 = r2_score(y_true_seasonal, y_pred_seasonal)

    result_str += "### Seasonal Model (Regression) Metrics ###\n"
    result_str += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
    result_str += f"Mean Absolute Error (MAE): {mae:.4f}\n"
    result_str += f"R-squared: {r2:.4f}\n"

    # Save results to a file
    with open("evaluation/summary.txt", "w") as f:
        f.write(result_str)

    print("Evaluation metrics have been saved to 'evaluation/summary.txt'")

def alternative_models():
    #code for the alternative models possible
    x = 2

if __name__ == "__main__":
    import_data = os.path.join("data_processed","Cleaned_Imports.csv")
    export_data = os.path.join("data_processed","Cleaned_Exports.csv")

    split_data(import_data, export_data)
    train_and_predict()
    model_evaluation()
    #alternative_models()



    
