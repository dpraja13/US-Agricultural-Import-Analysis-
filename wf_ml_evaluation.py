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
    # Load the training and testing data (X_train, y_train, X_test, y_test)
    X_train_import = pd.read_csv("models/X_test_import_dependency.csv")
    y_train_import = pd.read_csv("models/y_test_import_dependency.csv")
    X_test_import = pd.read_csv("models/X_test_import_dependency.csv")
    y_test_import = pd.read_csv("models/y_test_import_dependency.csv")

    X_train_seasonal = pd.read_csv("models/X_test_seasonal_imports.csv")
    y_train_seasonal = pd.read_csv("models/y_test_seasonal_imports.csv")
    X_test_seasonal = pd.read_csv("models/X_test_seasonal_imports.csv")
    y_test_seasonal = pd.read_csv("models/y_test_seasonal_imports.csv")

    # Load models
    import_dependency_model = joblib.load("models/import_dependency.joblib")
    seasonal_model = joblib.load("models/seasonal_imports.joblib")

    # Initialize result string to store evaluation metrics
    result_str = ""

    ### 1. Metrics for Import Dependency Model (Classification)
    # Predict the labels for the test set
    y_pred_import = import_dependency_model.predict(X_test_import)

    # Calculate accuracy, precision, recall, f1-score
    accuracy = accuracy_score(y_test_import, y_pred_import)
    precision = precision_score(y_test_import, y_pred_import, average='weighted', zero_division=0)
    recall = recall_score(y_test_import, y_pred_import, average='weighted')
    f1 = f1_score(y_test_import, y_pred_import, average='weighted')

    result_str += "### Import Dependency Model (Classification) Metrics ###\n"
    result_str += f"Accuracy: {accuracy:.4f}\n"
    result_str += f"Precision (Weighted): {precision:.4f}\n"
    result_str += f"Recall (Weighted): {recall:.4f}\n"
    result_str += f"F1 Score (Weighted): {f1:.4f}\n\n"

    ### 2. Metrics for Seasonal Model (Regression)
    # Predict the dollar values for the test set
    y_pred_seasonal = seasonal_model.predict(X_test_seasonal)

    # Calculate RMSE, MAE, and R-squared
    rmse = np.sqrt(mean_squared_error(y_test_seasonal, y_pred_seasonal))
    mae = mean_absolute_error(y_test_seasonal, y_pred_seasonal)
    r2 = r2_score(y_test_seasonal, y_pred_seasonal)

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



    
