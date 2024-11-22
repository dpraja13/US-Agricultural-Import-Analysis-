import pandas as pd
import pickle
import os
from wf_ml_training import get_season

# Function to load a saved model
def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Function to calculate and predict import dependency by category without combining tables
def predict_import_dependency_by_category_separate(imports_test, exports_test):
    # Aggregate dollar values for imports and exports separately
    imports_grouped = imports_test.groupby(["State", "Commodity name"])["Dollar value"].sum()
    exports_grouped = exports_test.groupby(["State", "Commodity name"])["Dollar value"].sum()

    # Create a consistent index for all states and categories
    all_states_categories = set(imports_grouped.index).union(exports_grouped.index)

    # Calculate the import-to-export ratio
    ratios = {}
    for state_category in all_states_categories:
        imports_value = imports_grouped.get(state_category, 0)
        exports_value = exports_grouped.get(state_category, 0)
        ratio = imports_value / exports_value if exports_value != 0 else 0
        ratios[state_category] = ratio

    # Prepare input for the model
    ratios_df = pd.DataFrame(list(ratios.items()), columns=["State_Category", "Ratio"])
    X_test = ratios_df[["Ratio"]]

    # Load the trained model and label encoder
    model = load_model("models/import_dependency_by_category_model.pkl")
    label_encoder = load_model("models/import_dependency_by_category_label_encoder.pkl")

    # Perform predictions
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Add predictions to the results
    ratios_df["Predicted Label"] = predicted_labels
    return ratios_df

# Function to analyze and predict seasonal fluctuations without combining tables
def predict_seasonal_fluctuations_by_category_separate(data_test, trade_type):
    # Add season information to the dataset
    data_test["Season"] = data_test["Fiscal quarter"].apply(get_season)

    # Aggregate dollar values by state, category, and season
    seasonal_data = data_test.groupby(["State", "Commodity name", "Season"])["Dollar value"].sum()

    # Prepare test features
    X_test = pd.get_dummies(seasonal_data.index, drop_first=True)

    # Load the trained model
    model_path = f"models/seasonal_{trade_type.lower()}_by_category_model.pkl"
    model = load_model(model_path)

    # Perform predictions
    predictions = model.predict(X_test)
    seasonal_data = seasonal_data.reset_index()  # Flatten the index for readable output
    seasonal_data["Predicted Season Index"] = predictions

    return seasonal_data

# Main entry for predictions
if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load test data
    imports_test = pd.read_csv("data_processed/import_test.csv")
    exports_test = pd.read_csv("data_processed/export_test.csv")

    # Predict import-export dependency
    dependency_predictions = predict_import_dependency_by_category_separate(imports_test, exports_test)
    print("Import-Export Dependency Predictions (Separate):")
    print(dependency_predictions.head())

    # Predict seasonal fluctuations for imports and exports separately
    seasonal_import_predictions = predict_seasonal_fluctuations_by_category_separate(imports_test, "Imports")
    seasonal_export_predictions = predict_seasonal_fluctuations_by_category_separate(exports_test, "Exports")

    print("\nSeasonal Import Predictions:")
    print(seasonal_import_predictions.head())

    print("\nSeasonal Export Predictions:")
    print(seasonal_export_predictions.head())
