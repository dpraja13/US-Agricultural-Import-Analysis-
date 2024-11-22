import pandas as pd
import os

# Ensure the "evaluation" directory exists
os.makedirs("evaluation", exist_ok=True)

# Function to calculate and predict import dependency by category without combining tables
def predict_import_dependency_by_category_separate(imports_test, exports_test):
    import joblib

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
    model = joblib.load("models/import_dependency_by_category_model.joblib")
    label_encoder = joblib.load("models/import_dependency_by_category_label_encoder.joblib")

    # Perform predictions
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Add predictions to the results
    ratios_df["Predicted Label"] = predicted_labels
    return ratios_df

def predict_seasonal_fluctuations_by_category_separate(data_test, trade_type):
    import joblib
        
    def get_season(quarter):
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        return seasons.get(quarter, 'Unknown')

    # Add season information to the dataset
    data_test["Season"] = data_test["Fiscal quarter"].apply(get_season)

    # Aggregate dollar values by state, category, and season
    seasonal_data = data_test.groupby(["State", "Commodity name", "Season"])["Dollar value"].sum().reset_index()  # Reset index here

    # Create dummies for categorical features
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Season"]], drop_first=True)
    
    # Load the trained model and column names
    model_path = f"models/seasonal_{trade_type.lower()}_by_category_model.joblib"
    columns_path = f"models/seasonal_{trade_type.lower()}_by_category_columns.joblib"
    
    model = joblib.load(model_path)
    
    # Load the columns used during training to ensure the same features are used in prediction
    with open(columns_path, "rb") as f:
        training_columns = joblib.load(f)
    
    # Align the test data columns with the training columns
    seasonal_data_dummies = seasonal_data_dummies.reindex(columns=training_columns, fill_value=0)
    
    # Prepare the features for prediction
    X_test = seasonal_data_dummies
    
    # Perform predictions
    predictions = model.predict(X_test)
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
    dependency_predictions.to_csv("evaluation/import_dependency_predictions.csv", index=False)

    # Predict seasonal fluctuations for imports and exports separately
    seasonal_import_predictions = predict_seasonal_fluctuations_by_category_separate(imports_test, "Imports")
    seasonal_import_predictions.to_csv("evaluation/seasonal_imports_predictions.csv", index=False)

    seasonal_export_predictions = predict_seasonal_fluctuations_by_category_separate(exports_test, "Exports")
    seasonal_export_predictions.to_csv("evaluation/seasonal_imports_predictions.csv", index=False)
