import pandas as pd
import os

# Ensure the "evaluation" directory exists
os.makedirs("evaluation", exist_ok=True)

# Function to calculate and predict import dependency by category without combining tables

def predict_import_dependency(imports_test, exports_test, mod = None):
    import joblib
    import pandas as pd
    import numpy as np

    # Aggregate dollar values for imports and exports
    imports_grouped = imports_test.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    exports_grouped = exports_test.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()

    # Merge imports and exports data
    combined_test = pd.concat([imports_grouped, exports_grouped], axis=1, keys=["Imports", "Exports"]).fillna(0)
    combined_test = combined_test.reset_index()  # Include State, Fiscal year, and Fiscal quarter as columns

    # Calculate the import-to-export ratio
    combined_test["Ratio"] = combined_test["Imports"] / combined_test["Exports"]
    combined_test["Ratio"] = combined_test["Ratio"].replace([float("inf"), -float("inf")], 0).fillna(0)

    # Load the trained model and label encoder
    if mod == 'dependency1':
        model_filename = "models/import_dependency_dependency1.joblib"
    elif mod == 'dependency2':
        model_filename = "models/import_dependency_dependency2.joblib"
    elif mod == 'dependency3':
        model_filename = "models/import_dependency_dependency3.joblib"
    elif mod == 'dependency4':
        model_filename = "models/import_dependency_dependency4.joblib"

    model = joblib.load(model_filename)
    label_encoder = joblib.load("models/import_dependency_label_encoder.joblib")

    # Prepare input features (X_test) for the model
    X_test = combined_test[["State", "Fiscal year", "Fiscal quarter", "Ratio"]]
    
    # Save X_test to models folder
    X_test.to_csv(f"models/X_test_import_dependency.csv", index=False)
    
    # Generate the "Encoded Label" column for the test set (like in training)
    bins = [0, 0.75, 2.0, float('inf')]
    labels = ['Low', 'Medium', 'High']
    combined_test['Dependency Level'] = pd.cut(combined_test['Ratio'], bins=bins, labels=labels, include_lowest=True)
    
    # Encode the dependency level
    combined_test["Encoded Label"] = label_encoder.transform(combined_test["Dependency Level"])

    # Prepare target (y_test) using the encoded labels from the model's target
    y_test = combined_test["Encoded Label"]
    
    # Save y_test to models folder
    y_test.to_csv(f"models/y_test_import_dependencys.csv", index=False)

    # Perform predictions
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Add predictions to the results
    combined_test["Predicted Label"] = predicted_labels
    
    if mod == 'dependency1':
        output_filename = "evaluation/import_dependency_predictions_dependency1.csv"
    elif mod == 'dependency2':
        output_filename = "evaluation/import_dependency_predictions_dependency2.csv"
    elif mod == 'dependency3':
        output_filename = "evaluation/import_dependency_predictions_dependency3.csv"
    elif mod == 'dependency4':
        output_filename = "evaluation/import_dependency_predictions_dependency4.csv"
        
    combined_test.to_csv(output_filename, index=False)

def predict_seasonal_fluctuations(data_test, trade_type, mod = None):
    import joblib
    import pandas as pd
        
    def get_season(quarter):
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        return seasons.get(quarter, 'Unknown')

    # Add season and year information to the dataset
    data_test["Season"] = data_test["Fiscal quarter"].apply(get_season)
    
    # Aggregate dollar values by state, category, year, fiscal quarter, and season
    seasonal_data = data_test.groupby(["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"])["Dollar value"].sum().reset_index()

    # Create dummies for categorical features
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"]], drop_first=True)
    
    # Save X_test for seasonal fluctuations to models folder
    seasonal_data_dummies.to_csv(f"models/X_test_seasonal_{trade_type.lower()}.csv", index=False)

    seasonal_data_values = seasonal_data["Dollar value"].values  # Extract the values column separately
    y_test = seasonal_data_values  # Target values for prediction
    
    # Save y_test for seasonal fluctuations to models folder
    pd.DataFrame(y_test, columns=["Dollar value"]).to_csv(f"models/y_test_seasonal_{trade_type.lower()}.csv", index=False)

    # Load the trained model and column names
    if mod == 'seasonal1':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal1.joblib"
    elif mod == 'seasonal2':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal2.joblib"
    elif mod == 'seasonal3':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal3.joblib"
    elif mod == 'seasonal4':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal4.joblib"

    columns_path = f"models/seasonal_columns_{trade_type.lower()}.joblib"  # Match the columns file path
    
    # Load the trained model
    model = joblib.load(model_filename)
    
    # Load the columns used during training to ensure the same features are used in prediction
    with open(columns_path, "rb") as f:
        training_columns = joblib.load(f)
    
    # Align the test data columns with the training columns
    seasonal_data_dummies = seasonal_data_dummies.reindex(columns=training_columns, fill_value=0)
    
    # Prepare the features for prediction
    X_test = seasonal_data_dummies
    
    # Perform predictions
    predictions = model.predict(X_test)
    seasonal_data["Predicted Dollar Value"] = predictions  # Adding predictions to the dataframe

    if mod == 'seasonal1':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal1.csv"
    elif mod == 'seasonal2':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal2.csv"
    elif mod == 'seasonal3':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal3.csv"
    elif mod == 'seasonal4':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal4.csv"
    
    seasonal_data.to_csv(output_filename, index=False)

# Main entry for predictions
if __name__ == "__main__":

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load test data
    imports_test = pd.read_csv("data_processed/import_test.csv")
    exports_test = pd.read_csv("data_processed/export_test.csv")

