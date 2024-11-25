import pandas as pd
import os

os.makedirs("evaluation", exist_ok=True)

def predict_import_dependency(imports_test, exports_test, mod = None):
    import joblib
    import pandas as pd
    import numpy as np

    imports_grouped = imports_test.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    exports_grouped = exports_test.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()

    combined_test = pd.concat([imports_grouped, exports_grouped], axis=1, keys=["Imports", "Exports"]).fillna(0)
    combined_test = combined_test.reset_index()

    combined_test["Ratio"] = combined_test["Imports"] / combined_test["Exports"]
    combined_test["Ratio"] = combined_test["Ratio"].replace([float("inf"), -float("inf")], 0).fillna(0)

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

    X_test = combined_test[["State", "Fiscal year", "Fiscal quarter", "Ratio"]]
        
    bins = [0, 0.75, 2.0, float('inf')]
    labels = ['Low', 'Medium', 'High']
    combined_test['Dependency Level'] = pd.cut(combined_test['Ratio'], bins=bins, labels=labels, include_lowest=True)
    
    combined_test["Encoded Label"] = label_encoder.transform(combined_test["Dependency Level"])
    
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

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

    data_test["Season"] = data_test["Fiscal quarter"].apply(get_season)
    
    seasonal_data = data_test.groupby(["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"])["Dollar value"].sum().reset_index()
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"]], drop_first=True)
        
    if mod == 'seasonal1':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal1.joblib"
    elif mod == 'seasonal2':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal2.joblib"
    elif mod == 'seasonal3':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal3.joblib"
    elif mod == 'seasonal4':
        model_filename = f"models/seasonal_{trade_type.lower()}_seasonal4.joblib"

    columns_path = f"models/seasonal_columns_{trade_type.lower()}.joblib"  
    
    model = joblib.load(model_filename)
    
    with open(columns_path, "rb") as f:
        training_columns = joblib.load(f)
    
    seasonal_data_dummies = seasonal_data_dummies.reindex(columns=training_columns, fill_value=0)
    X_test = seasonal_data_dummies
    
    predictions = model.predict(X_test)
    seasonal_data["Predicted Dollar Value"] = predictions  

    if mod == 'seasonal1':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal1.csv"
    elif mod == 'seasonal2':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal2.csv"
    elif mod == 'seasonal3':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal3.csv"
    elif mod == 'seasonal4':
        output_filename = f"evaluation/seasonal_{trade_type.lower()}_predictions_seasonal4.csv"
    
    seasonal_data.to_csv(output_filename, index=False)

if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)
    imports_test = pd.read_csv("data_processed/import_test.csv")
    exports_test = pd.read_csv("data_processed/export_test.csv")