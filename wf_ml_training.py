import pandas as pd
import numpy as np
import os

# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)

# Function 1: Classify import dependency by category
def classify_import_dependency_by_category(imports, exports):
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    # Aggregate dollar values by state and category (Commodity name)
    imports_grouped = imports.groupby(["State", "Commodity name"])["Dollar value"].sum()
    exports_grouped = exports.groupby(["State", "Commodity name"])["Dollar value"].sum()
    
    # Merge import and export data
    combined = pd.concat([imports_grouped, exports_grouped], axis=1, keys=["Imports", "Exports"]).fillna(0)
    
    # Calculate import-to-export ratio
    combined["Ratio"] = combined["Imports"] / combined["Exports"]
    combined["Ratio"] = combined["Ratio"].replace([np.inf, -np.inf], 0)
    
    # Label states and categories as high, medium, or low dependency
    try:
        labels = pd.qcut(combined["Ratio"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    except ValueError:
        # Fall back to manual binning if qcut fails
        bins = [0, 0.1, 1, combined["Ratio"].max()]
        labels = pd.cut(combined["Ratio"], bins=bins, labels=["Low", "Medium", "High"], include_lowest=True)

    combined["Label"] = labels

    # Encode labels
    le = LabelEncoder()
    combined["Encoded Label"] = le.fit_transform(combined["Label"])
    
    # Train a classifier
    X = combined[["Ratio"]]
    y = combined["Encoded Label"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Save the model and label encoder
    with open("models/import_dependency_by_category_model.joblib", "wb") as f:
        joblib.dump(model, f)
    with open("models/import_dependency_by_category_label_encoder.joblib", "wb") as f:
        joblib.dump(le, f)

# Function 2: Analyze seasonal fluctuations by category
def analyze_seasonal_fluctuations_by_category(data, trade_type):
    import joblib
    from sklearn.tree import DecisionTreeRegressor
    
    def get_season(quarter):
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        return seasons.get(quarter, 'Unknown')

    data["Season"] = data["Fiscal quarter"].apply(get_season)
    
    # Aggregate by state, category, and season
    seasonal_data = data.groupby(["State", "Commodity name", "Season"])["Dollar value"].sum().reset_index()  # Reset index here
    
    # Create dummies for categorical features
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Season"]], drop_first=True)
    
    seasonal_data_values = seasonal_data["Dollar value"].values  # Extract the values column separately
    
    # Train a model to predict seasonal dollar values
    X = seasonal_data_dummies
    y = seasonal_data_values
    
    model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X, y)
    
    # Save the model and the feature column names used during training
    filename = f"models/seasonal_{trade_type.lower()}_by_category_model.joblib"
    with open(filename, "wb") as f:
        joblib.dump(model, f)
    
    # Save the column names
    columns_filename = f"models/seasonal_{trade_type.lower()}_by_category_columns.joblib"
    with open(columns_filename, "wb") as f:
        joblib.dump(seasonal_data_dummies.columns, f)


# Main entry for testing
if __name__ == "__main__":
    # Load training data
    imports_train = pd.read_csv("data_processed/import_train.csv")
    exports_train = pd.read_csv("data_processed/export_train.csv")
    
    # Call function 1
    classify_import_dependency_by_category(imports_train, exports_train)
    
    # Call function 2 for imports and exports separately
    analyze_seasonal_fluctuations_by_category(imports_train, "Imports")
    analyze_seasonal_fluctuations_by_category(exports_train, "Exports")
