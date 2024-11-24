import pandas as pd
import numpy as np
import os

# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)

# Function 1: Classify import dependency by category
def classify_import_dependency(imports, exports):
    import joblib
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np

    # Aggregate dollar values by state, fiscal year, and fiscal quarter
    imports_grouped = imports.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    exports_grouped = exports.groupby(["State", "Fiscal year", "Fiscal quarter"])["Dollar value"].sum()
    
    # Merge import and export data
    combined = pd.concat([imports_grouped, exports_grouped], axis=1, keys=["Imports", "Exports"]).fillna(0)

    # Reset index to include "State", "Fiscal year", and "Fiscal quarter" as columns
    combined = combined.reset_index()

    # Calculate import-to-export ratio
    combined["Ratio"] = combined["Imports"] / combined["Exports"]
    combined["Ratio"] = combined["Ratio"].replace([np.inf, -np.inf], 0)
    
    # Label dependency level as high, medium, or low
    bins = [0, 0.75, 2.0, float('inf')]
    labels = ['Low', 'Medium', 'High']
    combined['Dependency Level'] = pd.cut(combined['Ratio'], bins=bins, labels=labels, include_lowest=True)
    
    # Encode the dependency level as numeric
    le = LabelEncoder()
    combined["Encoded Label"] = le.fit_transform(combined["Dependency Level"])

    # Define features and target
    categorical_features = ["State", "Fiscal year", "Fiscal quarter"]
    numerical_features = ["Ratio"]
    X = combined[categorical_features + numerical_features]
    y = combined["Encoded Label"]

    X.to_csv("models/X_train_import_dependency.csv", index=False)
    y.to_csv("models/y_train_import_dependency.csv", index=False)

    # Preprocessing: One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),  # One-hot encode "State", "Fiscal year", "Fiscal quarter"
        ],
        remainder="passthrough"  # Keep numerical features as is
    )

    # Define pipeline with preprocessing and model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Train the model
    model.fit(X, y)

    # Save the model and label encoder
    with open("models/import_dependency.joblib", "wb") as f:
        joblib.dump(model, f)
    with open("models/import_dependency_label_encoder.joblib", "wb") as f:
        joblib.dump(le, f)

# Function 2: Analyze seasonal fluctuations by category
def analyze_seasonal_fluctuations(data, trade_type):
    import joblib
    from xgboost import XGBRegressor
    import pandas as pd
    
    def get_season(quarter):
        seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        return seasons.get(quarter, 'Unknown')
    
    # Add 'Season' and 'Year' to the dataset
    data["Season"] = data["Fiscal quarter"].apply(get_season)
    
    # Aggregate by state, category, year, quarter, and season
    seasonal_data = data.groupby(["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"])["Dollar value"].sum().reset_index()
    
    # Create dummies for categorical features
    seasonal_data_dummies = pd.get_dummies(seasonal_data[["State", "Commodity name", "Fiscal year", "Fiscal quarter", "Season"]], drop_first=True)
    
    seasonal_data_values = seasonal_data["Dollar value"].values  # Extract the values column separately
    
    # Features (X) and target (y)
    X = seasonal_data_dummies
    y = seasonal_data_values

    X.to_csv(f"models/X_train_seasonal_{trade_type.lower()}.csv", index=False)
    y.to_csv(f"models/y_train_seasonal_{trade_type.lower()}.csv", index=False)
    
    # Define and train the XGBoost model
    model = XGBRegressor(random_state=30)
    model.fit(X, y)
    
    # Save the model and the feature column names used during training
    filename = f"models/seasonal_{trade_type.lower()}.joblib"
    with open(filename, "wb") as f:
        joblib.dump(model, f, compress=9)
    
    # Save the column names used for training (for future reference)
    columns_filename = f"models/seasonal_columns_{trade_type.lower()}.joblib"
    with open(columns_filename, "wb") as f:
        joblib.dump(seasonal_data_dummies.columns, f)

# Main entry for testing
if __name__ == "__main__":
    # Load training data
    imports_train = pd.read_csv("data_processed/import_train.csv")
    exports_train = pd.read_csv("data_processed/export_train.csv")
    
    # Call function 1
    classify_import_dependency(imports_train, exports_train)
    
    # Call function 2 for imports and exports separately
    analyze_seasonal_fluctuations(imports_train, "Imports")
    #analyze_seasonal_fluctuations(exports_train, "Exports")
